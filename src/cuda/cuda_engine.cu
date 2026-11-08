// CUDA/cuSPARSE GPU implementation of sparse matrix multiplication for IDS.
// Compile with: nvcc -DUSE_CUDA and link against cuSPARSE.

#ifdef USE_CUDA

#include "compute/cuda_engine.h"
#include "compute/cpu_engine.h"
#include "sparse/csr_matrix.h"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cstring>
#include <algorithm>

using Clock = std::chrono::high_resolution_clock;

// RAII cleanup helper to prevent GPU memory leaks on error paths
struct CudaCleanup {
    std::vector<void*> device_ptrs;
    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t matA = nullptr, matB = nullptr, matC = nullptr;
    cusparseSpGEMMDescr_t spgemmDesc = nullptr;

    ~CudaCleanup() {
        if (spgemmDesc) cusparseSpGEMM_destroyDescr(spgemmDesc);
        if (matA) cusparseDestroySpMat(matA);
        if (matB) cusparseDestroySpMat(matB);
        if (matC) cusparseDestroySpMat(matC);
        if (handle) cusparseDestroy(handle);
        for (void* p : device_ptrs) {
            if (p) cudaFree(p);
        }
    }

    void track(void* p) { device_ptrs.push_back(p); }
};

#define CUDA_CHECK(call, cleanup_obj)                                        \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << " — " << cudaGetErrorString(err) << std::endl;      \
            return {};                                                       \
        }                                                                    \
    } while (0)

#define CUSPARSE_CHECK(call, cleanup_obj)                                    \
    do {                                                                     \
        cusparseStatus_t st = call;                                          \
        if (st != CUSPARSE_STATUS_SUCCESS) {                                 \
            std::cerr << "cuSPARSE error at " << __FILE__ << ":" << __LINE__ \
                      << " — code " << static_cast<int>(st) << std::endl;    \
            return {};                                                       \
        }                                                                    \
    } while (0)

// Safe cudaMalloc that handles zero-size allocations
static cudaError_t safe_cudaMalloc(void** ptr, size_t size) {
    if (size == 0) {
        *ptr = nullptr;
        return cudaSuccess;
    }
    return cudaMalloc(ptr, size);
}

// Safe cudaMemcpy that handles zero-size copies
static cudaError_t safe_cudaMemcpy(void* dst, const void* src, size_t size,
                                   cudaMemcpyKind kind) {
    if (size == 0) return cudaSuccess;
    return cudaMemcpy(dst, src, size, kind);
}

namespace ids {

bool CUDAEngine::is_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

CSRMatrix CUDAEngine::spmm_transpose_gpu(
    const CSRMatrix& A, const CSRMatrix& B,
    double& h2d_ms, double& compute_ms, double& d2h_ms)
{
    // C = A * B^T using cuSPARSE SpGEMM
    // We compute B^T on CPU, then do C = A * B^T on GPU.

    int M = A.num_rows;     // packets
    int K = A.num_cols;     // vocab
    int N = B.num_rows;     // signatures

    CudaCleanup cleanup;

    CUSPARSE_CHECK(cusparseCreate(&cleanup.handle), cleanup);

    // --- Host to Device transfer ---
    auto t_h2d_start = Clock::now();

    // A (M x K) CSR
    int *d_A_row_ptr = nullptr, *d_A_col_idx = nullptr;
    float *d_A_values = nullptr;
    CUDA_CHECK(safe_cudaMalloc((void**)&d_A_row_ptr, (M + 1) * sizeof(int)), cleanup);
    cleanup.track(d_A_row_ptr);
    CUDA_CHECK(safe_cudaMalloc((void**)&d_A_col_idx, std::max(A.nnz, 1) * sizeof(int)), cleanup);
    cleanup.track(d_A_col_idx);
    CUDA_CHECK(safe_cudaMalloc((void**)&d_A_values, std::max(A.nnz, 1) * sizeof(float)), cleanup);
    cleanup.track(d_A_values);

    CUDA_CHECK(safe_cudaMemcpy(d_A_row_ptr, A.row_ptr.data(), (M + 1) * sizeof(int), cudaMemcpyHostToDevice), cleanup);
    if (A.nnz > 0) {
        CUDA_CHECK(cudaMemcpy(d_A_col_idx, A.col_idx.data(), A.nnz * sizeof(int), cudaMemcpyHostToDevice), cleanup);
        CUDA_CHECK(cudaMemcpy(d_A_values, A.values.data(), A.nnz * sizeof(float), cudaMemcpyHostToDevice), cleanup);
    }

    // Build B^T (K x N) from B (N x K) on CPU
    std::vector<int> bt_row_ptr(K + 1, 0);
    for (int i = 0; i < B.nnz; ++i) {
        bt_row_ptr[B.col_idx[i] + 1]++;
    }
    for (int i = 0; i < K; ++i) {
        bt_row_ptr[i + 1] += bt_row_ptr[i];
    }
    int bt_nnz = B.nnz;
    std::vector<int> bt_col_idx(std::max(bt_nnz, 1));
    std::vector<float> bt_values(std::max(bt_nnz, 1), 1.0f);
    std::vector<int> bt_offset(bt_row_ptr.begin(), bt_row_ptr.end());

    for (int i = 0; i < N; ++i) {
        for (int jj = B.row_ptr[i]; jj < B.row_ptr[i + 1]; ++jj) {
            int col = B.col_idx[jj];
            int pos = bt_offset[col]++;
            bt_col_idx[pos] = i;
            bt_values[pos] = B.values[jj];
        }
    }

    // Sort each row of B^T by column index (cuSPARSE requires sorted CSR)
    for (int r = 0; r < K; ++r) {
        int start = bt_row_ptr[r];
        int end = bt_row_ptr[r + 1];
        if (end - start <= 1) continue;
        // Simple insertion sort (rows are small)
        for (int i = start + 1; i < end; ++i) {
            int key_col = bt_col_idx[i];
            float key_val = bt_values[i];
            int j = i - 1;
            while (j >= start && bt_col_idx[j] > key_col) {
                bt_col_idx[j + 1] = bt_col_idx[j];
                bt_values[j + 1] = bt_values[j];
                --j;
            }
            bt_col_idx[j + 1] = key_col;
            bt_values[j + 1] = key_val;
        }
    }

    int *d_BT_row_ptr = nullptr, *d_BT_col_idx = nullptr;
    float *d_BT_values = nullptr;
    CUDA_CHECK(safe_cudaMalloc((void**)&d_BT_row_ptr, (K + 1) * sizeof(int)), cleanup);
    cleanup.track(d_BT_row_ptr);
    CUDA_CHECK(safe_cudaMalloc((void**)&d_BT_col_idx, std::max(bt_nnz, 1) * sizeof(int)), cleanup);
    cleanup.track(d_BT_col_idx);
    CUDA_CHECK(safe_cudaMalloc((void**)&d_BT_values, std::max(bt_nnz, 1) * sizeof(float)), cleanup);
    cleanup.track(d_BT_values);

    CUDA_CHECK(safe_cudaMemcpy(d_BT_row_ptr, bt_row_ptr.data(), (K + 1) * sizeof(int), cudaMemcpyHostToDevice), cleanup);
    if (bt_nnz > 0) {
        CUDA_CHECK(cudaMemcpy(d_BT_col_idx, bt_col_idx.data(), bt_nnz * sizeof(int), cudaMemcpyHostToDevice), cleanup);
        CUDA_CHECK(cudaMemcpy(d_BT_values, bt_values.data(), bt_nnz * sizeof(float), cudaMemcpyHostToDevice), cleanup);
    }

    auto t_h2d_end = Clock::now();
    h2d_ms = std::chrono::duration<double, std::milli>(t_h2d_end - t_h2d_start).count();

    // --- Create sparse matrix descriptors ---
    CUSPARSE_CHECK(cusparseCreateCsr(&cleanup.matA, M, K, A.nnz,
        d_A_row_ptr, d_A_col_idx, d_A_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F), cleanup);

    CUSPARSE_CHECK(cusparseCreateCsr(&cleanup.matB, K, N, bt_nnz,
        d_BT_row_ptr, d_BT_col_idx, d_BT_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F), cleanup);

    // C: M x N (output)
    int *d_C_row_ptr = nullptr;
    CUDA_CHECK(safe_cudaMalloc((void**)&d_C_row_ptr, (M + 1) * sizeof(int)), cleanup);
    cleanup.track(d_C_row_ptr);

    CUSPARSE_CHECK(cusparseCreateCsr(&cleanup.matC, M, N, 0,
        d_C_row_ptr, nullptr, nullptr,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F), cleanup);

    // --- SpGEMM computation ---
    auto t_comp_start = Clock::now();

    float alpha = 1.0f, beta = 0.0f;
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;

    CUSPARSE_CHECK(cusparseSpGEMM_createDescr(&cleanup.spgemmDesc), cleanup);

    // Work estimation
    size_t bufferSize1 = 0;
    void* buffer1 = nullptr;
    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(
        cleanup.handle, opA, opB, &alpha, cleanup.matA, cleanup.matB, &beta, cleanup.matC,
        CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
        cleanup.spgemmDesc, &bufferSize1, nullptr), cleanup);
    if (bufferSize1 > 0) {
        CUDA_CHECK(cudaMalloc(&buffer1, bufferSize1), cleanup);
        cleanup.track(buffer1);
    }
    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(
        cleanup.handle, opA, opB, &alpha, cleanup.matA, cleanup.matB, &beta, cleanup.matC,
        CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
        cleanup.spgemmDesc, &bufferSize1, buffer1), cleanup);

    // Compute
    size_t bufferSize2 = 0;
    void* buffer2 = nullptr;
    CUSPARSE_CHECK(cusparseSpGEMM_compute(
        cleanup.handle, opA, opB, &alpha, cleanup.matA, cleanup.matB, &beta, cleanup.matC,
        CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
        cleanup.spgemmDesc, &bufferSize2, nullptr), cleanup);
    if (bufferSize2 > 0) {
        CUDA_CHECK(cudaMalloc(&buffer2, bufferSize2), cleanup);
        cleanup.track(buffer2);
    }
    CUSPARSE_CHECK(cusparseSpGEMM_compute(
        cleanup.handle, opA, opB, &alpha, cleanup.matA, cleanup.matB, &beta, cleanup.matC,
        CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
        cleanup.spgemmDesc, &bufferSize2, buffer2), cleanup);

    // Get result dimensions
    int64_t C_rows, C_cols, C_nnz;
    CUSPARSE_CHECK(cusparseSpMatGetSize(cleanup.matC, &C_rows, &C_cols, &C_nnz), cleanup);

    // Allocate result arrays
    int *d_C_col_idx = nullptr;
    float *d_C_values = nullptr;
    if (C_nnz > 0) {
        CUDA_CHECK(cudaMalloc(&d_C_col_idx, C_nnz * sizeof(int)), cleanup);
        cleanup.track(d_C_col_idx);
        CUDA_CHECK(cudaMalloc(&d_C_values, C_nnz * sizeof(float)), cleanup);
        cleanup.track(d_C_values);
    }

    CUSPARSE_CHECK(cusparseCsrSetPointers(cleanup.matC, d_C_row_ptr, d_C_col_idx, d_C_values), cleanup);

    // Copy result
    CUSPARSE_CHECK(cusparseSpGEMM_copy(
        cleanup.handle, opA, opB, &alpha, cleanup.matA, cleanup.matB, &beta, cleanup.matC,
        CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, cleanup.spgemmDesc), cleanup);

    CUDA_CHECK(cudaDeviceSynchronize(), cleanup);
    auto t_comp_end = Clock::now();
    compute_ms = std::chrono::duration<double, std::milli>(t_comp_end - t_comp_start).count();

    // --- Device to Host ---
    auto t_d2h_start = Clock::now();

    CSRMatrix C_result;
    C_result.num_rows = M;
    C_result.num_cols = N;
    C_result.nnz = static_cast<int>(C_nnz);
    C_result.row_ptr.resize(M + 1);
    C_result.col_idx.resize(C_nnz);
    C_result.values.resize(C_nnz);

    CUDA_CHECK(cudaMemcpy(C_result.row_ptr.data(), d_C_row_ptr, (M + 1) * sizeof(int), cudaMemcpyDeviceToHost), cleanup);
    if (C_nnz > 0) {
        CUDA_CHECK(cudaMemcpy(C_result.col_idx.data(), d_C_col_idx, C_nnz * sizeof(int), cudaMemcpyDeviceToHost), cleanup);
        CUDA_CHECK(cudaMemcpy(C_result.values.data(), d_C_values, C_nnz * sizeof(float), cudaMemcpyDeviceToHost), cleanup);
    }

    auto t_d2h_end = Clock::now();
    d2h_ms = std::chrono::duration<double, std::milli>(t_d2h_end - t_d2h_start).count();

    // cleanup is handled by RAII destructor
    return C_result;
}

std::vector<Alert> CUDAEngine::run_gpu(
    const std::vector<Packet>& packets,
    std::vector<Signature>& signatures,
    const NGramFeatureExtractor& extractor,
    TimingResult& timing)
{
    auto t0 = Clock::now();

    // Feature extraction on CPU
    auto t1 = Clock::now();
    auto packet_features = CPUEngine::extract_all_features(packets, extractor);
    auto sig_features = CPUEngine::extract_signature_features(signatures, extractor);
    auto t2 = Clock::now();
    timing.feature_extract_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // CSR construction
    auto t3 = Clock::now();
    CSRMatrix P = CSRBuilder::build_from_features(packet_features, extractor.vocab_size());
    CSRMatrix S = CSRBuilder::build_from_features(sig_features, extractor.vocab_size());
    auto t4 = Clock::now();
    timing.csr_construct_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();

    CSRBuilder::print_info(P, "P (packets)");
    CSRBuilder::print_info(S, "S (signatures)");

    // GPU SpMM
    double h2d, comp, d2h;
    CSRMatrix C = spmm_transpose_gpu(P, S, h2d, comp, d2h);
    timing.host_to_device_ms = h2d;
    timing.sparse_compute_ms = comp;
    timing.device_to_host_ms = d2h;

    CSRBuilder::print_info(C, "C (scores)");

    // Candidate generation and verification on CPU
    auto candidates = CPUEngine::generate_candidates(C, signatures, extractor,
                                                     static_cast<int>(packets.size()));
    timing.num_candidates = static_cast<int>(candidates.size());
    std::cout << "Candidates: " << candidates.size() << std::endl;

    auto t5 = Clock::now();
    auto alerts = CPUEngine::verify_candidates(candidates, packets, signatures);
    auto t6 = Clock::now();
    timing.verification_ms = std::chrono::duration<double, std::milli>(t6 - t5).count();
    timing.num_alerts = static_cast<int>(alerts.size());

    timing.total_ms = std::chrono::duration<double, std::milli>(t6 - t0).count();
    return alerts;
}

} // namespace ids

#endif // USE_CUDA
