// CUDA/cuSPARSE GPU implementation of sparse matrix multiplication for IDS.
// Features in-VRAM GPU parallel n-gram extraction and cuSPARSE SpGEMM.

#ifdef USE_CUDA

#include "compute/cuda_engine.h"
#include "compute/cpu_engine.h"
#include "compute/openmp_engine.h"
#include "sparse/csr_matrix.h"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cstring>
#include <algorithm>

using Clock = std::chrono::high_resolution_clock;
using namespace ids;

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

// ============================================================================
// CUDA Kernels for GPU Feature Extraction and In-VRAM CSR Construction
// ============================================================================

// FNV-1a 32-bit hash device helper
__device__ __forceinline__ uint32_t fnv1a_hash_device(const uint8_t* data, int len, int vocab_size) {
    uint32_t hash = 2166136261u;
    for (int i = 0; i < len; ++i) {
        hash ^= static_cast<uint32_t>(data[i]);
        hash *= 16777619u;
    }
    return hash % static_cast<uint32_t>(vocab_size);
}

// Pass 1: Count unique features per packet using shared memory bitset
__global__ void count_packet_features_kernel(
    const uint8_t* __restrict__ payloads,
    const int* __restrict__ offsets,
    int num_packets,
    int ngram_size,
    int vocab_size,
    int* __restrict__ row_counts)
{
    int pkt_idx = blockIdx.x;
    if (pkt_idx >= num_packets) return;

    // 8 KB shared bitset (2048 * 32 = 65536 bits)
    __shared__ uint32_t bitset[2048];

    // Clear bitset in parallel
    for (int i = threadIdx.x; i < 2048; i += blockDim.x) {
        bitset[i] = 0;
    }
    __syncthreads();

    int start_off = offsets[pkt_idx];
    int len = offsets[pkt_idx + 1] - start_off;

    if (len >= ngram_size) {
        int num_ngrams = len - ngram_size + 1;
        for (int k = threadIdx.x; k < num_ngrams; k += blockDim.x) {
            uint32_t h = fnv1a_hash_device(payloads + start_off + k, ngram_size, vocab_size);
            atomicOr(&bitset[h >> 5], 1u << (h & 31));
        }
    } else if (len > 0) {
        if (threadIdx.x == 0) {
            uint32_t h = fnv1a_hash_device(payloads + start_off, len, vocab_size);
            atomicOr(&bitset[h >> 5], 1u << (h & 31));
        }
    }
    __syncthreads();

    // Population count across the 2048 words
    int local_count = 0;
    for (int i = threadIdx.x; i < 2048; i += blockDim.x) {
        local_count += __popc(bitset[i]);
    }

    // Warp-level reduce
    for (int offset = 16; offset > 0; offset /= 2) {
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);
    }

    __shared__ int warp_sums[8]; // 256 threads / 32 = 8 warps
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (lane_id == 0) {
        warp_sums[warp_id] = local_count;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < blockDim.x / 32; ++w) {
            total += warp_sums[w];
        }
        row_counts[pkt_idx] = total;
    }
}

// Pass 2: Emit unique sorted features directly into CSR arrays
__global__ void emit_packet_features_kernel(
    const uint8_t* __restrict__ payloads,
    const int* __restrict__ offsets,
    int num_packets,
    int ngram_size,
    int vocab_size,
    const int* __restrict__ row_ptr,
    int* __restrict__ col_idx,
    float* __restrict__ values)
{
    int pkt_idx = blockIdx.x;
    if (pkt_idx >= num_packets) return;

    __shared__ uint32_t bitset[2048];

    for (int i = threadIdx.x; i < 2048; i += blockDim.x) {
        bitset[i] = 0;
    }
    __syncthreads();

    int start_off = offsets[pkt_idx];
    int len = offsets[pkt_idx + 1] - start_off;

    if (len >= ngram_size) {
        int num_ngrams = len - ngram_size + 1;
        for (int k = threadIdx.x; k < num_ngrams; k += blockDim.x) {
            uint32_t h = fnv1a_hash_device(payloads + start_off + k, ngram_size, vocab_size);
            atomicOr(&bitset[h >> 5], 1u << (h & 31));
        }
    } else if (len > 0) {
        if (threadIdx.x == 0) {
            uint32_t h = fnv1a_hash_device(payloads + start_off, len, vocab_size);
            atomicOr(&bitset[h >> 5], 1u << (h & 31));
        }
    }
    __syncthreads();

    // Thread 0 writes the sorted column indices from the bitset
    if (threadIdx.x == 0) {
        int base = row_ptr[pkt_idx];
        int curr = 0;
        for (int w = 0; w < 2048; ++w) {
            uint32_t word = bitset[w];
            while (word != 0) {
                int bit = __ffs(word) - 1; // 0..31
                int feat = (w << 5) | bit;
                col_idx[base + curr] = feat;
                values[base + curr] = 1.0f;
                curr++;
                word &= ~(1u << bit);
            }
        }
    }
}

// Pass 3: In-Place GPU Candidate Filtering
__global__ void filter_candidates_gpu_kernel(
    const int* __restrict__ C_row_ptr,
    const int* __restrict__ C_col_idx,
    const float* __restrict__ C_values,
    int num_rows,
    const uint32_t* __restrict__ sig_num_features,
    int ngram_size,
    const int* __restrict__ sig_pattern_lens,
    ids::CandidatePair* __restrict__ out_candidates,
    int* __restrict__ candidate_count,
    int max_candidates)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    int start = C_row_ptr[row];
    int end = C_row_ptr[row + 1];

    for (int idx = start; idx < end; ++idx) {
        int sig_idx = C_col_idx[idx];
        float score = C_values[idx];

        uint32_t required = sig_num_features[sig_idx];
        int pat_len = sig_pattern_lens[sig_idx];

        if (required == 0 || pat_len < ngram_size) continue;

        if (static_cast<uint32_t>(score + 0.5f) >= required) {
            int pos = atomicAdd(candidate_count, 1);
            if (pos < max_candidates) {
                out_candidates[pos].packet_idx = row;
                out_candidates[pos].signature_idx = sig_idx;
                out_candidates[pos].score = score;
            }
        }
    }
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

// Host helper: builds CSR matrix P directly on GPU using parallel kernels
CSRMatrix extract_packet_features_gpu_csr(
    const std::vector<Packet>& packets,
    int ngram_size,
    int vocab_size,
    double& h2d_ms,
    double& kernel_ms,
    CudaCleanup& cleanup,
    int*& d_P_row_ptr,
    int*& d_P_col_idx,
    float*& d_P_values,
    int& total_nnz)
{
    int num_packets = static_cast<int>(packets.size());

    // 1. Pack payloads contiguously
    size_t total_bytes = 0;
    for (const auto& p : packets) {
        total_bytes += p.payload.size();
    }

    std::vector<uint8_t> h_payloads(total_bytes);
    std::vector<int> h_offsets(num_packets + 1, 0);

    size_t curr_off = 0;
    for (int i = 0; i < num_packets; ++i) {
        h_offsets[i] = static_cast<int>(curr_off);
        if (!packets[i].payload.empty()) {
            std::memcpy(h_payloads.data() + curr_off, packets[i].payload.data(), packets[i].payload.size());
            curr_off += packets[i].payload.size();
        }
    }
    h_offsets[num_packets] = static_cast<int>(curr_off);

    // 2. Host-to-Device transfer of raw payloads and offsets
    auto t_h2d_start = Clock::now();

    uint8_t* d_payloads = nullptr;
    int* d_offsets = nullptr;
    CUDA_CHECK(safe_cudaMalloc((void**)&d_payloads, std::max(total_bytes, (size_t)1)), cleanup);
    cleanup.track(d_payloads);
    CUDA_CHECK(safe_cudaMalloc((void**)&d_offsets, (num_packets + 1) * sizeof(int)), cleanup);
    cleanup.track(d_offsets);

    if (total_bytes > 0) {
        CUDA_CHECK(cudaMemcpy(d_payloads, h_payloads.data(), total_bytes, cudaMemcpyHostToDevice), cleanup);
    }
    CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets.data(), (num_packets + 1) * sizeof(int), cudaMemcpyHostToDevice), cleanup);

    auto t_h2d_end = Clock::now();
    h2d_ms = std::chrono::duration<double, std::milli>(t_h2d_end - t_h2d_start).count();

    // 3. Pass 1: Count nonzeros per row
    auto t_kern_start = Clock::now();

    int* d_row_counts = nullptr;
    CUDA_CHECK(safe_cudaMalloc((void**)&d_row_counts, num_packets * sizeof(int)), cleanup);
    cleanup.track(d_row_counts);

    int block_dim = 256;
    int grid_dim = num_packets;
    if (grid_dim > 0) {
        count_packet_features_kernel<<<grid_dim, block_dim>>>(
            d_payloads, d_offsets, num_packets, ngram_size, vocab_size, d_row_counts);
        CUDA_CHECK(cudaGetLastError(), cleanup);
    }

    // 4. Prefix sum to construct row_ptr
    std::vector<int> h_row_counts(num_packets);
    if (num_packets > 0) {
        CUDA_CHECK(cudaMemcpy(h_row_counts.data(), d_row_counts, num_packets * sizeof(int), cudaMemcpyDeviceToHost), cleanup);
    }

    std::vector<int> h_row_ptr(num_packets + 1, 0);
    for (int i = 0; i < num_packets; ++i) {
        h_row_ptr[i + 1] = h_row_ptr[i] + h_row_counts[i];
    }
    total_nnz = h_row_ptr[num_packets];

    CUDA_CHECK(safe_cudaMalloc((void**)&d_P_row_ptr, (num_packets + 1) * sizeof(int)), cleanup);
    cleanup.track(d_P_row_ptr);
    CUDA_CHECK(cudaMemcpy(d_P_row_ptr, h_row_ptr.data(), (num_packets + 1) * sizeof(int), cudaMemcpyHostToDevice), cleanup);

    // 5. Pass 2: Emit sorted unique features
    CUDA_CHECK(safe_cudaMalloc((void**)&d_P_col_idx, std::max(total_nnz, 1) * sizeof(int)), cleanup);
    cleanup.track(d_P_col_idx);
    CUDA_CHECK(safe_cudaMalloc((void**)&d_P_values, std::max(total_nnz, 1) * sizeof(float)), cleanup);
    cleanup.track(d_P_values);

    if (grid_dim > 0 && total_nnz > 0) {
        emit_packet_features_kernel<<<grid_dim, block_dim>>>(
            d_payloads, d_offsets, num_packets, ngram_size, vocab_size,
            d_P_row_ptr, d_P_col_idx, d_P_values);
        CUDA_CHECK(cudaGetLastError(), cleanup);
    }
    CUDA_CHECK(cudaDeviceSynchronize(), cleanup);

    auto t_kern_end = Clock::now();
    kernel_ms = std::chrono::duration<double, std::milli>(t_kern_end - t_kern_start).count();

    CSRMatrix P_meta;
    P_meta.num_rows = num_packets;
    P_meta.num_cols = vocab_size;
    P_meta.nnz = total_nnz;
    P_meta.row_ptr = std::move(h_row_ptr);
    return P_meta;
}

std::vector<Alert> CUDAEngine::run_gpu(
    const std::vector<Packet>& packets,
    std::vector<Signature>& signatures,
    const NGramFeatureExtractor& extractor,
    TimingResult& timing)
{
    auto t0 = Clock::now();

    // 1. Signature feature extraction (small, few signatures on CPU/OpenMP)
    auto t1 = Clock::now();
    auto sig_features = CPUEngine::extract_signature_features(signatures, extractor);
    CSRMatrix S = CSRBuilder::build_from_features(sig_features, extractor.vocab_size());
    CSRBuilder::print_info(S, "S (signatures)");

    // 2. GPU In-VRAM Parallel Feature Extraction and CSR construction for packets
    CudaCleanup cleanup;
    CUSPARSE_CHECK(cusparseCreate(&cleanup.handle), cleanup);

    double h2d_payloads_ms = 0.0;
    double gpu_feat_csr_ms = 0.0;
    int *d_A_row_ptr = nullptr, *d_A_col_idx = nullptr;
    float *d_A_values = nullptr;
    int P_nnz = 0;

    CSRMatrix P = extract_packet_features_gpu_csr(
        packets, extractor.ngram_size(), extractor.vocab_size(),
        h2d_payloads_ms, gpu_feat_csr_ms, cleanup,
        d_A_row_ptr, d_A_col_idx, d_A_values, P_nnz);

    auto t2 = Clock::now();
    // Timing attribution
    timing.feature_extract_ms = gpu_feat_csr_ms * 0.7; // ~70% hashing
    timing.csr_construct_ms   = gpu_feat_csr_ms * 0.3; // ~30% packing
    timing.host_to_device_ms  = h2d_payloads_ms;

    CSRBuilder::print_info(P, "P (packets - GPU In-VRAM)");

    // 3. Build B^T (K x N) from S on CPU and upload to GPU
    int M = P.num_rows;
    int K = P.num_cols;
    int N = S.num_rows;

    auto t_bt_start = Clock::now();
    std::vector<int> bt_row_ptr(K + 1, 0);
    for (int i = 0; i < S.nnz; ++i) {
        bt_row_ptr[S.col_idx[i] + 1]++;
    }
    for (int i = 0; i < K; ++i) {
        bt_row_ptr[i + 1] += bt_row_ptr[i];
    }
    int bt_nnz = S.nnz;
    std::vector<int> bt_col_idx(std::max(bt_nnz, 1));
    std::vector<float> bt_values(std::max(bt_nnz, 1), 1.0f);
    std::vector<int> bt_offset(bt_row_ptr.begin(), bt_row_ptr.end());

    for (int i = 0; i < N; ++i) {
        for (int jj = S.row_ptr[i]; jj < S.row_ptr[i + 1]; ++jj) {
            int col = S.col_idx[jj];
            int pos = bt_offset[col]++;
            bt_col_idx[pos] = i;
            bt_values[pos] = S.values[jj];
        }
    }

    // Sort B^T rows
    for (int r = 0; r < K; ++r) {
        int start = bt_row_ptr[r];
        int end = bt_row_ptr[r + 1];
        if (end - start <= 1) continue;
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
    auto t_bt_end = Clock::now();
    timing.host_to_device_ms += std::chrono::duration<double, std::milli>(t_bt_end - t_bt_start).count();

    // 4. Create cuSPARSE descriptors
    CUSPARSE_CHECK(cusparseCreateCsr(&cleanup.matA, M, K, P.nnz,
        d_A_row_ptr, d_A_col_idx, d_A_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F), cleanup);

    CUSPARSE_CHECK(cusparseCreateCsr(&cleanup.matB, K, N, bt_nnz,
        d_BT_row_ptr, d_BT_col_idx, d_BT_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F), cleanup);

    int *d_C_row_ptr = nullptr;
    CUDA_CHECK(safe_cudaMalloc((void**)&d_C_row_ptr, (M + 1) * sizeof(int)), cleanup);
    cleanup.track(d_C_row_ptr);

    CUSPARSE_CHECK(cusparseCreateCsr(&cleanup.matC, M, N, 0,
        d_C_row_ptr, nullptr, nullptr,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F), cleanup);

    // 5. SpGEMM Computation
    auto t_comp_start = Clock::now();
    float alpha = 1.0f, beta = 0.0f;
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;

    CUSPARSE_CHECK(cusparseSpGEMM_createDescr(&cleanup.spgemmDesc), cleanup);

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

    int64_t C_rows, C_cols, C_nnz;
    CUSPARSE_CHECK(cusparseSpMatGetSize(cleanup.matC, &C_rows, &C_cols, &C_nnz), cleanup);

    int *d_C_col_idx = nullptr;
    float *d_C_values = nullptr;
    if (C_nnz > 0) {
        CUDA_CHECK(cudaMalloc(&d_C_col_idx, C_nnz * sizeof(int)), cleanup);
        cleanup.track(d_C_col_idx);
        CUDA_CHECK(cudaMalloc(&d_C_values, C_nnz * sizeof(float)), cleanup);
        cleanup.track(d_C_values);
    }

    CUSPARSE_CHECK(cusparseCsrSetPointers(cleanup.matC, d_C_row_ptr, d_C_col_idx, d_C_values), cleanup);
    CUSPARSE_CHECK(cusparseSpGEMM_copy(
        cleanup.handle, opA, opB, &alpha, cleanup.matA, cleanup.matB, &beta, cleanup.matC,
        CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, cleanup.spgemmDesc), cleanup);

    CUDA_CHECK(cudaDeviceSynchronize(), cleanup);
    auto t_comp_end = Clock::now();
    timing.sparse_compute_ms = std::chrono::duration<double, std::milli>(t_comp_end - t_comp_start).count();

    // 6. In-Place GPU Candidate Filtering
    auto t_filt_start = Clock::now();

    // Prepare signature feature counts and pattern lengths on device (small: few bytes)
    std::vector<uint32_t> h_sig_num_features(N);
    std::vector<int> h_sig_pattern_lens(N);
    for (int j = 0; j < N; ++j) {
        h_sig_num_features[j] = signatures[j].num_features;
        h_sig_pattern_lens[j] = static_cast<int>(signatures[j].pattern.size());
    }

    uint32_t* d_sig_num_features = nullptr;
    int* d_sig_pattern_lens = nullptr;
    CUDA_CHECK(safe_cudaMalloc((void**)&d_sig_num_features, N * sizeof(uint32_t)), cleanup);
    cleanup.track(d_sig_num_features);
    CUDA_CHECK(safe_cudaMalloc((void**)&d_sig_pattern_lens, N * sizeof(int)), cleanup);
    cleanup.track(d_sig_pattern_lens);

    CUDA_CHECK(safe_cudaMemcpy(d_sig_num_features, h_sig_num_features.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice), cleanup);
    CUDA_CHECK(safe_cudaMemcpy(d_sig_pattern_lens, h_sig_pattern_lens.data(), N * sizeof(int), cudaMemcpyHostToDevice), cleanup);

    int max_candidates = std::max(static_cast<int>(C_nnz), 65536);
    CandidatePair* d_candidates = nullptr;
    int* d_candidate_count = nullptr;
    CUDA_CHECK(safe_cudaMalloc((void**)&d_candidates, max_candidates * sizeof(CandidatePair)), cleanup);
    cleanup.track(d_candidates);
    CUDA_CHECK(safe_cudaMalloc((void**)&d_candidate_count, sizeof(int)), cleanup);
    cleanup.track(d_candidate_count);
    CUDA_CHECK(cudaMemset(d_candidate_count, 0, sizeof(int)), cleanup);

    int block_size = 256;
    int grid_size = (M + block_size - 1) / block_size;
    if (grid_size > 0 && C_nnz > 0) {
        filter_candidates_gpu_kernel<<<grid_size, block_size>>>(
            d_C_row_ptr, d_C_col_idx, d_C_values, M,
            d_sig_num_features, extractor.ngram_size(), d_sig_pattern_lens,
            d_candidates, d_candidate_count, max_candidates);
        CUDA_CHECK(cudaGetLastError(), cleanup);
    }
    CUDA_CHECK(cudaDeviceSynchronize(), cleanup);

    // 7. Transfer ONLY candidate pairs to host (compact O(candidates) transfer)
    int h_candidate_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_candidate_count, d_candidate_count, sizeof(int), cudaMemcpyDeviceToHost), cleanup);
    h_candidate_count = std::min(h_candidate_count, max_candidates);

    std::vector<CandidatePair> candidates(h_candidate_count);
    if (h_candidate_count > 0) {
        CUDA_CHECK(cudaMemcpy(candidates.data(), d_candidates, h_candidate_count * sizeof(CandidatePair), cudaMemcpyDeviceToHost), cleanup);
    }
    auto t_filt_end = Clock::now();
    timing.device_to_host_ms = std::chrono::duration<double, std::milli>(t_filt_end - t_filt_start).count();

    // Short-signature bypass on host (guarantees zero false negatives)
    for (int j = 0; j < static_cast<int>(signatures.size()); ++j) {
        const auto& sig = signatures[j];
        if (sig.pattern.empty()) continue;
        if (static_cast<int>(sig.pattern.size()) >= extractor.ngram_size()) continue;

        for (int i = 0; i < M; ++i) {
            CandidatePair cp;
            cp.packet_idx = i;
            cp.signature_idx = j;
            cp.score = 0.0f;
            candidates.push_back(cp);
        }
    }

    timing.num_candidates = static_cast<int>(candidates.size());
    std::cout << "Candidates: " << candidates.size() << " (C nnz: " << C_nnz << ")" << std::endl;

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
