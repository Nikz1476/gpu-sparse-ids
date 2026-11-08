#include "compute/openmp_engine.h"
#include "compute/cpu_engine.h"
#include "sparse/csr_matrix.h"
#include <chrono>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <omp.h>

namespace ids {

using Clock = std::chrono::high_resolution_clock;
static double elapsed_ms(Clock::time_point s, Clock::time_point e) {
    return std::chrono::duration<double, std::milli>(e - s).count();
}

static bool contains_pattern_omp(const std::vector<uint8_t>& haystack,
                                 const std::vector<uint8_t>& needle) {
    if (needle.empty()) return true;
    if (haystack.size() < needle.size()) return false;
    size_t n = haystack.size(), m = needle.size();
    for (size_t i = 0; i <= n - m; ++i)
        if (memcmp(haystack.data() + i, needle.data(), m) == 0) return true;
    return false;
}

std::vector<std::vector<int>> OpenMPEngine::extract_all_features_parallel(
    const std::vector<Packet>& packets,
    const NGramFeatureExtractor& extractor,
    int num_threads)
{
    int n = static_cast<int>(packets.size());
    std::vector<std::vector<int>> features(n);

    #pragma omp parallel for schedule(dynamic, 64) num_threads(num_threads)
    for (int i = 0; i < n; ++i) {
        features[i] = extractor.extract_features(packets[i].payload);
    }
    return features;
}

CSRMatrix OpenMPEngine::spmm_transpose_parallel(
    const CSRMatrix& A, const CSRMatrix& B, int num_threads)
{
    int M = A.num_rows;
    int N = B.num_rows;

    // Per-row results (each row is independent — no data races)
    std::vector<std::vector<int>> row_cols(M);
    std::vector<std::vector<float>> row_vals(M);

    #pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads)
    for (int i = 0; i < M; ++i) {
        int a_start = A.row_ptr[i];
        int a_end   = A.row_ptr[i + 1];
        if (a_start == a_end) continue;

        for (int j = 0; j < N; ++j) {
            int b_start = B.row_ptr[j];
            int b_end   = B.row_ptr[j + 1];
            if (b_start == b_end) continue;

            float dot = 0.0f;
            int ai = a_start, bi = b_start;
            while (ai < a_end && bi < b_end) {
                if (A.col_idx[ai] == B.col_idx[bi]) {
                    dot += A.values[ai] * B.values[bi];
                    ++ai; ++bi;
                } else if (A.col_idx[ai] < B.col_idx[bi]) {
                    ++ai;
                } else {
                    ++bi;
                }
            }
            if (dot > 0.0f) {
                row_cols[i].push_back(j);
                row_vals[i].push_back(dot);
            }
        }
    }

    // Assemble CSR (single-threaded, safe)
    CSRMatrix C;
    C.num_rows = M;
    C.num_cols = N;
    C.row_ptr.resize(M + 1);
    C.row_ptr[0] = 0;

    int total_nnz = 0;
    for (int i = 0; i < M; ++i) {
        total_nnz += static_cast<int>(row_cols[i].size());
        C.row_ptr[i + 1] = total_nnz;
    }

    C.nnz = total_nnz;
    C.col_idx.resize(total_nnz);
    C.values.resize(total_nnz);

    for (int i = 0; i < M; ++i) {
        int start = C.row_ptr[i];
        for (size_t k = 0; k < row_cols[i].size(); ++k) {
            C.col_idx[start + k] = row_cols[i][k];
            C.values[start + k] = row_vals[i][k];
        }
    }

    return C;
}

std::vector<Alert> OpenMPEngine::verify_candidates_parallel(
    const std::vector<CandidatePair>& candidates,
    const std::vector<Packet>& packets,
    const std::vector<Signature>& signatures,
    int num_threads)
{
    int n = static_cast<int>(candidates.size());

    // Use actual thread count from runtime, not the requested count,
    // to avoid out-of-bounds on per_thread vector.
    int actual_threads = num_threads;

    // Collect per-thread results
    std::vector<std::vector<Alert>> per_thread(actual_threads);

    #pragma omp parallel num_threads(actual_threads)
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < n; ++i) {
            const auto& cp = candidates[i];
            const Packet& pkt = packets[cp.packet_idx];
            const Signature& sig = signatures[cp.signature_idx];

            if (sig.protocol != 0 && sig.protocol != pkt.protocol) continue;
            if (sig.dst_port != 0 && sig.dst_port != pkt.dst_port) continue;

            if (contains_pattern_omp(pkt.payload, sig.pattern)) {
                Alert alert;
                alert.packet_id = pkt.id;
                alert.signature_id = sig.id;
                alert.signature_name = sig.name;
                alert.src_ip = pkt.src_ip;
                alert.dst_ip = pkt.dst_ip;
                alert.src_port = pkt.src_port;
                alert.dst_port = pkt.dst_port;
                alert.timestamp = pkt.timestamp;
                alert.match_detail = "Exact match: " + sig.description;
                per_thread[tid].push_back(std::move(alert));
            }
        }
    }

    // Merge
    std::vector<Alert> alerts;
    for (auto& ta : per_thread)
        alerts.insert(alerts.end(), ta.begin(), ta.end());

    return alerts;
}

std::vector<Alert> OpenMPEngine::run_parallel(
    const std::vector<Packet>& packets,
    std::vector<Signature>& signatures,
    const NGramFeatureExtractor& extractor,
    TimingResult& timing,
    int num_threads)
{
    if (num_threads <= 0) num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    std::cout << "OpenMP threads: " << num_threads << std::endl;

    auto t0 = Clock::now();

    // 1. Parallel feature extraction
    auto t1 = Clock::now();
    auto packet_features = extract_all_features_parallel(packets, extractor, num_threads);

    // Signature features (usually small, sequential is fine)
    auto sig_features = CPUEngine::extract_signature_features(signatures, extractor);
    auto t2 = Clock::now();
    timing.feature_extract_ms = elapsed_ms(t1, t2);

    // 2. CSR construction
    auto t3 = Clock::now();
    CSRMatrix P = CSRBuilder::build_from_features(packet_features, extractor.vocab_size());
    CSRMatrix S = CSRBuilder::build_from_features(sig_features, extractor.vocab_size());
    auto t4 = Clock::now();
    timing.csr_construct_ms = elapsed_ms(t3, t4);

    CSRBuilder::print_info(P, "P (packets)");
    CSRBuilder::print_info(S, "S (signatures)");

    // 3. Parallel SpMM
    auto t5 = Clock::now();
    CSRMatrix C = spmm_transpose_parallel(P, S, num_threads);
    auto t6 = Clock::now();
    timing.sparse_compute_ms = elapsed_ms(t5, t6);

    CSRBuilder::print_info(C, "C (scores)");

    // 4. Generate candidates (includes short-signature bypass)
    auto candidates = CPUEngine::generate_candidates(C, signatures, extractor,
                                                     static_cast<int>(packets.size()));
    timing.num_candidates = static_cast<int>(candidates.size());
    std::cout << "Candidates: " << candidates.size() << std::endl;

    // 5. Parallel verification
    auto t7 = Clock::now();
    auto alerts = verify_candidates_parallel(candidates, packets, signatures, num_threads);
    auto t8 = Clock::now();
    timing.verification_ms = elapsed_ms(t7, t8);
    timing.num_alerts = static_cast<int>(alerts.size());

    timing.total_ms = elapsed_ms(t0, t8);
    return alerts;
}

} // namespace ids
