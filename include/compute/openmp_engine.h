#pragma once
#include "common/types.h"
#include "features/ngram_features.h"
#include "sparse/csr_matrix.h"
#include <vector>

namespace ids {

class OpenMPEngine {
public:
    // Run the full parallel detection pipeline.
    // Optionally returns CSR matrices P, S, C for analysis/reporting.
    static std::vector<Alert> run_parallel(
        const std::vector<Packet>& packets,
        std::vector<Signature>& signatures,
        const NGramFeatureExtractor& extractor,
        TimingResult& timing,
        int num_threads = 0,
        CSRMatrix* out_P = nullptr,
        CSRMatrix* out_S = nullptr,
        CSRMatrix* out_C = nullptr);

    static std::vector<std::vector<int>> extract_all_features_parallel(
        const std::vector<Packet>& packets,
        const NGramFeatureExtractor& extractor,
        int num_threads);

    static CSRMatrix spmm_transpose_parallel(
        const CSRMatrix& A, const CSRMatrix& B,
        int num_threads);

    static std::vector<Alert> verify_candidates_parallel(
        const std::vector<CandidatePair>& candidates,
        const std::vector<Packet>& packets,
        const std::vector<Signature>& signatures,
        int num_threads);
};

} // namespace ids
