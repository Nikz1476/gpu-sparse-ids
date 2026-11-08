#pragma once
#include "common/types.h"
#include "features/ngram_features.h"
#include "sparse/csr_matrix.h"
#include <vector>

namespace ids {

class CPUEngine {
public:
    // Run the full sequential detection pipeline.
    // Optionally returns CSR matrices P, S, C for analysis/reporting.
    static std::vector<Alert> run_sequential(
        const std::vector<Packet>& packets,
        const std::vector<Signature>& signatures,
        const NGramFeatureExtractor& extractor,
        TimingResult& timing,
        CSRMatrix* out_P = nullptr,
        CSRMatrix* out_S = nullptr,
        CSRMatrix* out_C = nullptr);

    static std::vector<std::vector<int>> extract_all_features(
        const std::vector<Packet>& packets,
        const NGramFeatureExtractor& extractor);

    static std::vector<std::vector<int>> extract_signature_features(
        std::vector<Signature>& signatures,
        const NGramFeatureExtractor& extractor);

    // Generate candidate pairs from SpMM result matrix C.
    // Short signatures (pattern < ngram_size) bypass the sparse filter
    // and are paired with every packet for direct verification.
    static std::vector<CandidatePair> generate_candidates(
        const CSRMatrix& C,
        const std::vector<Signature>& signatures,
        const NGramFeatureExtractor& extractor,
        int num_packets);

    static std::vector<Alert> verify_candidates(
        const std::vector<CandidatePair>& candidates,
        const std::vector<Packet>& packets,
        const std::vector<Signature>& signatures);
};

} // namespace ids
