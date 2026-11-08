#pragma once
#include "common/types.h"
#include "features/ngram_features.h"
#include <vector>

namespace ids {

class CUDAEngine {
public:
    static bool is_available();

    static std::vector<Alert> run_gpu(
        const std::vector<Packet>& packets,
        std::vector<Signature>& signatures,
        const NGramFeatureExtractor& extractor,
        TimingResult& timing);

    static CSRMatrix spmm_transpose_gpu(
        const CSRMatrix& A, const CSRMatrix& B,
        double& h2d_ms, double& compute_ms, double& d2h_ms);
};

} // namespace ids

