// CUDA engine stub — compiles on systems without CUDA.
// The actual GPU implementation is in src/cuda/cuda_engine.cu
// and requires nvcc + cuSPARSE.

#ifndef USE_CUDA

#include "compute/cuda_engine.h"
#include <iostream>

namespace ids {

bool CUDAEngine::is_available() {
    return false;
}

std::vector<Alert> CUDAEngine::run_gpu(
    const std::vector<Packet>&,
    std::vector<Signature>&,
    const NGramFeatureExtractor&,
    TimingResult&)
{
    std::cerr << "CUDA is not available. Compile with -DUSE_CUDA and nvcc.\n";
    return {};
}

CSRMatrix CUDAEngine::spmm_transpose_gpu(
    const CSRMatrix&, const CSRMatrix&,
    double&, double&, double&)
{
    std::cerr << "CUDA is not available.\n";
    return CSRMatrix();
}

} // namespace ids

#endif // !USE_CUDA

