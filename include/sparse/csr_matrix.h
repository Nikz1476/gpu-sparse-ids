#pragma once
#include "common/types.h"
#include <vector>
#include <string>

namespace ids {

class CSRBuilder {
public:
    // Build CSR matrix from per-row feature index lists
    static CSRMatrix build_from_features(
        const std::vector<std::vector<int>>& features,
        int num_cols);

    // Compute C = A * B^T  (A: MxK, B: NxK => C: MxN)
    static CSRMatrix spmm_transpose(const CSRMatrix& A, const CSRMatrix& B);

    // Print matrix info
    static void print_info(const CSRMatrix& mat, const std::string& name);
};

} // namespace ids

