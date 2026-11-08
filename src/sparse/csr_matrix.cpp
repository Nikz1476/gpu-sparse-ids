#include "sparse/csr_matrix.h"
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <iomanip>

namespace ids {

CSRMatrix CSRBuilder::build_from_features(
    const std::vector<std::vector<int>>& features,
    int num_cols)
{
    CSRMatrix mat;
    mat.num_rows = static_cast<int>(features.size());
    mat.num_cols = num_cols;

    mat.row_ptr.resize(mat.num_rows + 1);
    mat.row_ptr[0] = 0;

    // Count nnz and build row_ptr
    int total_nnz = 0;
    for (int i = 0; i < mat.num_rows; ++i) {
        total_nnz += static_cast<int>(features[i].size());
        mat.row_ptr[i + 1] = total_nnz;
    }

    mat.nnz = total_nnz;
    mat.col_idx.resize(total_nnz);
    mat.values.resize(total_nnz, 1.0f);

    // Fill col_idx (features are already sorted and unique)
    int idx = 0;
    for (int i = 0; i < mat.num_rows; ++i) {
        for (int col : features[i]) {
            mat.col_idx[idx++] = col;
        }
    }

    return mat;
}

CSRMatrix CSRBuilder::spmm_transpose(const CSRMatrix& A, const CSRMatrix& B) {
    // C = A * B^T where A is (M x K) and B is (N x K), result C is (M x N)
    // C[i][j] = dot(A_row_i, B_row_j) — count of shared column indices
    //
    // Algorithm: For each row i of A, iterate over all rows j of B,
    //   compute intersection of sorted column index lists.
    //   This is the standard merge-based sparse dot product.

    int M = A.num_rows;
    int N = B.num_rows;

    CSRMatrix C;
    C.num_rows = M;
    C.num_cols = N;

    C.row_ptr.resize(M + 1);
    C.row_ptr[0] = 0;

    std::vector<int> c_col_idx;
    std::vector<float> c_values;

    for (int i = 0; i < M; ++i) {
        int a_start = A.row_ptr[i];
        int a_end   = A.row_ptr[i + 1];

        for (int j = 0; j < N; ++j) {
            int b_start = B.row_ptr[j];
            int b_end   = B.row_ptr[j + 1];

            // Merge-based intersection count
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
                c_col_idx.push_back(j);
                c_values.push_back(dot);
            }
        }

        C.row_ptr[i + 1] = static_cast<int>(c_col_idx.size());
    }

    C.nnz = static_cast<int>(c_col_idx.size());
    C.col_idx = std::move(c_col_idx);
    C.values = std::move(c_values);

    return C;
}

void CSRBuilder::print_info(const CSRMatrix& mat, const std::string& name) {
    std::cout << "Matrix [" << name << "]: "
              << mat.num_rows << " x " << mat.num_cols
              << ", nnz = " << mat.nnz
              << ", sparsity = " << std::fixed << std::setprecision(6)
              << mat.sparsity() * 100.0 << "%" << std::endl;
}

} // namespace ids

