#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <chrono>

namespace ids {

// Packet representation
struct Packet {
    uint32_t id;
    double timestamp;
    std::string src_ip;
    std::string dst_ip;
    uint16_t src_port;
    uint16_t dst_port;
    uint8_t protocol; // 6=TCP, 17=UDP
    std::vector<uint8_t> payload;
};

// Attack signature
struct Signature {
    uint32_t id;
    std::string name;
    std::string description;
    std::vector<uint8_t> pattern;       // byte pattern to match
    uint8_t protocol;                    // 0=any, 6=TCP, 17=UDP
    uint16_t dst_port;                   // 0=any
    uint32_t num_features;               // number of distinct n-gram features
};

// CSR Sparse Matrix
struct CSRMatrix {
    int num_rows;
    int num_cols;
    int nnz;
    std::vector<int> row_ptr;    // size: num_rows + 1
    std::vector<int> col_idx;    // size: nnz
    std::vector<float> values;   // size: nnz (all 1.0 for binary)

    CSRMatrix() : num_rows(0), num_cols(0), nnz(0) {}

    double sparsity() const {
        if (num_rows == 0 || num_cols == 0) return 1.0;
        return 1.0 - static_cast<double>(nnz) / (static_cast<double>(num_rows) * num_cols);
    }
};

// Detection result
struct Alert {
    uint32_t packet_id;
    uint32_t signature_id;
    std::string signature_name;
    std::string src_ip;
    std::string dst_ip;
    uint16_t src_port;
    uint16_t dst_port;
    double timestamp;
    std::string match_detail;
};

// Candidate pair (packet_idx, signature_idx)
struct CandidatePair {
    int packet_idx;
    int signature_idx;
    float score;  // number of matching features
};

// Benchmark timing
struct TimingResult {
    double pcap_parse_ms = 0.0;
    double feature_extract_ms = 0.0;
    double csr_construct_ms = 0.0;
    double sparse_compute_ms = 0.0;
    double host_to_device_ms = 0.0;
    double device_to_host_ms = 0.0;
    double verification_ms = 0.0;
    double total_ms = 0.0;
    int num_candidates = 0;
    int num_alerts = 0;

    double total_compute_ms() const {
        return feature_extract_ms + csr_construct_ms + sparse_compute_ms + verification_ms;
    }
};

// Hardware info
struct HardwareInfo {
    std::string cpu_model;
    int num_cores = 0;
    size_t ram_bytes = 0;
    bool has_gpu = false;
    std::string gpu_model;
    size_t gpu_memory_bytes = 0;
};

// Execution mode
enum class ExecutionMode {
    SEQUENTIAL,
    OPENMP,
    CUDA
};

inline std::string mode_to_string(ExecutionMode mode) {
    switch (mode) {
        case ExecutionMode::SEQUENTIAL: return "Sequential";
        case ExecutionMode::OPENMP: return "OpenMP";
        case ExecutionMode::CUDA: return "CUDA";
    }
    return "Unknown";
}

} // namespace ids

