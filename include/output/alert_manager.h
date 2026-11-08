#pragma once
#include "common/types.h"
#include <vector>
#include <string>

namespace ids {

// Summary of one full IDS pipeline run
struct ExecutionSummary {
    std::string mode;           // "Sequential", "OpenMP", "CUDA"
    int num_threads = 1;
    int num_packets = 0;
    int num_signatures = 0;
    int num_candidates = 0;
    int num_alerts = 0;
    int false_positive_candidates = 0; // candidates rejected by exact verification
    int ngram_size = 3;
    int vocab_size = 65536;
    int nnz_packets = 0;
    int nnz_sigs = 0;
    int C_nnz = 0;              // nnz of score matrix C
    double sparsity_P = 0.0;
    double sparsity_S = 0.0;
    double sparsity_C = 0.0;
    TimingResult timing;
};

class AlertManager {
public:
    // Terminal output
    static void print_alerts(const std::vector<Alert>& alerts);
    static void print_summary(const ExecutionSummary& summary);

    // CSV alert export
    static bool write_csv(const std::string& filepath, const std::vector<Alert>& alerts);

    // JSON alert export
    static bool write_json(const std::string& filepath,
                           const std::vector<Alert>& alerts,
                           const ExecutionSummary& summary);

    // Benchmark CSV (timing row per run)
    static bool write_timing_csv(const std::string& filepath,
        const std::string& mode, const TimingResult& timing,
        int num_packets, int num_sigs, int vocab_size,
        int nnz_packets, int nnz_sigs,
        int num_candidates, int num_alerts, int num_threads = 1);
    static bool append_timing_csv(const std::string& filepath,
        const std::string& mode, const TimingResult& timing,
        int num_packets, int num_sigs, int vocab_size,
        int nnz_packets, int nnz_sigs,
        int num_candidates, int num_alerts, int num_threads = 1);
};

} // namespace ids

