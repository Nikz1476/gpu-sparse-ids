#pragma once
#include "common/types.h"
#include <vector>
#include <string>

namespace ids {

class AlertManager {
public:
    static void print_alerts(const std::vector<Alert>& alerts);
    static bool write_csv(const std::string& filepath, const std::vector<Alert>& alerts);
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

