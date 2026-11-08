#include "output/alert_manager.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace ids {

// ============================================================
// Terminal output
// ============================================================

void AlertManager::print_alerts(const std::vector<Alert>& alerts) {
    if (alerts.empty()) {
        std::cout << "\n=== No alerts generated ===\n";
        return;
    }

    std::cout << "\n=== IDS ALERTS (" << alerts.size() << ") ===" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    for (const auto& a : alerts) {
        const char* proto = (a.protocol == 6) ? "TCP" :
                            (a.protocol == 17) ? "UDP" : "???";
        std::cout << "[ALERT] "
                  << "Sig=" << a.signature_id << " (" << a.signature_name << ")"
                  << "  Pkt=" << a.packet_id
                  << "  " << proto
                  << "  " << a.src_ip << ":" << a.src_port
                  << " -> " << a.dst_ip << ":" << a.dst_port
                  << "  ts=" << std::fixed << std::setprecision(3) << a.timestamp
                  << "  | " << a.match_detail
                  << std::endl;
    }
    std::cout << std::string(80, '-') << std::endl;
}

void AlertManager::print_summary(const ExecutionSummary& summary) {
    std::cout << "\n=== Execution Summary ===" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    std::cout << "Mode:               " << summary.mode << std::endl;
    if (summary.mode == "OpenMP")
        std::cout << "Threads:            " << summary.num_threads << std::endl;
    std::cout << "Packets processed:  " << summary.num_packets << std::endl;
    std::cout << "Signatures loaded:  " << summary.num_signatures << std::endl;
    std::cout << "N-gram size:        " << summary.ngram_size << std::endl;
    std::cout << "Vocab size:         " << summary.vocab_size << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    std::cout << "Matrix P:           " << summary.num_packets << " x " << summary.vocab_size
              << "  nnz=" << summary.nnz_packets
              << "  sparsity=" << std::fixed << std::setprecision(4)
              << summary.sparsity_P * 100.0 << "%" << std::endl;
    std::cout << "Matrix S:           " << summary.num_signatures << " x " << summary.vocab_size
              << "  nnz=" << summary.nnz_sigs
              << "  sparsity=" << std::fixed << std::setprecision(4)
              << summary.sparsity_S * 100.0 << "%" << std::endl;
    std::cout << "Matrix C:           " << summary.num_packets << " x " << summary.num_signatures
              << "  nnz=" << summary.C_nnz
              << "  sparsity=" << std::fixed << std::setprecision(4)
              << summary.sparsity_C * 100.0 << "%" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    std::cout << "Candidate pairs:    " << summary.num_candidates << std::endl;
    std::cout << "Verified alerts:    " << summary.num_alerts << std::endl;
    std::cout << "FP candidates rej.: " << summary.false_positive_candidates << std::endl;
    std::cout << std::string(40, '-') << std::endl;

    double pps = (summary.timing.total_ms > 0)
        ? (summary.num_packets / (summary.timing.total_ms / 1000.0)) : 0;
    std::cout << "PCAP parsing:       " << std::fixed << std::setprecision(3)
              << summary.timing.pcap_parse_ms << " ms" << std::endl;
    std::cout << "Feature extraction: " << summary.timing.feature_extract_ms << " ms" << std::endl;
    std::cout << "CSR construction:   " << summary.timing.csr_construct_ms << " ms" << std::endl;
    std::cout << "Sparse computation: " << summary.timing.sparse_compute_ms << " ms" << std::endl;
    if (summary.timing.host_to_device_ms > 0)
        std::cout << "Host->Device:       " << summary.timing.host_to_device_ms << " ms" << std::endl;
    if (summary.timing.device_to_host_ms > 0)
        std::cout << "Device->Host:       " << summary.timing.device_to_host_ms << " ms" << std::endl;
    std::cout << "Exact verification: " << summary.timing.verification_ms << " ms" << std::endl;
    std::cout << "Total:              " << summary.timing.total_ms << " ms" << std::endl;
    std::cout << "Throughput:         " << std::setprecision(0) << pps << " pkts/sec" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
}

// ============================================================
// CSV alert export
// ============================================================

bool AlertManager::write_csv(const std::string& filepath, const std::vector<Alert>& alerts) {
    std::ofstream f(filepath);
    if (!f.is_open()) return false;

    f << "packet_id,signature_id,signature_name,protocol,"
      << "src_ip,src_port,dst_ip,dst_port,timestamp,match_detail\n";

    for (const auto& a : alerts) {
        const char* proto = (a.protocol == 6) ? "TCP" :
                            (a.protocol == 17) ? "UDP" : "other";
        // Escape commas in string fields with quotes
        f << a.packet_id << ","
          << a.signature_id << ","
          << "\"" << a.signature_name << "\","
          << proto << ","
          << a.src_ip << ","
          << a.src_port << ","
          << a.dst_ip << ","
          << a.dst_port << ","
          << std::fixed << std::setprecision(6) << a.timestamp << ","
          << "\"" << a.match_detail << "\"\n";
    }
    return true;
}

// ============================================================
// JSON alert export
// ============================================================

// Minimal JSON string escaping
static std::string json_str(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 4);
    out += '"';
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:   out += c; break;
        }
    }
    out += '"';
    return out;
}

bool AlertManager::write_json(const std::string& filepath,
                              const std::vector<Alert>& alerts,
                              const ExecutionSummary& summary)
{
    std::ofstream f(filepath);
    if (!f.is_open()) return false;

    f << "{\n";

    // --- Execution summary section ---
    f << "  \"summary\": {\n";
    f << "    \"mode\": " << json_str(summary.mode) << ",\n";
    f << "    \"num_threads\": " << summary.num_threads << ",\n";
    f << "    \"num_packets\": " << summary.num_packets << ",\n";
    f << "    \"num_signatures\": " << summary.num_signatures << ",\n";
    f << "    \"ngram_size\": " << summary.ngram_size << ",\n";
    f << "    \"vocab_size\": " << summary.vocab_size << ",\n";
    f << "    \"num_candidates\": " << summary.num_candidates << ",\n";
    f << "    \"num_alerts\": " << summary.num_alerts << ",\n";
    f << "    \"false_positive_candidates\": " << summary.false_positive_candidates << ",\n";
    f << "    \"matrix_P\": {"
      << "\"rows\": " << summary.num_packets
      << ", \"cols\": " << summary.vocab_size
      << ", \"nnz\": " << summary.nnz_packets
      << ", \"sparsity_pct\": " << std::fixed << std::setprecision(4) << (summary.sparsity_P * 100.0)
      << "},\n";
    f << "    \"matrix_S\": {"
      << "\"rows\": " << summary.num_signatures
      << ", \"cols\": " << summary.vocab_size
      << ", \"nnz\": " << summary.nnz_sigs
      << ", \"sparsity_pct\": " << (summary.sparsity_S * 100.0)
      << "},\n";
    f << "    \"matrix_C\": {"
      << "\"rows\": " << summary.num_packets
      << ", \"cols\": " << summary.num_signatures
      << ", \"nnz\": " << summary.C_nnz
      << ", \"sparsity_pct\": " << (summary.sparsity_C * 100.0)
      << "},\n";

    double pps = (summary.timing.total_ms > 0)
        ? (summary.num_packets / (summary.timing.total_ms / 1000.0)) : 0;
    f << "    \"timing_ms\": {"
      << "\"pcap_parse\": " << std::setprecision(4) << summary.timing.pcap_parse_ms
      << ", \"feature_extract\": " << summary.timing.feature_extract_ms
      << ", \"csr_construct\": " << summary.timing.csr_construct_ms
      << ", \"sparse_compute\": " << summary.timing.sparse_compute_ms
      << ", \"host_to_device\": " << summary.timing.host_to_device_ms
      << ", \"device_to_host\": " << summary.timing.device_to_host_ms
      << ", \"verification\": " << summary.timing.verification_ms
      << ", \"total\": " << summary.timing.total_ms
      << "},\n";
    f << "    \"throughput_pkts_per_sec\": " << std::setprecision(2) << pps << "\n";
    f << "  },\n";

    // --- Alerts section ---
    f << "  \"alerts\": [\n";
    for (size_t i = 0; i < alerts.size(); ++i) {
        const auto& a = alerts[i];
        const char* proto = (a.protocol == 6) ? "TCP" :
                            (a.protocol == 17) ? "UDP" : "other";
        f << "    {\n";
        f << "      \"packet_id\": " << a.packet_id << ",\n";
        f << "      \"signature_id\": " << a.signature_id << ",\n";
        f << "      \"signature_name\": " << json_str(a.signature_name) << ",\n";
        f << "      \"protocol\": " << json_str(proto) << ",\n";
        f << "      \"src_ip\": " << json_str(a.src_ip) << ",\n";
        f << "      \"src_port\": " << a.src_port << ",\n";
        f << "      \"dst_ip\": " << json_str(a.dst_ip) << ",\n";
        f << "      \"dst_port\": " << a.dst_port << ",\n";
        f << "      \"timestamp\": " << std::fixed << std::setprecision(6) << a.timestamp << ",\n";
        f << "      \"match_detail\": " << json_str(a.match_detail) << "\n";
        f << "    }" << (i + 1 < alerts.size() ? "," : "") << "\n";
    }
    f << "  ]\n";
    f << "}\n";

    return true;
}

// ============================================================
// Benchmark CSV helpers
// ============================================================

static void write_timing_header(std::ofstream& f) {
    f << "mode,num_threads,num_packets,num_signatures,vocab_size,"
      << "nnz_packets,nnz_signatures,"
      << "pcap_parse_ms,feature_extract_ms,csr_construct_ms,"
      << "sparse_compute_ms,host_to_device_ms,device_to_host_ms,"
      << "verification_ms,total_ms,"
      << "num_candidates,num_alerts,"
      << "packets_per_sec,speedup\n";
}

static void write_timing_row(std::ofstream& f,
    const std::string& mode, const TimingResult& t,
    int num_packets, int num_sigs, int vocab_size,
    int nnz_p, int nnz_s,
    int num_cand, int num_alerts, int num_threads)
{
    double pps = (t.total_ms > 0) ? (num_packets / (t.total_ms / 1000.0)) : 0;
    f << mode << "," << num_threads << "," << num_packets << "," << num_sigs << ","
      << vocab_size << "," << nnz_p << "," << nnz_s << ","
      << std::fixed << std::setprecision(4)
      << t.pcap_parse_ms << "," << t.feature_extract_ms << ","
      << t.csr_construct_ms << "," << t.sparse_compute_ms << ","
      << t.host_to_device_ms << "," << t.device_to_host_ms << ","
      << t.verification_ms << "," << t.total_ms << ","
      << num_cand << "," << num_alerts << ","
      << std::setprecision(2) << pps << ",1.00\n";
}

bool AlertManager::write_timing_csv(const std::string& filepath,
    const std::string& mode, const TimingResult& t,
    int num_packets, int num_sigs, int vocab_size,
    int nnz_p, int nnz_s,
    int num_cand, int num_alerts, int num_threads)
{
    std::ofstream f(filepath);
    if (!f.is_open()) return false;
    write_timing_header(f);
    write_timing_row(f, mode, t, num_packets, num_sigs, vocab_size, nnz_p, nnz_s,
                     num_cand, num_alerts, num_threads);
    return true;
}

bool AlertManager::append_timing_csv(const std::string& filepath,
    const std::string& mode, const TimingResult& t,
    int num_packets, int num_sigs, int vocab_size,
    int nnz_p, int nnz_s,
    int num_cand, int num_alerts, int num_threads)
{
    bool exists = false;
    {
        std::ifstream test(filepath);
        exists = test.good();
    }

    std::ofstream f(filepath, std::ios::app);
    if (!f.is_open()) return false;
    if (!exists) write_timing_header(f);
    write_timing_row(f, mode, t, num_packets, num_sigs, vocab_size, nnz_p, nnz_s,
                     num_cand, num_alerts, num_threads);
    return true;
}

} // namespace ids
