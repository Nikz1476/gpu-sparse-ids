#include "output/alert_manager.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace ids {

void AlertManager::print_alerts(const std::vector<Alert>& alerts) {
    if (alerts.empty()) {
        std::cout << "\n=== No alerts generated ===\n";
        return;
    }

    std::cout << "\n=== IDS ALERTS (" << alerts.size() << ") ===" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    for (const auto& a : alerts) {
        std::cout << "[ALERT] Sig=" << a.signature_id
                  << " (" << a.signature_name << ") "
                  << " Pkt=" << a.packet_id
                  << " " << a.src_ip << ":" << a.src_port
                  << " -> " << a.dst_ip << ":" << a.dst_port
                  << " | " << a.match_detail
                  << std::endl;
    }
    std::cout << std::string(80, '-') << std::endl;
}

bool AlertManager::write_csv(const std::string& filepath, const std::vector<Alert>& alerts) {
    std::ofstream f(filepath);
    if (!f.is_open()) return false;

    f << "packet_id,signature_id,signature_name,src_ip,dst_ip,src_port,dst_port,timestamp,match_detail\n";
    for (const auto& a : alerts) {
        f << a.packet_id << "," << a.signature_id << ","
          << a.signature_name << "," << a.src_ip << "," << a.dst_ip << ","
          << a.src_port << "," << a.dst_port << ","
          << std::fixed << std::setprecision(6) << a.timestamp << ","
          << "\"" << a.match_detail << "\"\n";
    }
    return true;
}

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
    // Check if file exists
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

