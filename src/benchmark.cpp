// Improved benchmark: compares Sequential vs OpenMP (and optionally CUDA).
// Reports detailed timing for each pipeline stage, speedup, efficiency,
// and writes benchmark_results.csv for matplotlib analysis.

#include "ingestion/pcap_reader.h"
#include "signatures/signature_db.h"
#include "features/ngram_features.h"
#include "sparse/csr_matrix.h"
#include "compute/cpu_engine.h"
#include "compute/openmp_engine.h"
#include "compute/cuda_engine.h"
#include "output/alert_manager.h"
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <algorithm>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace ids;
using Clock = std::chrono::high_resolution_clock;

// ============================================================
// Benchmark result for one pipeline run
// ============================================================

struct BenchResult {
    std::string mode;
    int threads = 1;
    int num_packets = 0;
    int num_sigs = 0;
    int vocab_size = 0;
    int nnz_packets = 0;
    int nnz_sigs = 0;
    int C_nnz = 0;
    TimingResult timing;
    int num_alerts = 0;
};

// ============================================================
// Hardware detection
// ============================================================

static std::string get_cpu_info() {
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line)) {
        if (line.find("model name") != std::string::npos) {
            auto pos = line.find(':');
            if (pos != std::string::npos)
                return line.substr(pos + 2);
        }
    }
    return "Unknown";
}

static int get_num_cpu_cores() {
    int n = 0;
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line)) {
        if (line.find("processor") != std::string::npos) ++n;
    }
    return n;
}

static size_t get_ram_kb() {
    std::ifstream f("/proc/meminfo");
    std::string line;
    while (std::getline(f, line)) {
        if (line.find("MemTotal") != std::string::npos) {
            size_t kb = 0;
            std::sscanf(line.c_str(), "MemTotal: %zu kB", &kb);
            return kb;
        }
    }
    return 0;
}

// ============================================================
// Print one result row
// ============================================================

static void print_result(const BenchResult& r, double ref_total_ms) {
    double pps = (r.timing.total_ms > 0)
        ? (r.num_packets / (r.timing.total_ms / 1000.0)) : 0;
    double speedup = (ref_total_ms > 0 && r.timing.total_ms > 0)
        ? (ref_total_ms / r.timing.total_ms) : 1.0;
    double efficiency = (r.threads > 1) ? (speedup / r.threads) : 1.0;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n--- " << r.mode << " (threads=" << r.threads << ") ---\n"
              << "  PCAP parsing:       " << r.timing.pcap_parse_ms << " ms\n"
              << "  Feature extraction: " << r.timing.feature_extract_ms << " ms\n"
              << "  CSR construction:   " << r.timing.csr_construct_ms << " ms\n"
              << "  Sparse SpMM:        " << r.timing.sparse_compute_ms << " ms\n";
    if (r.timing.host_to_device_ms > 0)
        std::cout << "  Host->Device:       " << r.timing.host_to_device_ms << " ms\n"
                  << "  Device->Host:       " << r.timing.device_to_host_ms << " ms\n";
    std::cout << "  Exact verification: " << r.timing.verification_ms << " ms\n"
              << "  TOTAL (compute):    " << r.timing.total_compute_ms() << " ms\n"
              << "  TOTAL (end-to-end): " << r.timing.total_ms << " ms\n"
              << "  Throughput:         " << std::setprecision(0) << pps << " pkts/sec\n"
              << "  Candidates:         " << r.timing.num_candidates << "\n"
              << "  Alerts:             " << r.num_alerts << "\n"
              << "  Speedup (vs seq):   " << std::setprecision(3) << speedup << "x\n"
              << "  Parallel efficiency:" << std::setprecision(1) << efficiency * 100.0 << "%\n"
              << "  P nnz=" << r.nnz_packets << "  S nnz=" << r.nnz_sigs
              << "  C nnz=" << r.C_nnz << "\n";
}

// ============================================================
// Write CSV row
// ============================================================

static void write_csv_header(std::ofstream& f) {
    f << "mode,threads,num_packets,num_signatures,vocab_size,"
      << "nnz_packets,nnz_sigs,C_nnz,"
      << "pcap_parse_ms,feature_extract_ms,csr_construct_ms,"
      << "sparse_compute_ms,host_to_device_ms,device_to_host_ms,"
      << "verification_ms,total_compute_ms,total_ms,"
      << "num_candidates,num_alerts,packets_per_sec,speedup,parallel_efficiency\n";
}

static void write_csv_row(std::ofstream& f, const BenchResult& r, double ref_total_ms) {
    double pps = (r.timing.total_ms > 0)
        ? (r.num_packets / (r.timing.total_ms / 1000.0)) : 0;
    double speedup = (ref_total_ms > 0 && r.timing.total_ms > 0)
        ? (ref_total_ms / r.timing.total_ms) : 1.0;
    double efficiency = (r.threads > 1) ? (speedup / r.threads) : 1.0;

    f << r.mode << "," << r.threads << "," << r.num_packets << "," << r.num_sigs << ","
      << r.vocab_size << "," << r.nnz_packets << "," << r.nnz_sigs << "," << r.C_nnz << ","
      << std::fixed << std::setprecision(4)
      << r.timing.pcap_parse_ms << "," << r.timing.feature_extract_ms << ","
      << r.timing.csr_construct_ms << "," << r.timing.sparse_compute_ms << ","
      << r.timing.host_to_device_ms << "," << r.timing.device_to_host_ms << ","
      << r.timing.verification_ms << "," << r.timing.total_compute_ms() << "," << r.timing.total_ms << ","
      << r.timing.num_candidates << "," << r.num_alerts << ","
      << std::setprecision(2) << pps << ","
      << std::setprecision(4) << speedup << "," << efficiency << "\n";
}

// ============================================================
// Roofline analysis
// ============================================================

static void roofline_analysis(const CSRMatrix& P, const CSRMatrix& S,
                              const std::vector<BenchResult>& results) {
    // SpMM kernel: for each (i,j) pair, merge A_row_i and B_row_j
    // Operations: comparisons in merge = O(nnz_A_row + nnz_B_row) per row-pair
    double total_ops = 0;
    for (int i = 0; i < P.num_rows; ++i) {
        double a_nnz = P.row_ptr[i+1] - P.row_ptr[i];
        for (int j = 0; j < S.num_rows; ++j) {
            double b_nnz = S.row_ptr[j+1] - S.row_ptr[j];
            total_ops += a_nnz + b_nnz;  // merge comparison ops
        }
    }

    // Data accessed: row_ptr (4 bytes each), col_idx (4 bytes each), values (4 bytes each)
    double bytes_P = (P.num_rows + 1) * 4.0 + P.nnz * 8.0;  // row_ptr + col_idx + values
    double bytes_S = (S.num_rows + 1) * 4.0 + S.nnz * 8.0;
    double total_bytes = bytes_P + bytes_S;
    double arithmetic_intensity = total_ops / total_bytes;

    // Find best measured SpMM throughput
    double best_spmm_gops = 0;
    for (const auto& r : results) {
        if (r.timing.sparse_compute_ms > 0) {
            double gops = (total_ops / r.timing.sparse_compute_ms) * 1e-6;  // GOps/s
            best_spmm_gops = std::max(best_spmm_gops, gops);
        }
    }

    std::cout << "\n--- Roofline Analysis (SpMM kernel) ---\n"
              << "  P: " << P.num_rows << " x " << P.num_cols << "  nnz=" << P.nnz << "\n"
              << "  S: " << S.num_rows << " x " << S.num_cols << "  nnz=" << S.nnz << "\n"
              << "  Merge operations (flop proxy): " << std::scientific << std::setprecision(3)
              << total_ops << "\n"
              << "  Data accessed (bytes):         " << total_bytes << "\n"
              << "  Arithmetic intensity:          " << std::fixed << std::setprecision(4)
              << arithmetic_intensity << " ops/byte\n"
              << "  Classification:                " << (arithmetic_intensity < 1.0 ? "MEMORY-BOUND" : "COMPUTE-BOUND") << "\n";
    if (best_spmm_gops > 0)
        std::cout << "  Best measured SpMM rate:       " << std::setprecision(3) << best_spmm_gops << " GOps/s\n";
    std::cout << "  NOTE: Sparse irregular access further limits effective bandwidth.\n"
              << "        Hardware peak values not available without dedicated measurement.\n"
              << "        This analysis is PRELIMINARY — marked as estimated.\n";
}

// ============================================================
// Main
// ============================================================

int main(int argc, char* argv[]) {
    std::string pcap_file  = "data/pcap/benchmark_traffic.pcap";
    std::string sig_file   = "data/signatures/test_signatures.txt";
    std::string csv_file   = "benchmarks/benchmark_results.csv";
    int ngram_size = 3;
    int vocab_size = 65536;

    if (argc > 1) pcap_file = argv[1];
    if (argc > 2) sig_file  = argv[2];
    if (argc > 3) csv_file  = argv[3];
    if (argc > 4) ngram_size = std::atoi(argv[4]);
    if (argc > 5) vocab_size = std::atoi(argv[5]);

    // ---- Hardware info ----
    std::cout << "========================================\n"
              << " GPU Sparse IDS — Benchmark\n"
              << "========================================\n";
    std::string cpu = get_cpu_info();
    int cores = get_num_cpu_cores();
    size_t ram_mb = get_ram_kb() / 1024;
    std::cout << "CPU:     " << cpu << "\n"
              << "Cores:   " << cores << " logical\n"
              << "RAM:     " << ram_mb << " MB\n";

#ifdef _OPENMP
    int omp_max = omp_get_max_threads();
    std::cout << "OpenMP:  available, max_threads=" << omp_max << "\n";
#else
    std::cout << "OpenMP:  not compiled\n";
    int omp_max = 1;
#endif

    bool cuda_ok = CUDAEngine::is_available();
    std::cout << "CUDA:    " << (cuda_ok ? "available" : "not available") << "\n"
              << "PCAP:    " << pcap_file << "\n"
              << "Sigs:    " << sig_file << "\n"
              << "N-gram:  " << ngram_size << "\n"
              << "Vocab:   " << vocab_size << "\n"
              << "========================================\n\n";

    // ---- Load data ----
    auto t_load0 = Clock::now();
    auto packets = PcapReader::read_pcap(pcap_file);
    auto t_load1 = Clock::now();
    double pcap_ms = std::chrono::duration<double, std::milli>(t_load1 - t_load0).count();

    auto sigs = SignatureDB::load_signatures(sig_file);

    std::cout << "Loaded " << packets.size() << " packets (" << std::fixed
              << std::setprecision(2) << pcap_ms << " ms)\n"
              << "Loaded " << sigs.size() << " signatures\n\n";

    if (packets.empty() || sigs.empty()) {
        std::cerr << "No data to benchmark.\n";
        return 1;
    }

    NGramFeatureExtractor extractor(ngram_size, vocab_size);

    // Pre-compute nnz for consistent reporting
    auto pkt_feats_ref = CPUEngine::extract_all_features(packets, extractor);
    auto sigs_ref = sigs;
    auto sig_feats_ref = CPUEngine::extract_signature_features(sigs_ref, extractor);
    CSRMatrix P_ref = CSRBuilder::build_from_features(pkt_feats_ref, vocab_size);
    CSRMatrix S_ref = CSRBuilder::build_from_features(sig_feats_ref, vocab_size);

    std::cout << "Reference matrices:\n";
    CSRBuilder::print_info(P_ref, "P (packets)");
    CSRBuilder::print_info(S_ref, "S (signatures)");
    std::cout << "\n";

    std::vector<BenchResult> results;
    double ref_total_ms = 0;

    // ---- 1. Sequential ----
    {
        BenchResult br;
        br.mode = "Sequential";
        br.threads = 1;
        br.num_packets = static_cast<int>(packets.size());
        br.num_sigs = static_cast<int>(sigs.size());
        br.vocab_size = vocab_size;
        br.nnz_packets = P_ref.nnz;
        br.nnz_sigs = S_ref.nnz;
        br.timing.pcap_parse_ms = pcap_ms;

        auto sigs_copy = sigs;
        CSRMatrix out_C;
        auto alerts = CPUEngine::run_sequential(packets, sigs_copy, extractor, br.timing,
                                                nullptr, nullptr, &out_C);
        br.C_nnz = out_C.nnz;
        br.num_alerts = static_cast<int>(alerts.size());
        br.timing.pcap_parse_ms = pcap_ms;

        ref_total_ms = br.timing.total_ms;
        results.push_back(br);
    }

    // ---- 2. OpenMP with multiple thread counts ----
    std::vector<int> thread_counts = {1, 2, 4};
    if (omp_max >= 8)  thread_counts.push_back(8);
    if (omp_max >= 16) thread_counts.push_back(16);

    for (int nt : thread_counts) {
        BenchResult br;
        br.mode = "OpenMP";
        br.threads = nt;
        br.num_packets = static_cast<int>(packets.size());
        br.num_sigs = static_cast<int>(sigs.size());
        br.vocab_size = vocab_size;
        br.nnz_packets = P_ref.nnz;
        br.nnz_sigs = S_ref.nnz;

        auto sigs_copy = sigs;
        CSRMatrix out_C;
        auto alerts = OpenMPEngine::run_parallel(packets, sigs_copy, extractor, br.timing,
                                                 nt, nullptr, nullptr, &out_C);
        br.C_nnz = out_C.nnz;
        br.num_alerts = static_cast<int>(alerts.size());
        br.timing.pcap_parse_ms = pcap_ms;
        results.push_back(br);
    }

    // ---- 3. CUDA (if available) ----
    if (cuda_ok) {
        BenchResult br;
        br.mode = "CUDA";
        br.threads = 0;
        br.num_packets = static_cast<int>(packets.size());
        br.num_sigs = static_cast<int>(sigs.size());
        br.vocab_size = vocab_size;
        br.nnz_packets = P_ref.nnz;
        br.nnz_sigs = S_ref.nnz;
        br.C_nnz = 0; // populated by GPU

        auto sigs_copy = sigs;
        auto alerts = CUDAEngine::run_gpu(packets, sigs_copy, extractor, br.timing);
        br.num_alerts = static_cast<int>(alerts.size());
        br.timing.pcap_parse_ms = pcap_ms;
        results.push_back(br);
    } else {
        std::cout << "\nCUDA not available — skipping GPU benchmark.\n";
    }

    // ---- Print results ----
    std::cout << "\n\n========================================\n"
              << " BENCHMARK RESULTS (" << packets.size() << " packets, "
              << sigs.size() << " sigs)\n"
              << "========================================\n";

    for (const auto& r : results) {
        print_result(r, ref_total_ms);
    }

    // ---- Speedup table ----
    std::cout << "\n--- Speedup Summary ---\n"
              << std::left << std::setw(16) << "Mode"
              << std::setw(10) << "Threads"
              << std::setw(14) << "Total (ms)"
              << std::setw(12) << "Speedup"
              << std::setw(12) << "Pkts/sec" << "\n";
    std::cout << std::string(64, '-') << "\n";
    for (const auto& r : results) {
        double pps = (r.timing.total_ms > 0)
            ? (r.num_packets / (r.timing.total_ms / 1000.0)) : 0;
        double speedup = (ref_total_ms > 0 && r.timing.total_ms > 0)
            ? (ref_total_ms / r.timing.total_ms) : 1.0;
        std::cout << std::left << std::setw(16) << r.mode
                  << std::setw(10) << r.threads
                  << std::setw(14) << std::fixed << std::setprecision(1) << r.timing.total_ms
                  << std::setw(12) << std::setprecision(2) << speedup
                  << std::setw(12) << std::setprecision(0) << pps << "\n";
    }

    // ---- CSV ----
    {
        std::ofstream csv(csv_file);
        if (csv.is_open()) {
            write_csv_header(csv);
            for (const auto& r : results)
                write_csv_row(csv, r, ref_total_ms);
            std::cout << "\nBenchmark CSV written to: " << csv_file << "\n";
        } else {
            std::cerr << "Failed to write CSV: " << csv_file << "\n";
        }
    }

    // ---- Memory usage ----
    {
        size_t pkt_mem = 0;
        for (const auto& p : packets) pkt_mem += p.payload.size() + 64;
        size_t p_mem = P_ref.row_ptr.size()*4 + P_ref.col_idx.size()*4 + P_ref.values.size()*4;
        size_t s_mem = S_ref.row_ptr.size()*4 + S_ref.col_idx.size()*4 + S_ref.values.size()*4;
        std::cout << "\n--- Memory Usage Estimate ---\n"
                  << "  Packets data:  " << pkt_mem / 1024 << " KB\n"
                  << "  Matrix P CSR:  " << p_mem / 1024 << " KB\n"
                  << "  Matrix S CSR:  " << s_mem / 1024 << " KB\n"
                  << "  (excludes feature lists, candidates, alerts)\n";
    }

    // ---- Roofline analysis ----
    roofline_analysis(P_ref, S_ref, results);

    return 0;
}
