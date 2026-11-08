// Benchmark executable — compares Sequential vs OpenMP execution
// and optionally CUDA, with detailed timing breakdowns.

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

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace ids;
using Clock = std::chrono::high_resolution_clock;

struct BenchResult {
    std::string mode;
    int threads;
    TimingResult timing;
    int num_packets;
    int num_sigs;
    int vocab_size;
    int nnz_packets;
    int nnz_sigs;
};

static void print_bench_result(const BenchResult& r, double ref_total_ms = 0) {
    double pps = (r.timing.total_ms > 0)
        ? (r.num_packets / (r.timing.total_ms / 1000.0)) : 0;
    double speedup = (ref_total_ms > 0 && r.timing.total_ms > 0)
        ? (ref_total_ms / r.timing.total_ms) : 1.0;
    double efficiency = (r.threads > 0) ? (speedup / r.threads) : speedup;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n--- " << r.mode << " (threads=" << r.threads << ") ---\n"
              << "  Feature extraction: " << r.timing.feature_extract_ms << " ms\n"
              << "  CSR construction:   " << r.timing.csr_construct_ms << " ms\n"
              << "  Sparse compute:     " << r.timing.sparse_compute_ms << " ms\n";
    if (r.timing.host_to_device_ms > 0)
        std::cout << "  Host->Device:       " << r.timing.host_to_device_ms << " ms\n"
                  << "  Device->Host:       " << r.timing.device_to_host_ms << " ms\n";
    std::cout << "  Verification:       " << r.timing.verification_ms << " ms\n"
              << "  TOTAL:              " << r.timing.total_ms << " ms\n"
              << "  Packets/sec:        " << std::setprecision(0) << pps << "\n"
              << "  Candidates:         " << r.timing.num_candidates << "\n"
              << "  Alerts:             " << r.timing.num_alerts << "\n"
              << "  Speedup:            " << std::setprecision(3) << speedup << "x\n"
              << "  Parallel efficiency:" << std::setprecision(3) << efficiency * 100 << "%\n";
}

int main(int argc, char* argv[]) {
    std::string pcap_file = "data/pcap/benchmark_traffic.pcap";
    std::string sig_file  = "data/signatures/test_signatures.txt";
    std::string csv_file  = "benchmarks/benchmark_results.csv";
    int ngram_size = 3;
    int vocab_size = 65536;

    if (argc > 1) pcap_file = argv[1];
    if (argc > 2) sig_file  = argv[2];
    if (argc > 3) csv_file  = argv[3];

    std::cout << "========================================\n"
              << " GPU Sparse IDS — Benchmark\n"
              << "========================================\n"
              << "PCAP: " << pcap_file << "\n"
              << "Sigs: " << sig_file << "\n\n";

    // Load data
    auto t0 = Clock::now();
    auto packets = PcapReader::read_pcap(pcap_file);
    auto t1 = Clock::now();
    double pcap_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto sigs = SignatureDB::load_signatures(sig_file);

    std::cout << "Loaded " << packets.size() << " packets (" << pcap_ms << " ms)\n"
              << "Loaded " << sigs.size() << " signatures\n\n";

    if (packets.empty() || sigs.empty()) {
        std::cerr << "No data to benchmark.\n";
        return 1;
    }

    NGramFeatureExtractor extractor(ngram_size, vocab_size);

    // Pre-compute nnz for reporting
    auto pkt_feats = CPUEngine::extract_all_features(packets, extractor);
    auto sigs_copy = sigs;
    auto sig_feats = CPUEngine::extract_signature_features(sigs_copy, extractor);
    CSRMatrix P_info = CSRBuilder::build_from_features(pkt_feats, vocab_size);
    CSRMatrix S_info = CSRBuilder::build_from_features(sig_feats, vocab_size);

    std::cout << "Matrix P: " << P_info.num_rows << "x" << P_info.num_cols
              << " nnz=" << P_info.nnz << " sparsity=" << P_info.sparsity() * 100 << "%\n";
    std::cout << "Matrix S: " << S_info.num_rows << "x" << S_info.num_cols
              << " nnz=" << S_info.nnz << " sparsity=" << S_info.sparsity() * 100 << "%\n\n";

    std::vector<BenchResult> results;

    // 1. Sequential
    {
        BenchResult br;
        br.mode = "Sequential";
        br.threads = 1;
        br.num_packets = packets.size();
        br.num_sigs = sigs.size();
        br.vocab_size = vocab_size;
        br.nnz_packets = P_info.nnz;
        br.nnz_sigs = S_info.nnz;
        br.timing.pcap_parse_ms = pcap_ms;

        auto alerts = CPUEngine::run_sequential(packets, sigs, extractor, br.timing);
        br.timing.pcap_parse_ms = pcap_ms;
        results.push_back(br);
    }

    double ref_total = results[0].timing.total_ms;

    // 2. OpenMP with different thread counts
    std::vector<int> thread_counts = {1, 2};
    #ifdef _OPENMP
    int max_threads = omp_get_max_threads();
    if (max_threads > 2) thread_counts.push_back(max_threads);
    if (max_threads > 4) thread_counts.push_back(4);
    // Sort and deduplicate
    std::sort(thread_counts.begin(), thread_counts.end());
    thread_counts.erase(std::unique(thread_counts.begin(), thread_counts.end()), thread_counts.end());
    #endif

    for (int nt : thread_counts) {
        BenchResult br;
        br.mode = "OpenMP";
        br.threads = nt;
        br.num_packets = packets.size();
        br.num_sigs = sigs.size();
        br.vocab_size = vocab_size;
        br.nnz_packets = P_info.nnz;
        br.nnz_sigs = S_info.nnz;

        auto sigs_copy2 = sigs;
        auto alerts = OpenMPEngine::run_parallel(packets, sigs_copy2, extractor, br.timing, nt);
        br.timing.pcap_parse_ms = pcap_ms;
        results.push_back(br);
    }

    // 3. CUDA (if available)
    if (CUDAEngine::is_available()) {
        BenchResult br;
        br.mode = "CUDA";
        br.threads = 0;
        br.num_packets = packets.size();
        br.num_sigs = sigs.size();
        br.vocab_size = vocab_size;
        br.nnz_packets = P_info.nnz;
        br.nnz_sigs = S_info.nnz;

        auto sigs_copy2 = sigs;
        auto alerts = CUDAEngine::run_gpu(packets, sigs_copy2, extractor, br.timing);
        br.timing.pcap_parse_ms = pcap_ms;
        results.push_back(br);
    } else {
        std::cout << "\nCUDA not available — skipping GPU benchmark.\n";
    }

    // Print results
    std::cout << "\n\n========================================\n"
              << " BENCHMARK RESULTS\n"
              << "========================================\n";

    for (const auto& r : results) {
        print_bench_result(r, ref_total);
    }

    // Write CSV
    {
        std::ofstream csv(csv_file);
        if (csv.is_open()) {
            csv << "mode,threads,num_packets,num_signatures,vocab_size,"
                << "nnz_packets,nnz_signatures,"
                << "pcap_parse_ms,feature_extract_ms,csr_construct_ms,"
                << "sparse_compute_ms,host_to_device_ms,device_to_host_ms,"
                << "verification_ms,total_ms,"
                << "num_candidates,num_alerts,packets_per_sec,speedup\n";

            for (const auto& r : results) {
                double pps = (r.timing.total_ms > 0)
                    ? (r.num_packets / (r.timing.total_ms / 1000.0)) : 0;
                double speedup = (ref_total > 0 && r.timing.total_ms > 0)
                    ? (ref_total / r.timing.total_ms) : 1.0;

                csv << r.mode << "," << r.threads << "," << r.num_packets << ","
                    << r.num_sigs << "," << r.vocab_size << ","
                    << r.nnz_packets << "," << r.nnz_sigs << ","
                    << std::fixed << std::setprecision(4)
                    << r.timing.pcap_parse_ms << "," << r.timing.feature_extract_ms << ","
                    << r.timing.csr_construct_ms << "," << r.timing.sparse_compute_ms << ","
                    << r.timing.host_to_device_ms << "," << r.timing.device_to_host_ms << ","
                    << r.timing.verification_ms << "," << r.timing.total_ms << ","
                    << r.timing.num_candidates << "," << r.timing.num_alerts << ","
                    << std::setprecision(2) << pps << "," << std::setprecision(4) << speedup << "\n";
            }
            std::cout << "\nBenchmark CSV written to: " << csv_file << "\n";
        }
    }

    // Memory estimate
    std::cout << "\n--- Memory Usage Estimate ---\n";
    size_t pkt_mem = 0;
    for (const auto& p : packets) pkt_mem += p.payload.size() + 64;
    size_t p_mem = P_info.row_ptr.size() * 4 + P_info.col_idx.size() * 4 + P_info.values.size() * 4;
    size_t s_mem = S_info.row_ptr.size() * 4 + S_info.col_idx.size() * 4 + S_info.values.size() * 4;
    std::cout << "Packets data:  " << pkt_mem / 1024 << " KB\n"
              << "Matrix P:      " << p_mem / 1024 << " KB\n"
              << "Matrix S:      " << s_mem / 1024 << " KB\n";

    // Roofline estimate
    std::cout << "\n--- Roofline Analysis ---\n";
    // SpMM operations: for each nonzero in result, we do ~1 FLOP (multiply+accumulate)
    // Data movement: reading A's row pointers + col indices, B's row pointers + col indices
    double total_flops = 0;
    for (int i = 0; i < P_info.num_rows; ++i) {
        int a_nnz_row = P_info.row_ptr[i+1] - P_info.row_ptr[i];
        for (int j = 0; j < S_info.num_rows; ++j) {
            int b_nnz_row = S_info.row_ptr[j+1] - S_info.row_ptr[j];
            // Merge-based comparison: O(a_nnz + b_nnz) comparisons
            total_flops += a_nnz_row + b_nnz_row;
        }
    }
    // Data bytes accessed: all of A and B arrays
    double data_bytes = (P_info.nnz + S_info.nnz) * (4 + 4) + // col_idx + values
                        (P_info.num_rows + S_info.num_rows + 2) * 4; // row_ptrs
    double ai = total_flops / data_bytes;
    std::cout << "Total operations (merge comparisons): " << total_flops << "\n"
              << "Data movement (bytes): " << data_bytes << "\n"
              << "Arithmetic intensity: " << std::setprecision(4) << ai << " ops/byte\n"
              << "Classification: " << ((ai < 1.0) ? "MEMORY-BOUND" : "COMPUTE-BOUND") << "\n"
              << "Note: Sparse irregular access patterns further limit effective bandwidth.\n";

    return 0;
}
