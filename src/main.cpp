// Main IDS executable — runs the full pipeline
// Usage: sparse_ids <pcap> <sigs> [mode] [threads] [ngram_size] [vocab_size]
//                   [--out-csv <path>] [--out-json <path>]
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
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <algorithm>

using namespace ids;
using Clock = std::chrono::high_resolution_clock;

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <pcap_file> <sig_file> [mode] [threads] [ngram_size] [vocab_size]\n"
              << "         [--out-csv <path>] [--out-json <path>]\n"
              << "  mode:       seq | omp | cuda  (default: seq)\n"
              << "  threads:    OpenMP thread count (default: auto)\n"
              << "  ngram_size: N-gram window (default: 3)\n"
              << "  vocab_size: Hash vocabulary size (default: 65536)\n"
              << "  --out-csv   Write alerts CSV to path\n"
              << "  --out-json  Write alerts+summary JSON to path\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    std::string pcap_file  = argv[1];
    std::string sig_file   = argv[2];
    std::string mode       = "seq";
    int num_threads        = 0;
    int ngram_size         = 3;
    int vocab_size         = 65536;
    std::string out_csv;
    std::string out_json;

    // Parse positional args
    if (argc > 3) mode        = argv[3];
    if (argc > 4) num_threads = std::atoi(argv[4]);
    if (argc > 5) ngram_size  = std::atoi(argv[5]);
    if (argc > 6) vocab_size  = std::atoi(argv[6]);

    // Parse keyword args
    for (int i = 7; i < argc; ++i) {
        if (std::string(argv[i]) == "--out-csv" && i + 1 < argc)
            out_csv = argv[++i];
        else if (std::string(argv[i]) == "--out-json" && i + 1 < argc)
            out_json = argv[++i];
    }

    // Validation
    if (ngram_size < 1) { std::cerr << "ngram_size must be >= 1\n"; return 1; }
    if (vocab_size < 1) { std::cerr << "vocab_size must be >= 1\n"; return 1; }
    if (mode != "seq" && mode != "omp" && mode != "cuda") {
        std::cerr << "Unknown mode: " << mode << "\n";
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "========================================\n"
              << " GPU Sparse IDS — Network Intrusion Detection\n"
              << "========================================\n"
              << "PCAP:        " << pcap_file << "\n"
              << "Signatures:  " << sig_file << "\n"
              << "Mode:        " << mode << "\n"
              << "N-gram:      " << ngram_size << "\n"
              << "Vocab size:  " << vocab_size << "\n"
              << "========================================\n\n";

    // 1. Read PCAP
    auto t0 = Clock::now();
    auto packets = PcapReader::read_pcap(pcap_file);
    auto t1 = Clock::now();
    double pcap_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "Loaded " << packets.size() << " packets in " << pcap_ms << " ms\n";
    if (packets.empty()) {
        std::cerr << "No packets loaded — check PCAP file.\n";
        return 1;
    }

    // 2. Load signatures
    auto signatures = SignatureDB::load_signatures(sig_file);
    std::cout << "Loaded " << signatures.size() << " signatures\n\n";
    if (signatures.empty()) {
        std::cerr << "No signatures loaded — check signature file.\n";
        return 1;
    }

    // 3. Create feature extractor
    NGramFeatureExtractor extractor(ngram_size, vocab_size);

    // 4. Run detection pipeline
    TimingResult timing;
    timing.pcap_parse_ms = pcap_ms;
    std::vector<Alert> alerts;
    CSRMatrix P_info, S_info, C_info;

    if (mode == "seq") {
        std::cout << "--- Sequential CPU Execution ---\n";
        alerts = CPUEngine::run_sequential(packets, signatures, extractor, timing,
                                           &P_info, &S_info, &C_info);
    } else if (mode == "omp") {
        std::cout << "--- OpenMP Parallel Execution ---\n";
        alerts = OpenMPEngine::run_parallel(packets, signatures, extractor, timing,
                                            num_threads, &P_info, &S_info, &C_info);
    } else if (mode == "cuda") {
        if (!CUDAEngine::is_available()) {
            std::cerr << "CUDA not available. Falling back to sequential CPU.\n";
            alerts = CPUEngine::run_sequential(packets, signatures, extractor, timing,
                                               &P_info, &S_info, &C_info);
        } else {
            std::cout << "--- CUDA GPU Execution ---\n";
            alerts = CUDAEngine::run_gpu(packets, signatures, extractor, timing);
        }
    }

    // 5. Print alerts
    AlertManager::print_alerts(alerts);

    // 6. Build execution summary
    ExecutionSummary summary;
    summary.mode           = (mode == "seq") ? "Sequential" :
                             (mode == "omp") ? "OpenMP" : "CUDA";
    summary.num_threads    = num_threads <= 0 ? 1 : num_threads;
    summary.num_packets    = static_cast<int>(packets.size());
    summary.num_signatures = static_cast<int>(signatures.size());
    summary.num_candidates = timing.num_candidates;
    summary.num_alerts     = static_cast<int>(alerts.size());
    summary.false_positive_candidates = timing.num_candidates - static_cast<int>(alerts.size());
    summary.ngram_size     = ngram_size;
    summary.vocab_size     = vocab_size;
    summary.nnz_packets    = P_info.nnz;
    summary.nnz_sigs       = S_info.nnz;
    summary.C_nnz          = C_info.nnz;
    summary.sparsity_P     = P_info.sparsity();
    summary.sparsity_S     = S_info.sparsity();
    summary.sparsity_C     = C_info.sparsity();
    summary.timing         = timing;

    AlertManager::print_summary(summary);

    // 7. Optional CSV export
    if (!out_csv.empty()) {
        if (AlertManager::write_csv(out_csv, alerts))
            std::cout << "Alerts CSV written to: " << out_csv << "\n";
        else
            std::cerr << "Failed to write CSV to: " << out_csv << "\n";
    }

    // 8. Optional JSON export
    if (!out_json.empty()) {
        if (AlertManager::write_json(out_json, alerts, summary))
            std::cout << "Alerts JSON written to: " << out_json << "\n";
        else
            std::cerr << "Failed to write JSON to: " << out_json << "\n";
    }

    return 0;
}
