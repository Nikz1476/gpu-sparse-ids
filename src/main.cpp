// Main IDS executable — runs the full pipeline
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

using namespace ids;
using Clock = std::chrono::high_resolution_clock;

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <pcap_file> <signature_file> [mode] [threads] [ngram_size] [vocab_size]\n"
              << "  mode: seq | omp | cuda  (default: seq)\n"
              << "  threads: number of OpenMP threads (default: auto)\n"
              << "  ngram_size: n-gram size (default: 3)\n"
              << "  vocab_size: vocabulary/hash-table size (default: 65536)\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    std::string pcap_file = argv[1];
    std::string sig_file  = argv[2];
    std::string mode      = (argc > 3) ? argv[3] : "seq";
    int num_threads       = (argc > 4) ? std::atoi(argv[4]) : 0;
    int ngram_size        = (argc > 5) ? std::atoi(argv[5]) : 3;
    int vocab_size        = (argc > 6) ? std::atoi(argv[6]) : 65536;

    std::cout << "========================================\n"
              << " GPU Sparse IDS - Network Intrusion Detection\n"
              << "========================================\n"
              << "PCAP:       " << pcap_file << "\n"
              << "Signatures: " << sig_file << "\n"
              << "Mode:       " << mode << "\n"
              << "N-gram:     " << ngram_size << "\n"
              << "Vocab size: " << vocab_size << "\n"
              << "========================================\n\n";

    // 1. Read PCAP
    auto t0 = Clock::now();
    auto packets = PcapReader::read_pcap(pcap_file);
    auto t1 = Clock::now();
    double pcap_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "Loaded " << packets.size() << " packets in "
              << pcap_ms << " ms\n";

    if (packets.empty()) {
        std::cerr << "No packets loaded.\n";
        return 1;
    }

    // 2. Load signatures
    auto signatures = SignatureDB::load_signatures(sig_file);
    std::cout << "Loaded " << signatures.size() << " signatures\n\n";

    if (signatures.empty()) {
        std::cerr << "No signatures loaded.\n";
        return 1;
    }

    // 3. Create feature extractor
    NGramFeatureExtractor extractor(ngram_size, vocab_size);

    // 4. Run detection
    TimingResult timing;
    timing.pcap_parse_ms = pcap_ms;
    std::vector<Alert> alerts;

    if (mode == "seq") {
        std::cout << "--- Sequential CPU Execution ---\n";
        alerts = CPUEngine::run_sequential(packets, signatures, extractor, timing);
    } else if (mode == "omp") {
        std::cout << "--- OpenMP Parallel Execution ---\n";
        alerts = OpenMPEngine::run_parallel(packets, signatures, extractor, timing, num_threads);
    } else if (mode == "cuda") {
        if (!CUDAEngine::is_available()) {
            std::cerr << "CUDA is not available on this system.\n";
            std::cerr << "Falling back to sequential CPU.\n";
            alerts = CPUEngine::run_sequential(packets, signatures, extractor, timing);
        } else {
            std::cout << "--- CUDA GPU Execution ---\n";
            alerts = CUDAEngine::run_gpu(packets, signatures, extractor, timing);
        }
    } else {
        std::cerr << "Unknown mode: " << mode << "\n";
        return 1;
    }

    // 5. Print results
    AlertManager::print_alerts(alerts);

    // 6. Print timing
    std::cout << "\n--- Timing Results ---\n"
              << "PCAP parsing:      " << timing.pcap_parse_ms << " ms\n"
              << "Feature extraction: " << timing.feature_extract_ms << " ms\n"
              << "CSR construction:  " << timing.csr_construct_ms << " ms\n"
              << "Sparse computation: " << timing.sparse_compute_ms << " ms\n";
    if (timing.host_to_device_ms > 0)
        std::cout << "Host->Device:      " << timing.host_to_device_ms << " ms\n";
    if (timing.device_to_host_ms > 0)
        std::cout << "Device->Host:      " << timing.device_to_host_ms << " ms\n";
    std::cout << "Verification:      " << timing.verification_ms << " ms\n"
              << "Total:             " << timing.total_ms << " ms\n"
              << "Candidates:        " << timing.num_candidates << "\n"
              << "Alerts:            " << timing.num_alerts << "\n";

    double pps = (timing.total_ms > 0)
        ? (packets.size() / (timing.total_ms / 1000.0)) : 0;
    std::cout << "Packets/sec:       " << pps << "\n";

    return 0;
}

