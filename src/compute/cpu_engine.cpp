#include "compute/cpu_engine.h"
#include "sparse/csr_matrix.h"
#include <chrono>
#include <iostream>
#include <algorithm>
#include <cstring>

namespace ids {

using Clock = std::chrono::high_resolution_clock;

static double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Exact byte substring search
static bool contains_pattern(const std::vector<uint8_t>& haystack,
                             const std::vector<uint8_t>& needle) {
    if (needle.empty()) return true;
    if (haystack.size() < needle.size()) return false;

    size_t n = haystack.size();
    size_t m = needle.size();

    for (size_t i = 0; i <= n - m; ++i) {
        if (memcmp(haystack.data() + i, needle.data(), m) == 0) {
            return true;
        }
    }
    return false;
}

std::vector<std::vector<int>> CPUEngine::extract_all_features(
    const std::vector<Packet>& packets,
    const NGramFeatureExtractor& extractor)
{
    std::vector<std::vector<int>> features(packets.size());
    for (size_t i = 0; i < packets.size(); ++i) {
        features[i] = extractor.extract_features(packets[i].payload);
    }
    return features;
}

std::vector<std::vector<int>> CPUEngine::extract_signature_features(
    std::vector<Signature>& signatures,
    const NGramFeatureExtractor& extractor)
{
    std::vector<std::vector<int>> features(signatures.size());
    for (size_t i = 0; i < signatures.size(); ++i) {
        features[i] = extractor.extract_features(signatures[i].pattern);
        signatures[i].num_features = static_cast<uint32_t>(features[i].size());
    }
    return features;
}

std::vector<CandidatePair> CPUEngine::generate_candidates(
    const CSRMatrix& C,
    const std::vector<Signature>& signatures,
    const NGramFeatureExtractor& extractor,
    int num_packets)
{
    std::vector<CandidatePair> candidates;

    // --- Standard sparse-filter candidates ---
    // For signatures whose pattern length >= ngram_size, the n-gram feature
    // representation is sound: if a packet contains the signature as a
    // contiguous substring, ALL signature n-grams must appear in the packet.
    // Therefore score >= num_features is a necessary condition (no false negatives).
    for (int i = 0; i < C.num_rows; ++i) {
        for (int idx = C.row_ptr[i]; idx < C.row_ptr[i + 1]; ++idx) {
            int j = C.col_idx[idx];
            float score = C.values[idx];

            // Only consider signatures that have enough n-gram features
            // for the sparse filter to be meaningful
            if (signatures[j].num_features == 0) continue;
            if (static_cast<int>(signatures[j].pattern.size()) < extractor.ngram_size()) continue;

            // Candidate if all signature features are present in the packet
            if (static_cast<uint32_t>(score + 0.5f) >= signatures[j].num_features) {
                CandidatePair cp;
                cp.packet_idx = i;
                cp.signature_idx = j;
                cp.score = score;
                candidates.push_back(cp);
            }
        }
    }

    // --- Short-signature bypass ---
    // Signatures whose pattern is shorter than ngram_size cannot be reliably
    // filtered using n-gram features (the hash of the short pattern will NOT
    // appear in the packet's standard n-gram feature set). These signatures
    // must be sent directly to exact verification against ALL packets.
    // This is safe because short signatures are rare and verification is fast.
    for (int j = 0; j < static_cast<int>(signatures.size()); ++j) {
        const auto& sig = signatures[j];
        if (sig.pattern.empty()) continue;
        if (static_cast<int>(sig.pattern.size()) >= extractor.ngram_size()) continue;

        // Pair this short signature with every packet
        for (int i = 0; i < num_packets; ++i) {
            CandidatePair cp;
            cp.packet_idx = i;
            cp.signature_idx = j;
            cp.score = 0.0f;  // no sparse score for short sigs
            candidates.push_back(cp);
        }
    }

    return candidates;
}

std::vector<Alert> CPUEngine::verify_candidates(
    const std::vector<CandidatePair>& candidates,
    const std::vector<Packet>& packets,
    const std::vector<Signature>& signatures)
{
    std::vector<Alert> alerts;

    for (const auto& cp : candidates) {
        const Packet& pkt = packets[cp.packet_idx];
        const Signature& sig = signatures[cp.signature_idx];

        // Protocol check
        if (sig.protocol != 0 && sig.protocol != pkt.protocol) continue;
        // Port check
        if (sig.dst_port != 0 && sig.dst_port != pkt.dst_port) continue;

        // Exact byte-level substring match
        if (contains_pattern(pkt.payload, sig.pattern)) {
            Alert alert;
            alert.packet_id = pkt.id;
            alert.signature_id = sig.id;
            alert.signature_name = sig.name;
            alert.protocol = pkt.protocol;
            alert.src_ip = pkt.src_ip;
            alert.dst_ip = pkt.dst_ip;
            alert.src_port = pkt.src_port;
            alert.dst_port = pkt.dst_port;
            alert.timestamp = pkt.timestamp;
            alert.match_detail = "Exact match: " + sig.description;
            alerts.push_back(std::move(alert));
        }
    }

    return alerts;
}

std::vector<Alert> CPUEngine::run_sequential(
    const std::vector<Packet>& packets,
    const std::vector<Signature>& signatures_in,
    const NGramFeatureExtractor& extractor,
    TimingResult& timing,
    CSRMatrix* out_P,
    CSRMatrix* out_S,
    CSRMatrix* out_C)
{
    // Make a mutable copy for setting num_features
    std::vector<Signature> signatures = signatures_in;

    auto t0 = Clock::now();

    // 1. Feature extraction - packets
    auto t1 = Clock::now();
    auto packet_features = extract_all_features(packets, extractor);
    auto t2 = Clock::now();

    // 2. Feature extraction - signatures
    auto sig_features = extract_signature_features(signatures, extractor);
    auto t3 = Clock::now();

    timing.feature_extract_ms = elapsed_ms(t1, t3);

    // 3. Build CSR matrices
    auto t4 = Clock::now();
    CSRMatrix P = CSRBuilder::build_from_features(packet_features, extractor.vocab_size());
    CSRMatrix S = CSRBuilder::build_from_features(sig_features, extractor.vocab_size());
    auto t5 = Clock::now();
    timing.csr_construct_ms = elapsed_ms(t4, t5);

    CSRBuilder::print_info(P, "P (packets)");
    CSRBuilder::print_info(S, "S (signatures)");

    // 4. Sparse multiplication: C = P * S^T
    auto t6 = Clock::now();
    CSRMatrix C = CSRBuilder::spmm_transpose(P, S);
    auto t7 = Clock::now();
    timing.sparse_compute_ms = elapsed_ms(t6, t7);

    CSRBuilder::print_info(C, "C (scores)");

    // 5. Generate candidates (includes short-signature bypass)
    auto candidates = generate_candidates(C, signatures, extractor,
                                          static_cast<int>(packets.size()));
    timing.num_candidates = static_cast<int>(candidates.size());
    std::cout << "Candidates: " << candidates.size() << std::endl;

    // 6. Verify candidates
    auto t8 = Clock::now();
    auto alerts = verify_candidates(candidates, packets, signatures);
    auto t9 = Clock::now();
    timing.verification_ms = elapsed_ms(t8, t9);
    timing.num_alerts = static_cast<int>(alerts.size());

    timing.total_ms = elapsed_ms(t0, t9);

    // Optionally export matrix info
    if (out_P) *out_P = P;
    if (out_S) *out_S = S;
    if (out_C) *out_C = C;

    return alerts;
}

} // namespace ids
