#include "compute/cpu_engine.h"
#include "sparse/csr_matrix.h"
#include <chrono>
#include <iostream>
#include <algorithm>
#include <cstring>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace ids {

using Clock = std::chrono::high_resolution_clock;

static double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ──────────────────────────────────────────────────────────────────────
// SIMD-Vectorized Scanning (AVX2) — exact byte substring search
//
// Modeled after production engines (Intel Hyperscan, Ripgrep):
//   1. Broadcast the FIRST byte of the needle into a 256-bit register.
//   2. Stream 32 bytes of haystack per iteration with _mm256_loadu_si256.
//   3. Compare all 32 bytes in parallel: _mm256_cmpeq_epi8.
//   4. Extract a 32-bit hit mask: _mm256_movemask_epi8.
//   5. For each set bit, call memcmp at that offset for the full needle.
//
// Fallback: scalar memcmp loop for non-AVX2 targets.
// ──────────────────────────────────────────────────────────────────────
static bool contains_pattern(const std::vector<uint8_t>& haystack,
                             const std::vector<uint8_t>& needle) {
    if (needle.empty()) return true;
    if (haystack.size() < needle.size()) return false;

    const size_t n = haystack.size();
    const size_t m = needle.size();
    const uint8_t* hay = haystack.data();
    const uint8_t* ndl = needle.data();

    // Fast-path for single-byte patterns
    if (m == 1) {
        return std::memchr(hay, ndl[0], n) != nullptr;
    }

    // Last valid starting position for a full match
    const size_t last_pos = n - m;

#ifdef __AVX2__
    // ── AVX2 vectorized first-character probe ──────────────────────
    // Broadcast needle[0] across all 32 lanes of a 256-bit register
    const __m256i first_byte = _mm256_set1_epi8(static_cast<char>(ndl[0]));

    size_t pos = 0;
    // Process 32-byte aligned chunks
    while (pos + 31 <= last_pos) {
        // Load 32 bytes of haystack (unaligned load is fine on modern CPUs)
        __m256i chunk = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(hay + pos));

        // Compare all 32 bytes against needle[0] simultaneously
        __m256i cmp = _mm256_cmpeq_epi8(chunk, first_byte);

        // Extract a 32-bit mask: bit i is set if hay[pos+i] == needle[0]
        uint32_t mask = static_cast<uint32_t>(_mm256_movemask_epi8(cmp));

        // Process each hit in the mask
        while (mask != 0) {
            // Find lowest set bit → offset within this 32-byte chunk
            int bit = __builtin_ctz(mask);
            size_t candidate = pos + static_cast<size_t>(bit);

            // Bounds check: ensure full needle fits from this position
            if (candidate <= last_pos) {
                // Full verification: compare remaining bytes of needle
                if (std::memcmp(hay + candidate + 1, ndl + 1, m - 1) == 0) {
                    return true;
                }
            }
            // Clear the lowest set bit and continue
            mask &= mask - 1;
        }
        pos += 32;
    }

    // ── Scalar tail: handle remaining bytes not covered by 32-byte chunks ──
    for (size_t i = pos; i <= last_pos; ++i) {
        if (hay[i] == ndl[0] &&
            std::memcmp(hay + i + 1, ndl + 1, m - 1) == 0) {
            return true;
        }
    }

    return false;

#else
    // ── Scalar fallback for non-AVX2 targets ──────────────────────
    for (size_t i = 0; i <= last_pos; ++i) {
        if (hay[i] == ndl[0] &&
            std::memcmp(hay + i + 1, ndl + 1, m - 1) == 0) {
            return true;
        }
    }
    return false;
#endif
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
