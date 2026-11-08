// Correctness test suite for GPU Sparse IDS
// Tests feature extraction, CSR construction, SpMM, candidate generation,
// verification, and end-to-end detection with ground-truth validation.

#include "ingestion/pcap_reader.h"
#include "signatures/signature_db.h"
#include "features/ngram_features.h"
#include "sparse/csr_matrix.h"
#include "compute/cpu_engine.h"
#include "compute/openmp_engine.h"
#include "output/alert_manager.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <set>
#include <map>
#include <algorithm>
#include <cstring>
#include <random>
#include <fstream>
#include <pcap/pcap.h>

using namespace ids;

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    std::cout << "TEST: " << name << " ... ";

#define PASS() \
    do { std::cout << "PASSED" << std::endl; tests_passed++; } while(0)

#define FAIL(msg) \
    do { std::cout << "FAILED: " << msg << std::endl; tests_failed++; } while(0)

// ============================================================
// Unit Tests — N-gram Features
// ============================================================

void test_ngram_empty_payload() {
    TEST("NGram: empty payload");
    NGramFeatureExtractor ext(3, 1024);
    std::vector<uint8_t> empty;
    auto feats = ext.extract_features(empty);
    if (feats.empty()) PASS(); else FAIL("Expected empty features for empty payload");
}

void test_ngram_short_payload() {
    TEST("NGram: short payload (< ngram_size)");
    NGramFeatureExtractor ext(3, 1024);
    std::vector<uint8_t> data = {0x41, 0x42}; // "AB"
    auto feats = ext.extract_features(data);
    if (feats.size() == 1) PASS(); else FAIL("Expected 1 feature for 2-byte payload with n=3, got " << feats.size());
}

void test_ngram_exact_ngram_size() {
    TEST("NGram: payload == ngram_size");
    NGramFeatureExtractor ext(3, 1024);
    std::vector<uint8_t> data = {0x41, 0x42, 0x43}; // "ABC"
    auto feats = ext.extract_features(data);
    if (feats.size() == 1) PASS(); else FAIL("Expected 1 feature for 3-byte payload with n=3, got " << feats.size());
}

void test_ngram_deterministic() {
    TEST("NGram: deterministic hashing");
    NGramFeatureExtractor ext(3, 65536);
    std::vector<uint8_t> data = {0x41, 0x42, 0x43, 0x44, 0x45};
    auto f1 = ext.extract_features(data);
    auto f2 = ext.extract_features(data);
    if (f1 == f2) PASS(); else FAIL("Features should be deterministic");
}

void test_ngram_unique_and_sorted() {
    TEST("NGram: features are unique and sorted");
    NGramFeatureExtractor ext(3, 65536);
    // "AAAA" has n-grams "AAA", "AAA" — should produce only 1 unique feature
    std::vector<uint8_t> data = {0x41, 0x41, 0x41, 0x41};
    auto feats = ext.extract_features(data);
    bool sorted = std::is_sorted(feats.begin(), feats.end());
    auto it = std::unique(feats.begin(), feats.end());
    bool unique = (it == feats.end());
    if (sorted && unique) PASS(); else FAIL("Features must be sorted and unique");
}

void test_ngram_identical_for_same_content() {
    TEST("NGram: packet and signature with same bytes produce same features");
    NGramFeatureExtractor ext(3, 65536);
    std::string s = "test_pattern_data";
    std::vector<uint8_t> data(s.begin(), s.end());
    auto f1 = ext.extract_features(data);
    auto f2 = ext.extract_features(data);
    if (f1 == f2) PASS(); else FAIL("Same content must produce identical features");
}

// ============================================================
// Unit Tests — CSR Matrix
// ============================================================

void test_csr_build() {
    TEST("CSR: build from features");
    std::vector<std::vector<int>> features = {{0, 2, 5}, {1, 3}, {}, {0, 1, 2, 3, 4, 5}};
    CSRMatrix mat = CSRBuilder::build_from_features(features, 10);
    bool ok = (mat.num_rows == 4 && mat.num_cols == 10 && mat.nnz == 11);
    ok = ok && (mat.row_ptr[0] == 0 && mat.row_ptr[1] == 3 && mat.row_ptr[2] == 5
                && mat.row_ptr[3] == 5 && mat.row_ptr[4] == 11);
    ok = ok && (static_cast<int>(mat.row_ptr.size()) == mat.num_rows + 1);
    ok = ok && (mat.row_ptr.back() == mat.nnz);
    if (ok) PASS(); else FAIL("CSR structure mismatch");
}

void test_csr_empty_matrix() {
    TEST("CSR: empty matrix");
    std::vector<std::vector<int>> features = {{}, {}, {}};
    CSRMatrix mat = CSRBuilder::build_from_features(features, 5);
    bool ok = (mat.num_rows == 3 && mat.num_cols == 5 && mat.nnz == 0);
    ok = ok && (mat.row_ptr.size() == 4);
    for (auto v : mat.row_ptr) ok = ok && (v == 0);
    if (ok) PASS(); else FAIL("Empty CSR should have all-zero row_ptr");
}

void test_csr_row_ptr_monotonic() {
    TEST("CSR: row_ptr is non-decreasing");
    NGramFeatureExtractor ext(3, 256);
    std::vector<Packet> packets(50);
    std::mt19937 rng(123);
    for (auto& p : packets) {
        int len = rng() % 200;
        p.payload.resize(len);
        for (auto& b : p.payload) b = rng() % 256;
    }
    auto feats = CPUEngine::extract_all_features(packets, ext);
    CSRMatrix P = CSRBuilder::build_from_features(feats, 256);

    bool monotonic = true;
    for (int i = 0; i < P.num_rows; ++i) {
        if (P.row_ptr[i] > P.row_ptr[i + 1]) {
            monotonic = false;
            break;
        }
    }
    bool final_ok = (P.row_ptr.back() == P.nnz);
    if (monotonic && final_ok) PASS();
    else FAIL("row_ptr not monotonic or final != nnz");
}

// ============================================================
// Unit Tests — SpMM
// ============================================================

void test_spmm_known_products() {
    TEST("SpMM: known dot products");
    // A = [[1,0,1], [0,1,0]]  (2x3)
    // B = [[1,1,0], [0,0,1]]  (2x3)
    // C = A * B^T = [[1, 1], [1, 0]]
    std::vector<std::vector<int>> fa = {{0, 2}, {1}};
    std::vector<std::vector<int>> fb = {{0, 1}, {2}};
    CSRMatrix A = CSRBuilder::build_from_features(fa, 3);
    CSRMatrix B = CSRBuilder::build_from_features(fb, 3);
    CSRMatrix C = CSRBuilder::spmm_transpose(A, B);

    bool ok = (C.num_rows == 2 && C.num_cols == 2 && C.nnz == 3);
    // Row 0: (col 0, val 1), (col 1, val 1)
    // Row 1: (col 0, val 1)
    if (ok && C.row_ptr[0] == 0 && C.row_ptr[1] == 2 && C.row_ptr[2] == 3)
        PASS();
    else
        FAIL("SpMM result incorrect, nnz=" << C.nnz);
}

void test_spmm_vs_dense_reference() {
    TEST("SpMM: sparse vs dense reference on random matrices");
    // Create small random sparse matrices and compare sparse SpMM against dense
    std::mt19937 rng(42);
    int M = 10, K = 20, N = 5;

    // Random feature lists for A (M x K) and B (N x K)
    std::vector<std::vector<int>> fa(M), fb(N);
    for (int i = 0; i < M; ++i) {
        std::set<int> cols;
        int nnz = 1 + rng() % 8;
        for (int j = 0; j < nnz; ++j) cols.insert(rng() % K);
        fa[i].assign(cols.begin(), cols.end());
    }
    for (int i = 0; i < N; ++i) {
        std::set<int> cols;
        int nnz = 1 + rng() % 6;
        for (int j = 0; j < nnz; ++j) cols.insert(rng() % K);
        fb[i].assign(cols.begin(), cols.end());
    }

    CSRMatrix A = CSRBuilder::build_from_features(fa, K);
    CSRMatrix B = CSRBuilder::build_from_features(fb, K);
    CSRMatrix C = CSRBuilder::spmm_transpose(A, B);

    // Dense reference: C_dense[i][j] = |fa[i] ∩ fb[j]|
    bool ok = true;
    for (int i = 0; i < M && ok; ++i) {
        std::set<int> a_set(fa[i].begin(), fa[i].end());
        int sparse_idx = C.row_ptr[i];
        for (int j = 0; j < N && ok; ++j) {
            // Compute dense intersection count
            float dense_val = 0;
            for (int c : fb[j]) {
                if (a_set.count(c)) dense_val += 1.0f;
            }
            // Find sparse value
            float sparse_val = 0;
            if (sparse_idx < C.row_ptr[i + 1] && C.col_idx[sparse_idx] == j) {
                sparse_val = C.values[sparse_idx];
                ++sparse_idx;
            }
            if (std::abs(dense_val - sparse_val) > 1e-6f) {
                std::cout << "\n  Mismatch at C[" << i << "][" << j
                          << "]: sparse=" << sparse_val << " dense=" << dense_val;
                ok = false;
            }
        }
    }
    if (ok) PASS(); else { std::cout << "\n"; FAIL("Sparse vs dense mismatch"); }
}

// ============================================================
// Unit Tests — Candidate Generation & Verification
// ============================================================

void test_candidate_threshold() {
    TEST("Candidate generation: threshold filtering");
    NGramFeatureExtractor ext(3, 1024);

    Signature sig;
    sig.id = 1;
    sig.name = "test";
    sig.num_features = 3;
    sig.pattern = {0x41, 0x42, 0x43, 0x44, 0x45}; // len=5 >= ngram_size=3

    CSRMatrix C;
    C.num_rows = 2;
    C.num_cols = 1;
    C.nnz = 2;
    C.row_ptr = {0, 1, 2};
    C.col_idx = {0, 0};
    C.values = {3.0f, 2.0f};  // pkt0 matches 3/3, pkt1 matches 2/3

    auto cands = CPUEngine::generate_candidates(C, {sig}, ext, 2);
    if (cands.size() == 1 && cands[0].packet_idx == 0)
        PASS();
    else
        FAIL("Expected 1 candidate (packet 0), got " << cands.size());
}

void test_short_sig_bypass() {
    TEST("Short signature: bypass sparse filter");
    NGramFeatureExtractor ext(3, 1024);

    Signature sig;
    sig.id = 1;
    sig.name = "short";
    sig.pattern = {0x41, 0x42}; // len=2 < ngram_size=3
    sig.num_features = 1;       // has a short-content feature

    // Empty SpMM result (no sparse matches possible for short sigs)
    CSRMatrix C;
    C.num_rows = 3;
    C.num_cols = 1;
    C.nnz = 0;
    C.row_ptr = {0, 0, 0, 0};

    auto cands = CPUEngine::generate_candidates(C, {sig}, ext, 3);
    // Short sig should be paired with ALL 3 packets
    if (cands.size() == 3) PASS();
    else FAIL("Expected 3 candidates for short sig bypass, got " << cands.size());
}

void test_exact_verification() {
    TEST("Verification: exact byte match");
    Packet pkt;
    pkt.id = 0;
    pkt.protocol = 6;
    pkt.src_ip = "1.1.1.1"; pkt.dst_ip = "2.2.2.2";
    pkt.src_port = 1234; pkt.dst_port = 80;
    std::string payload = "Hello world test pattern here";
    pkt.payload.assign(payload.begin(), payload.end());

    Signature sig;
    sig.id = 1; sig.name = "TEST"; sig.description = "test";
    sig.protocol = 6; sig.dst_port = 0;
    std::string pat = "test pattern";
    sig.pattern.assign(pat.begin(), pat.end());
    sig.num_features = 5;

    CandidatePair cp{0, 0, 5.0f};
    auto alerts = CPUEngine::verify_candidates({cp}, {pkt}, {sig});
    if (alerts.size() == 1) PASS(); else FAIL("Expected 1 alert, got " << alerts.size());
}

void test_verification_rejects_non_match() {
    TEST("Verification: rejects non-substring");
    Packet pkt;
    pkt.id = 0; pkt.protocol = 6;
    pkt.src_ip = "1.1.1.1"; pkt.dst_ip = "2.2.2.2";
    pkt.src_port = 1234; pkt.dst_port = 80;
    std::string payload = "Hello world";
    pkt.payload.assign(payload.begin(), payload.end());

    Signature sig;
    sig.id = 1; sig.name = "TEST"; sig.description = "test";
    sig.protocol = 6; sig.dst_port = 0;
    std::string pat = "not present";
    sig.pattern.assign(pat.begin(), pat.end());
    sig.num_features = 5;

    CandidatePair cp{0, 0, 5.0f};
    auto alerts = CPUEngine::verify_candidates({cp}, {pkt}, {sig});
    if (alerts.empty()) PASS(); else FAIL("Expected 0 alerts for non-match");
}

void test_verification_protocol_filter() {
    TEST("Verification: protocol mismatch rejects");
    Packet pkt;
    pkt.id = 0; pkt.protocol = 17; // UDP
    pkt.src_ip = "1.1.1.1"; pkt.dst_ip = "2.2.2.2";
    pkt.src_port = 1234; pkt.dst_port = 80;
    std::string payload = "test pattern";
    pkt.payload.assign(payload.begin(), payload.end());

    Signature sig;
    sig.id = 1; sig.name = "TEST"; sig.description = "test";
    sig.protocol = 6; // TCP only
    sig.dst_port = 0;
    sig.pattern.assign(payload.begin(), payload.end());
    sig.num_features = 5;

    CandidatePair cp{0, 0, 5.0f};
    auto alerts = CPUEngine::verify_candidates({cp}, {pkt}, {sig});
    if (alerts.empty()) PASS(); else FAIL("Should reject: UDP packet vs TCP-only signature");
}

void test_verification_port_filter() {
    TEST("Verification: port mismatch rejects");
    Packet pkt;
    pkt.id = 0; pkt.protocol = 6;
    pkt.src_ip = "1.1.1.1"; pkt.dst_ip = "2.2.2.2";
    pkt.src_port = 1234; pkt.dst_port = 5353; // wrong port

    std::string payload = "AAAA";
    pkt.payload.assign(payload.begin(), payload.end());

    Signature sig;
    sig.id = 1; sig.name = "TEST"; sig.description = "test";
    sig.protocol = 0;    // any protocol
    sig.dst_port = 53;   // requires port 53
    sig.pattern.assign(payload.begin(), payload.end());
    sig.num_features = 1;

    CandidatePair cp{0, 0, 1.0f};
    auto alerts = CPUEngine::verify_candidates({cp}, {pkt}, {sig});
    if (alerts.empty()) PASS(); else FAIL("Should reject: wrong destination port");
}

// ============================================================
// Property Test — No False Negatives
// ============================================================

void test_no_false_negatives_property() {
    TEST("Property: n-gram filter + short-sig bypass = no false negatives");
    NGramFeatureExtractor ext(3, 65536);

    // Test patterns of various lengths including short ones
    std::vector<std::string> patterns = {
        "abcdef", "' OR 1=1--", "<script>alert(", "xy", "a",
        "AAAA", "\x90\x90\x90\x90\x90", "AB", "X"
    };

    bool all_ok = true;
    for (const auto& pat_str : patterns) {
        Signature sig;
        sig.id = 1;
        sig.name = "test";
        sig.pattern.assign(pat_str.begin(), pat_str.end());

        // Create a packet containing this pattern
        std::string pkt_payload = "PREFIX_" + pat_str + "_SUFFIX";
        Packet pkt;
        pkt.id = 0;
        pkt.protocol = 6;
        pkt.dst_port = 80;
        pkt.payload.assign(pkt_payload.begin(), pkt_payload.end());

        // Extract features
        auto pkt_feats = ext.extract_features(pkt.payload);
        auto sig_feats = ext.extract_features(sig.pattern);
        sig.num_features = static_cast<uint32_t>(sig_feats.size());

        bool is_short = static_cast<int>(sig.pattern.size()) < ext.ngram_size();

        if (is_short) {
            // Short signatures bypass the sparse filter entirely —
            // they go straight to verification. No feature check needed.
            continue;
        }

        // For long signatures: all signature features must be present in packet
        std::set<int> pkt_set(pkt_feats.begin(), pkt_feats.end());
        for (int f : sig_feats) {
            if (pkt_set.count(f) == 0) {
                std::cout << "\n  Pattern \"" << pat_str << "\" (len=" << pat_str.size()
                          << "): signature feature " << f << " missing from packet";
                all_ok = false;
                break;
            }
        }
    }

    if (all_ok) PASS(); else { std::cout << "\n"; FAIL("False negative detected in n-gram filter"); }
}

void test_short_sig_e2e_detection() {
    TEST("Short signature: end-to-end detection");
    NGramFeatureExtractor ext(3, 65536);

    // Signature "xy" (2 bytes, shorter than n=3)
    Signature sig;
    sig.id = 999;
    sig.name = "SHORT_XY";
    sig.description = "Short pattern xy";
    sig.protocol = 0;
    sig.dst_port = 0;
    std::string pat = "xy";
    sig.pattern.assign(pat.begin(), pat.end());

    // Packet containing "xy"
    Packet pkt;
    pkt.id = 0;
    pkt.protocol = 6;
    pkt.dst_port = 80;
    pkt.src_ip = "1.1.1.1"; pkt.dst_ip = "2.2.2.2";
    pkt.src_port = 1234;
    std::string payload = "PREFIX_xy_SUFFIX";
    pkt.payload.assign(payload.begin(), payload.end());

    // Run sequential detection
    TimingResult timing;
    auto alerts = CPUEngine::run_sequential({pkt}, {sig}, ext, timing);

    if (alerts.size() == 1 && alerts[0].signature_id == 999)
        PASS();
    else
        FAIL("Expected 1 alert for short signature 'xy', got " << alerts.size());
}

// ============================================================
// End-to-End Integration Tests
// ============================================================

static std::string resolve_test_path(const std::string& p) {
    std::ifstream f(p);
    if (f.good()) return p;
    std::string up = "../" + p;
    std::ifstream f_up(up);
    if (f_up.good()) return up;
    return p;
}

void test_e2e_sequential() {
    TEST("E2E Sequential: synthetic PCAP ground truth");

    auto packets = PcapReader::read_pcap(resolve_test_path("data/pcap/test_traffic.pcap"));
    auto sigs = SignatureDB::load_signatures(resolve_test_path("data/signatures/test_signatures.txt"));

    if (packets.empty() || sigs.empty()) {
        FAIL("Could not load test data. Run generate_test_data first.");
        return;
    }

    std::cout << "(" << packets.size() << " pkts, " << sigs.size() << " sigs) ... ";

    NGramFeatureExtractor extractor(3, 65536);
    TimingResult timing;
    auto alerts = CPUEngine::run_sequential(packets, sigs, extractor, timing);

    // Build alert lookup: packet_id -> set of signature_ids
    std::map<uint32_t, std::set<uint32_t>> alert_map;
    for (const auto& a : alerts) {
        alert_map[a.packet_id].insert(a.signature_id);
    }

    bool all_ok = true;

    auto expect_match = [&](uint32_t pkt_id, uint32_t sig_id) {
        if (alert_map.count(pkt_id) == 0 || alert_map[pkt_id].count(sig_id) == 0) {
            std::cout << "\n  MISS: pkt " << pkt_id << " should match sig " << sig_id;
            all_ok = false;
        }
    };

    auto expect_no_match = [&](uint32_t pkt_id) {
        if (alert_map.count(pkt_id) > 0) {
            std::cout << "\n  FALSE POS: pkt " << pkt_id << " should have no alerts, got "
                      << alert_map[pkt_id].size();
            all_ok = false;
        }
    };

    expect_match(0, 1001);   // SQL injection
    expect_match(1, 1002);   // XSS
    expect_match(2, 1003);   // Command injection
    expect_match(3, 1004);   // Directory traversal
    expect_match(4, 1005);   // NOP sled
    expect_match(5, 1006);   // Malicious UA
    expect_match(6, 1007);   // DNS tunnel
    expect_match(7, 1008);   // PHP shell
    expect_no_match(8);      // Normal traffic
    expect_no_match(9);      // Empty payload
    // Pkt 10: near-miss "' OR 2=2--" should NOT match sig 1001 "' OR 1=1--"
    if (alert_map.count(10) > 0 && alert_map[10].count(1001) > 0) {
        std::cout << "\n  FALSE POS: pkt 10 (near-miss) should NOT match sig 1001";
        all_ok = false;
    }
    expect_match(11, 1001);  // Multi-sig: SQL
    expect_match(11, 1002);  // Multi-sig: XSS
    expect_no_match(12);     // Short payload ("AB" — no signature matches)
    // Pkt 13: AAAA on wrong port (5353 vs required 53)
    if (alert_map.count(13) > 0 && alert_map[13].count(1007) > 0) {
        std::cout << "\n  FALSE POS: pkt 13 should NOT match sig 1007 (wrong port)";
        all_ok = false;
    }
    expect_match(14, 1008);  // Repeated pattern

    std::cout << "[" << alerts.size() << " alerts] ";
    if (all_ok) PASS(); else { std::cout << "\n"; FAIL("Ground truth mismatch"); }
}

void test_e2e_openmp_matches_sequential() {
    TEST("E2E OpenMP: results match sequential");

    auto packets = PcapReader::read_pcap(resolve_test_path("data/pcap/test_traffic.pcap"));
    auto sigs = SignatureDB::load_signatures(resolve_test_path("data/signatures/test_signatures.txt"));

    if (packets.empty() || sigs.empty()) {
        FAIL("Could not load test data.");
        return;
    }

    NGramFeatureExtractor extractor(3, 65536);

    // Run sequential
    TimingResult t1;
    auto alerts_seq = CPUEngine::run_sequential(packets, sigs, extractor, t1);

    // Run OpenMP with 1, 2, and 4 threads
    std::vector<int> thread_counts = {1, 2, 4};
    auto make_set = [](const std::vector<Alert>& alerts) {
        std::set<std::pair<uint32_t, uint32_t>> s;
        for (const auto& a : alerts) s.insert({a.packet_id, a.signature_id});
        return s;
    };
    auto s_seq = make_set(alerts_seq);

    bool all_ok = true;
    for (int nt : thread_counts) {
        auto sigs_copy = sigs;
        TimingResult t2;
        auto alerts_omp = OpenMPEngine::run_parallel(packets, sigs_copy, extractor, t2, nt);
        auto s_omp = make_set(alerts_omp);

        if (s_seq != s_omp) {
            std::cout << "\n  Thread count " << nt << ": OpenMP alerts ("
                      << s_omp.size() << ") differ from sequential (" << s_seq.size() << ")";
            all_ok = false;
        }
    }

    if (all_ok) PASS(); else { std::cout << "\n"; FAIL("OpenMP/Sequential mismatch"); }
}

// ============================================================
// Hash Collision Documentation Test
// ============================================================

void test_hash_collision_effect() {
    TEST("Hash collisions: explained and bounded");
    // With vocab_size=65536 and FNV-1a, collisions are possible but rare.
    // A collision means two different n-grams map to the same feature index.
    // Effect: more candidates pass the sparse filter (false positives in
    // candidate set), but exact verification always rejects non-matches.
    // Therefore: collisions increase candidate count but NEVER cause
    // false alerts or false negatives.
    //
    // Measure collision rate on a sample:
    NGramFeatureExtractor ext(3, 65536);
    int total_ngrams = 0;
    int collisions = 0;
    std::map<int, int> idx_count;

    // Generate all possible 2-byte-prefix 3-grams for "A"-"Z"
    for (uint8_t a = 'A'; a <= 'Z'; ++a) {
        for (uint8_t b = 'A'; b <= 'Z'; ++b) {
            for (uint8_t c = 'A'; c <= 'Z'; ++c) {
                uint8_t d[3] = {a, b, c};
                int idx = ext.hash_ngram(d, 3);
                idx_count[idx]++;
                total_ngrams++;
            }
        }
    }
    for (auto& [idx, cnt] : idx_count) {
        if (cnt > 1) collisions += cnt - 1;
    }

    double collision_rate = 100.0 * collisions / total_ngrams;
    std::cout << "[" << collisions << "/" << total_ngrams
              << " collisions, " << collision_rate << "%] ";

    // Collisions should be low with 65536 buckets and ~17576 ngrams
    // Birthday paradox: expect some collisions, but < 50%
    if (collision_rate < 50.0) PASS();
    else FAIL("Collision rate too high: " << collision_rate << "%");
}

void test_pcap_hardening_and_malformed_packets() {
    TEST("PCAP Hardening: truncated, non-IPv4, and malformed header rejection");

    std::string test_pcap = "/tmp/malformed_test.pcap";
    pcap_t* dead = pcap_open_dead(DLT_RAW, 65535);
    if (!dead) {
        FAIL("Could not open dead pcap handle");
        return;
    }

    pcap_dumper_t* dumper = pcap_dump_open(dead, test_pcap.c_str());
    if (!dumper) {
        pcap_close(dead);
        FAIL("Could not open pcap dump file");
        return;
    }

    struct pcap_pkthdr h;
    memset(&h, 0, sizeof(h));

    // 1. Truncated packet (< 20 bytes)
    uint8_t pkt_trunc[10] = {0x45, 0x00, 0x00, 0x0A};
    h.caplen = 10; h.len = 10;
    pcap_dump((u_char*)dumper, &h, pkt_trunc);

    // 2. Non-IPv4 packet (IPv6 version 0x60)
    uint8_t pkt_ipv6[40] = {0x60, 0x00, 0x00, 0x00};
    h.caplen = 40; h.len = 40;
    pcap_dump((u_char*)dumper, &h, pkt_ipv6);

    // 3. Invalid TCP header length (offset < 20 bytes)
    uint8_t pkt_bad_tcp[40] = {
        0x45, 0x00, 0x00, 0x28, // IPv4, 40 bytes total len
        0x00, 0x01, 0x00, 0x00,
        0x40, 0x06, 0x00, 0x00, // protocol 6 (TCP)
        127, 0, 0, 1,           // src
        127, 0, 0, 1,           // dst
        0x00, 80, 0x00, 80,     // ports
        0, 0, 0, 0, 0, 0, 0, 0,
        0x10, 0x00,             // offset = 1 (4 bytes, < 20! INVALID)
        0, 0, 0, 0
    };
    h.caplen = 40; h.len = 40;
    pcap_dump((u_char*)dumper, &h, pkt_bad_tcp);

    // 4. Valid UDP packet with valid payload
    uint8_t pkt_valid[36] = {
        0x45, 0x00, 0x00, 36,   // IPv4, 36 bytes total len
        0x00, 0x02, 0x00, 0x00,
        0x40, 17,   0x00, 0x00, // protocol 17 (UDP)
        10, 0, 0, 1,
        10, 0, 0, 2,
        0x1F, 0x90, 0x00, 53,   // sport 8080, dport 53
        0x00, 16, 0x00, 0x00,   // UDP length = 16 (8 hdr + 8 payload)
        'H', 'A', 'R', 'D', 'E', 'N', 'E', 'D'
    };
    h.caplen = 36; h.len = 36;
    pcap_dump((u_char*)dumper, &h, pkt_valid);

    pcap_dump_close(dumper);
    pcap_close(dead);

    // Read back and verify only the valid packet was accepted
    auto ingested = PcapReader::read_pcap(test_pcap);
    if (ingested.size() != 1) {
        FAIL("Expected exactly 1 valid packet surviving malformed PCAP ingestion");
        return;
    }
    if (ingested[0].protocol != 17 || ingested[0].dst_port != 53) {
        FAIL("Surviving packet metadata corrupted");
        return;
    }
    std::string payload_str(ingested[0].payload.begin(), ingested[0].payload.end());
    if (payload_str != "HARDENED") {
        FAIL("Surviving packet payload mismatch");
        return;
    }

    PASS();
}

void test_pcap_ipv6_and_vlan() {
    TEST("PCAP Ingestion: 802.1Q VLAN and IPv6 packet parsing");

    std::string test_pcap = "/tmp/vlan_ipv6_test.pcap";
    pcap_t* dead = pcap_open_dead(DLT_EN10MB, 65535);
    if (!dead) {
        FAIL("Could not open dead pcap handle");
        return;
    }

    pcap_dumper_t* dumper = pcap_dump_open(dead, test_pcap.c_str());
    if (!dumper) {
        pcap_close(dead);
        FAIL("Could not open pcap dump file");
        return;
    }

    struct pcap_pkthdr h;
    memset(&h, 0, sizeof(h));

    // 1. Ethernet with 802.1Q VLAN tag (EtherType 0x8100, tag 4 bytes) + IPv4 UDP
    // Ethernet: 12 bytes MACs + 0x8100 (2 bytes) + VLAN ID 100 (2 bytes) + 0x0800 IPv4 (2 bytes)
    std::vector<uint8_t> vlan_pkt = {
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, // dst MAC
        0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, // src MAC
        0x81, 0x00,                         // 802.1Q EtherType
        0x00, 0x64,                         // VLAN ID 100
        0x45, 0x00, 0x00, 0x24,             // IPv4, 36 bytes total
        0x00, 0x01, 0x00, 0x00,
        0x40, 17,   0x00, 0x00,             // UDP
        192, 168, 1, 10,
        192, 168, 1, 20,
        0x04, 0xD2, 0x04, 0xD2,             // port 1234 -> 1234
        0x00, 0x10, 0x00, 0x00,             // UDP len = 16 (8 hdr + 8 payload)
        'V', 'L', 'A', 'N', 'T', 'E', 'S', 'T'
    };
    h.caplen = vlan_pkt.size();
    h.len = vlan_pkt.size();
    pcap_dump((u_char*)dumper, &h, vlan_pkt.data());

    // 2. Ethernet with IPv6 TCP (EtherType 0x86DD)
    std::vector<uint8_t> ipv6_pkt = {
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, // dst MAC
        0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, // src MAC
        0x86, 0xDD,                         // IPv6 EtherType (or direct offset 14)
        // IPv6 Header (40 bytes)
        0x60, 0x00, 0x00, 0x00,             // Version 6
        0x00, 28,                           // Payload len = 28 (20 TCP + 8 payload)
        6,                                  // Next header = TCP (6)
        64,                                 // Hop limit
        // Source IPv6 (2001:db8::1)
        0x20, 0x01, 0x0d, 0xb8, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
        // Dst IPv6 (2001:db8::2)
        0x20, 0x01, 0x0d, 0xb8, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02,
        // TCP Header (20 bytes)
        0x1F, 0x90, 0x00, 80,               // sport 8080, dport 80
        0x00, 0x00, 0x00, 0x01,             // seq
        0x00, 0x00, 0x00, 0x00,             // ack
        0x50, 0x18, 0xFF, 0xFF,             // offset 5 (20 bytes)
        0x00, 0x00, 0x00, 0x00,
        // Payload (8 bytes)
        'I', 'P', 'V', '6', 'T', 'E', 'S', 'T'
    };
    h.caplen = ipv6_pkt.size();
    h.len = ipv6_pkt.size();
    pcap_dump((u_char*)dumper, &h, ipv6_pkt.data());

    pcap_dump_close(dumper);
    pcap_close(dead);

    auto pkts = PcapReader::read_pcap(test_pcap);
    if (pkts.size() != 2) {
        FAIL("Expected 2 packets (1 VLAN, 1 IPv6), got " << pkts.size());
        return;
    }

    std::string vlan_pay(pkts[0].payload.begin(), pkts[0].payload.end());
    std::string ipv6_pay(pkts[1].payload.begin(), pkts[1].payload.end());

    if (vlan_pay != "VLANTEST") {
        FAIL("VLAN payload mismatch: got " << vlan_pay);
        return;
    }
    if (ipv6_pay != "IPV6TEST" || pkts[1].protocol != 6 || pkts[1].dst_port != 80) {
        FAIL("IPv6 payload/metadata mismatch: got " << ipv6_pay);
        return;
    }

    PASS();
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "========================================\n"
              << " GPU Sparse IDS — Test Suite\n"
              << "========================================\n\n";

    // Unit tests — features
    test_ngram_empty_payload();
    test_ngram_short_payload();
    test_ngram_exact_ngram_size();
    test_ngram_deterministic();
    test_ngram_unique_and_sorted();
    test_ngram_identical_for_same_content();

    // Unit tests — CSR
    test_csr_build();
    test_csr_empty_matrix();
    test_csr_row_ptr_monotonic();

    // Unit tests — SpMM
    test_spmm_known_products();
    test_spmm_vs_dense_reference();

    // Unit tests — candidates & verification
    test_candidate_threshold();
    test_short_sig_bypass();
    test_exact_verification();
    test_verification_rejects_non_match();
    test_verification_protocol_filter();
    test_verification_port_filter();

    // Property tests
    test_no_false_negatives_property();
    test_short_sig_e2e_detection();
    test_hash_collision_effect();

    // Hardened ingestion tests
    test_pcap_hardening_and_malformed_packets();
    test_pcap_ipv6_and_vlan();

    // Integration tests
    test_e2e_sequential();
    test_e2e_openmp_matches_sequential();

    std::cout << "\n========================================\n"
              << "Results: " << tests_passed << " passed, "
              << tests_failed << " failed\n"
              << "========================================\n";

    return tests_failed > 0 ? 1 : 0;
}
