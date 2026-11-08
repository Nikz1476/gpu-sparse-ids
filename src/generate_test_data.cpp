// Synthetic test data generator — creates PCAP files and signature files
// with known ground-truth matches for IDS correctness testing.

#include "ingestion/pcap_reader.h"
#include "signatures/signature_db.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <random>

using namespace ids;

static std::vector<uint8_t> str_to_bytes(const std::string& s) {
    return std::vector<uint8_t>(s.begin(), s.end());
}

static Packet make_tcp_packet(uint32_t id, const std::string& src, const std::string& dst,
                              uint16_t sport, uint16_t dport, const std::vector<uint8_t>& payload) {
    Packet p;
    p.id = id;
    p.timestamp = 1000000.0 + id;
    p.src_ip = src;
    p.dst_ip = dst;
    p.src_port = sport;
    p.dst_port = dport;
    p.protocol = 6;
    p.payload = payload;
    return p;
}

static Packet make_udp_packet(uint32_t id, const std::string& src, const std::string& dst,
                              uint16_t sport, uint16_t dport, const std::vector<uint8_t>& payload) {
    Packet p = make_tcp_packet(id, src, dst, sport, dport, payload);
    p.protocol = 17;
    return p;
}

int main(int argc, char* argv[]) {
    std::string out_dir = "data";
    int bench_count_arg = 10000;
    if (argc > 1) out_dir = argv[1];
    if (argc > 2) bench_count_arg = std::atoi(argv[2]);

    std::cout << "=== Generating synthetic test data ===" << std::endl;

    // --- Create signatures ---
    auto sigs = SignatureDB::create_test_signatures();
    std::string sig_file = out_dir + "/signatures/test_signatures.txt";
    SignatureDB::save_signatures(sig_file, sigs);
    std::cout << "Wrote " << sigs.size() << " signatures to " << sig_file << std::endl;

    // --- Create packets with known matches ---
    std::vector<Packet> packets;
    uint32_t pid = 0;

    // Pkt 0: Contains SQL injection "' OR 1=1--" -> should match sig 1001
    {
        std::string payload = "GET /login?user=admin' OR 1=1-- HTTP/1.1\r\nHost: target.com\r\n\r\n";
        packets.push_back(make_tcp_packet(pid++, "10.0.0.1", "192.168.1.100", 45678, 80,
                                          str_to_bytes(payload)));
    }

    // Pkt 1: Contains XSS "<script>alert(" -> should match sig 1002
    {
        std::string payload = "POST /comment HTTP/1.1\r\nHost: target.com\r\n\r\n"
                              "msg=<script>alert('XSS')</script>";
        packets.push_back(make_tcp_packet(pid++, "10.0.0.2", "192.168.1.100", 45679, 80,
                                          str_to_bytes(payload)));
    }

    // Pkt 2: Contains command injection "; cat /etc/passwd" -> match sig 1003
    {
        std::string payload = "GET /api?cmd=ls; cat /etc/passwd HTTP/1.1\r\n\r\n";
        packets.push_back(make_tcp_packet(pid++, "10.0.0.3", "192.168.1.100", 45680, 80,
                                          str_to_bytes(payload)));
    }

    // Pkt 3: Contains directory traversal "../../etc/shadow" -> match sig 1004
    {
        std::string payload = "GET /files/../../etc/shadow HTTP/1.1\r\n\r\n";
        packets.push_back(make_tcp_packet(pid++, "10.0.0.4", "192.168.1.100", 45681, 80,
                                          str_to_bytes(payload)));
    }

    // Pkt 4: NOP sled (binary) -> match sig 1005
    {
        std::vector<uint8_t> payload(100, 0x41); // 'A' padding
        // Insert NOP sled in the middle
        for (int i = 30; i < 35; ++i) payload[i] = 0x90;
        packets.push_back(make_tcp_packet(pid++, "10.0.0.5", "192.168.1.100", 45682, 8080,
                                          payload));
    }

    // Pkt 5: Malicious user-agent "User-Agent: Nikto" on port 80 -> match sig 1006
    {
        std::string payload = "GET / HTTP/1.1\r\nUser-Agent: Nikto/2.1.6\r\nHost: target.com\r\n\r\n";
        packets.push_back(make_tcp_packet(pid++, "10.0.0.6", "192.168.1.100", 45683, 80,
                                          str_to_bytes(payload)));
    }

    // Pkt 6: DNS tunneling "AAAA" on UDP port 53 -> match sig 1007
    {
        std::string payload = "AAAAexample.tunnel.com";
        packets.push_back(make_udp_packet(pid++, "10.0.0.7", "8.8.8.8", 12345, 53,
                                          str_to_bytes(payload)));
    }

    // Pkt 7: PHP shell "eval(base64_decode(" -> match sig 1008
    {
        std::string payload = "<?php eval(base64_decode('c3lzdGVtKCJscyIp')); ?>";
        packets.push_back(make_tcp_packet(pid++, "10.0.0.8", "192.168.1.100", 45684, 80,
                                          str_to_bytes(payload)));
    }

    // Pkt 8: Normal HTTP traffic (no match expected)
    {
        std::string payload = "GET /index.html HTTP/1.1\r\nHost: www.example.com\r\n"
                              "User-Agent: Mozilla/5.0\r\n\r\n";
        packets.push_back(make_tcp_packet(pid++, "10.0.0.10", "93.184.216.34", 45690, 80,
                                          str_to_bytes(payload)));
    }

    // Pkt 9: Empty payload (no match expected)
    {
        packets.push_back(make_tcp_packet(pid++, "10.0.0.11", "192.168.1.100", 45691, 443, {}));
    }

    // Pkt 10: Near-miss — similar to SQL injection but not exact match
    {
        std::string payload = "GET /login?user=admin' OR 2=2-- HTTP/1.1\r\n\r\n";
        packets.push_back(make_tcp_packet(pid++, "10.0.0.12", "192.168.1.100", 45692, 80,
                                          str_to_bytes(payload)));
    }

    // Pkt 11: MULTIPLE signatures in one packet (SQL injection + XSS)
    {
        std::string payload = "POST /vuln HTTP/1.1\r\n\r\n"
                              "q=' OR 1=1--&msg=<script>alert('XSS')</script>";
        packets.push_back(make_tcp_packet(pid++, "10.0.0.13", "192.168.1.100", 45693, 80,
                                          str_to_bytes(payload)));
    }

    // Pkt 12: Short payload (shorter than n-gram size) - no match expected
    {
        packets.push_back(make_tcp_packet(pid++, "10.0.0.14", "192.168.1.100", 45694, 80,
                                          str_to_bytes("AB")));
    }

    // Pkt 13: DNS query with AAAA on wrong port (no match for 1007 which requires port 53)
    {
        std::string payload = "AAAAsome.domain.com";
        packets.push_back(make_udp_packet(pid++, "10.0.0.15", "8.8.4.4", 12346, 5353,
                                          str_to_bytes(payload)));
    }

    // Pkt 14: Repeated signature content within payload
    {
        std::string payload = "eval(base64_decode('abc')) and eval(base64_decode('def'))";
        packets.push_back(make_tcp_packet(pid++, "10.0.0.16", "192.168.1.100", 45695, 80,
                                          str_to_bytes(payload)));
    }

    // Write PCAP
    std::string pcap_file = out_dir + "/pcap/test_traffic.pcap";
    if (PcapReader::write_pcap(pcap_file, packets)) {
        std::cout << "Wrote " << packets.size() << " packets to " << pcap_file << std::endl;
    } else {
        std::cerr << "ERROR writing PCAP file" << std::endl;
        return 1;
    }

    // Verify readback
    auto read_back = PcapReader::read_pcap(pcap_file);
    std::cout << "Read back " << read_back.size() << " packets from PCAP" << std::endl;

    // --- Generate a larger synthetic PCAP for benchmarking ---
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> payload_len_dist(50, 1500);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_int_distribution<int> sig_choice(0, static_cast<int>(sigs.size()) - 1);

    int bench_count = bench_count_arg;
    std::vector<Packet> bench_packets;
    int injected_count = 0;

    for (int i = 0; i < bench_count; ++i) {
        int plen = payload_len_dist(rng);
        std::vector<uint8_t> payload(plen);
        for (int j = 0; j < plen; ++j)
            payload[j] = static_cast<uint8_t>(byte_dist(rng));

        // Inject a signature into ~5% of packets
        if (rng() % 20 == 0 && !sigs.empty()) {
            int si = sig_choice(rng);
            const auto& pat = sigs[si].pattern;
            if (static_cast<int>(pat.size()) <= plen) {
                int pos = rng() % (plen - static_cast<int>(pat.size()) + 1);
                std::copy(pat.begin(), pat.end(), payload.begin() + pos);
                injected_count++;
            }
        }

        uint16_t dport = (rng() % 2 == 0) ? 80 : 443;
        uint8_t proto = (rng() % 5 == 0) ? 17 : 6;

        Packet pkt;
        pkt.id = i;
        pkt.timestamp = 2000000.0 + i * 0.001;
        pkt.src_ip = "10.0." + std::to_string((i / 256) % 256) + "." + std::to_string(i % 256);
        pkt.dst_ip = "192.168.1.1";
        pkt.src_port = 30000 + (i % 30000);
        pkt.dst_port = dport;
        pkt.protocol = proto;
        pkt.payload = std::move(payload);
        bench_packets.push_back(std::move(pkt));
    }

    std::string bench_file = out_dir + "/pcap/benchmark_traffic.pcap";
    PcapReader::write_pcap(bench_file, bench_packets);
    std::cout << "Wrote " << bench_count << " benchmark packets (" << injected_count
              << " with injected sigs) to " << bench_file << std::endl;

    // --- Write ground truth ---
    std::string gt_file = out_dir + "/expected_matches.txt";
    std::ofstream gt(gt_file);
    gt << "# Expected matches for test_traffic.pcap with test_signatures.txt\n";
    gt << "# Format: packet_id -> signature_id (signature_name)\n\n";
    gt << "0 -> 1001 (SQL_INJECTION_OR)\n";
    gt << "1 -> 1002 (XSS_SCRIPT_ALERT)\n";
    gt << "2 -> 1003 (CMD_INJECTION_PASSWD)\n";
    gt << "3 -> 1004 (DIR_TRAVERSAL_SHADOW)\n";
    gt << "4 -> 1005 (NOP_SLED)\n";
    gt << "5 -> 1006 (MALICIOUS_UA)\n";
    gt << "6 -> 1007 (DNS_TUNNEL)\n";
    gt << "7 -> 1008 (PHP_SHELL_EVAL)\n";
    gt << "8 -> NONE (normal traffic)\n";
    gt << "9 -> NONE (empty payload)\n";
    gt << "10 -> NONE (near-miss, ' OR 2=2-- not ' OR 1=1--)\n";
    gt << "11 -> 1001 (SQL_INJECTION_OR) + 1002 (XSS_SCRIPT_ALERT)\n";
    gt << "12 -> NONE (short payload)\n";
    gt << "13 -> NONE (AAAA on wrong port 5353, sig requires port 53)\n";
    gt << "14 -> 1008 (PHP_SHELL_EVAL) [repeated pattern, single alert]\n";
    gt.close();
    std::cout << "Wrote ground truth to " << gt_file << std::endl;

    std::cout << "\n=== Test data generation complete ===" << std::endl;
    return 0;
}

