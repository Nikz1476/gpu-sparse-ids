#include "ingestion/pcap_reader.h"
#include <pcap/pcap.h>
#include <arpa/inet.h>
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <netinet/udp.h>
#include <net/ethernet.h>
#include <cstring>
#include <iostream>
#include <algorithm>

namespace ids {

std::vector<Packet> PcapReader::read_pcap(const std::string& filepath) {
    std::vector<Packet> packets;
    char errbuf[PCAP_ERRBUF_SIZE];

    pcap_t* handle = pcap_open_offline(filepath.c_str(), errbuf);
    if (!handle) {
        std::cerr << "Error opening PCAP: " << errbuf << std::endl;
        return packets;
    }

    int link_type = pcap_datalink(handle);
    int link_header_len = 14; // default ethernet
    if (link_type == DLT_EN10MB)        link_header_len = 14;
    else if (link_type == DLT_LINUX_SLL) link_header_len = 16;
    else if (link_type == DLT_RAW)       link_header_len = 0;

    struct pcap_pkthdr* header;
    const u_char* data;
    uint32_t pkt_id = 0;

    while (pcap_next_ex(handle, &header, &data) == 1) {
        // Handle 802.1Q VLAN encapsulation on Ethernet
        int actual_link_len = link_header_len;
        if (link_type == DLT_EN10MB && header->caplen >= 14) {
            uint16_t eth_type = ntohs(*reinterpret_cast<const uint16_t*>(data + 12));
            if (eth_type == 0x8100 && header->caplen >= 18) {
                actual_link_len = 18; // Skip 4-byte 802.1Q VLAN tag
            }
        }

        // Boundary check: at least link header + minimal IP header (IPv4 min 20 bytes)
        if (header->caplen < static_cast<uint32_t>(actual_link_len + 20)) continue;

        const u_char* ip_data = data + actual_link_len;
        uint8_t ip_version = (ip_data[0] >> 4) & 0x0F;

        std::string src_ip_str, dst_ip_str;
        uint8_t protocol = 0;
        int ip_hdr_len = 0;
        int ip_total_len = 0;

        if (ip_version == 4) {
            ip_hdr_len = (ip_data[0] & 0x0F) * 4;
            if (ip_hdr_len < 20) continue; // Malformed IPv4 header length
            if (header->caplen < static_cast<uint32_t>(actual_link_len + ip_hdr_len)) continue;

            ip_total_len = ntohs(*reinterpret_cast<const uint16_t*>(ip_data + 2));
            if (ip_total_len < ip_hdr_len) continue;

            protocol = ip_data[9];
            if (protocol != 6 && protocol != 17) continue; // TCP (6), UDP (17)

            char src_buf[INET_ADDRSTRLEN], dst_buf[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, ip_data + 12, src_buf, sizeof(src_buf));
            inet_ntop(AF_INET, ip_data + 16, dst_buf, sizeof(dst_buf));
            src_ip_str = src_buf;
            dst_ip_str = dst_buf;
        } else if (ip_version == 6) {
            // IPv6 header is fixed 40 bytes
            if (header->caplen < static_cast<uint32_t>(actual_link_len + 40)) continue;
            ip_hdr_len = 40;

            uint16_t payload_len_v6 = ntohs(*reinterpret_cast<const uint16_t*>(ip_data + 4));
            ip_total_len = 40 + payload_len_v6;
            protocol = ip_data[6];
            if (protocol != 6 && protocol != 17) continue;

            char src_buf[INET6_ADDRSTRLEN], dst_buf[INET6_ADDRSTRLEN];
            inet_ntop(AF_INET6, ip_data + 8, src_buf, sizeof(src_buf));
            inet_ntop(AF_INET6, ip_data + 24, dst_buf, sizeof(dst_buf));
            src_ip_str = src_buf;
            dst_ip_str = dst_buf;
        } else {
            continue; // Unsupported IP version
        }

        const u_char* transport_data = ip_data + ip_hdr_len;
        uint16_t src_port = 0, dst_port = 0;
        int payload_offset = 0;

        if (protocol == 6) { // TCP
            if (static_cast<int>(ip_hdr_len) + 20 > ip_total_len) continue;
            if (header->caplen < static_cast<uint32_t>(actual_link_len + ip_hdr_len + 20)) continue;

            src_port = ntohs(*reinterpret_cast<const uint16_t*>(transport_data));
            dst_port = ntohs(*reinterpret_cast<const uint16_t*>(transport_data + 2));
            uint8_t tcp_hdr_len = ((transport_data[12] >> 4) & 0x0F) * 4;
            if (tcp_hdr_len < 20) continue;
            if (static_cast<int>(ip_hdr_len) + tcp_hdr_len > ip_total_len) continue;

            payload_offset = ip_hdr_len + tcp_hdr_len;
        } else { // UDP
            if (static_cast<int>(ip_hdr_len) + 8 > ip_total_len) continue;
            if (header->caplen < static_cast<uint32_t>(actual_link_len + ip_hdr_len + 8)) continue;

            src_port = ntohs(*reinterpret_cast<const uint16_t*>(transport_data));
            dst_port = ntohs(*reinterpret_cast<const uint16_t*>(transport_data + 2));
            uint16_t udp_len = ntohs(*reinterpret_cast<const uint16_t*>(transport_data + 4));
            if (udp_len < 8) continue;

            payload_offset = ip_hdr_len + 8;
        }

        int payload_len = ip_total_len - payload_offset;
        if (payload_len < 0) continue;

        Packet pkt;
        pkt.id = pkt_id++;
        pkt.timestamp = header->ts.tv_sec + header->ts.tv_usec / 1e6;
        pkt.src_ip = std::move(src_ip_str);
        pkt.dst_ip = std::move(dst_ip_str);
        pkt.src_port = src_port;
        pkt.dst_port = dst_port;
        pkt.protocol = protocol;

        if (payload_len > 0) {
            int actual_start = actual_link_len + payload_offset;
            int available = static_cast<int>(header->caplen) - actual_start;
            if (available > 0) {
                int copy_len = std::min(payload_len, available);
                pkt.payload.assign(data + actual_start, data + actual_start + copy_len);
            }
        }

        packets.push_back(std::move(pkt));
    }

    pcap_close(handle);
    return packets;
}

bool PcapReader::write_pcap(const std::string& filepath, const std::vector<Packet>& packets) {
    pcap_t* dead = pcap_open_dead(DLT_RAW, 65535);
    if (!dead) return false;

    pcap_dumper_t* dumper = pcap_dump_open(dead, filepath.c_str());
    if (!dumper) {
        pcap_close(dead);
        return false;
    }

    for (const auto& pkt : packets) {
        std::vector<uint8_t> raw;

        int transport_hdr_len = (pkt.protocol == 6) ? 20 : 8;
        int total_len = 20 + transport_hdr_len + static_cast<int>(pkt.payload.size());

        // IP header (20 bytes)
        raw.resize(20, 0);
        raw[0] = 0x45;
        raw[2] = (total_len >> 8) & 0xFF;
        raw[3] = total_len & 0xFF;
        raw[6] = 0x40; // DF
        raw[8] = 64;   // TTL
        raw[9] = pkt.protocol;

        struct in_addr src_addr, dst_addr;
        inet_pton(AF_INET, pkt.src_ip.c_str(), &src_addr);
        inet_pton(AF_INET, pkt.dst_ip.c_str(), &dst_addr);
        memcpy(&raw[12], &src_addr, 4);
        memcpy(&raw[16], &dst_addr, 4);

        if (pkt.protocol == 6) {
            std::vector<uint8_t> tcp(20, 0);
            tcp[0] = (pkt.src_port >> 8) & 0xFF;
            tcp[1] = pkt.src_port & 0xFF;
            tcp[2] = (pkt.dst_port >> 8) & 0xFF;
            tcp[3] = pkt.dst_port & 0xFF;
            tcp[12] = 0x50; // offset = 5
            tcp[13] = 0x18; // PSH+ACK
            tcp[14] = 0xFF; tcp[15] = 0xFF;
            raw.insert(raw.end(), tcp.begin(), tcp.end());
        } else {
            std::vector<uint8_t> udp(8, 0);
            udp[0] = (pkt.src_port >> 8) & 0xFF;
            udp[1] = pkt.src_port & 0xFF;
            udp[2] = (pkt.dst_port >> 8) & 0xFF;
            udp[3] = pkt.dst_port & 0xFF;
            uint16_t udp_len = 8 + static_cast<uint16_t>(pkt.payload.size());
            udp[4] = (udp_len >> 8) & 0xFF;
            udp[5] = udp_len & 0xFF;
            raw.insert(raw.end(), udp.begin(), udp.end());
        }

        raw.insert(raw.end(), pkt.payload.begin(), pkt.payload.end());

        struct pcap_pkthdr hdr;
        hdr.ts.tv_sec = static_cast<time_t>(pkt.timestamp);
        hdr.ts.tv_usec = static_cast<suseconds_t>((pkt.timestamp - hdr.ts.tv_sec) * 1e6);
        hdr.caplen = raw.size();
        hdr.len = raw.size();

        pcap_dump(reinterpret_cast<u_char*>(dumper), &hdr, raw.data());
    }

    pcap_dump_close(dumper);
    pcap_close(dead);
    return true;
}

size_t PcapReader::count_packets(const std::string& filepath) {
    char errbuf[PCAP_ERRBUF_SIZE];
    pcap_t* handle = pcap_open_offline(filepath.c_str(), errbuf);
    if (!handle) return 0;

    size_t count = 0;
    struct pcap_pkthdr* header;
    const u_char* data;
    while (pcap_next_ex(handle, &header, &data) == 1) count++;
    pcap_close(handle);
    return count;
}

} // namespace ids

