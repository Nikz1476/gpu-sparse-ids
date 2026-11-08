#pragma once
#include "common/types.h"
#include <string>
#include <vector>

namespace ids {

class PcapReader {
public:
    static std::vector<Packet> read_pcap(const std::string& filepath);
    static bool write_pcap(const std::string& filepath, const std::vector<Packet>& packets);
    static size_t count_packets(const std::string& filepath);
};

} // namespace ids

