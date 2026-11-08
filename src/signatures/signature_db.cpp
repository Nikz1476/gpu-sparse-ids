#include "signatures/signature_db.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

namespace ids {

static std::vector<uint8_t> hex_to_bytes(const std::string& hex) {
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i + 1 < hex.size(); i += 2) {
        bytes.push_back(static_cast<uint8_t>(std::stoi(hex.substr(i, 2), nullptr, 16)));
    }
    return bytes;
}

static std::string bytes_to_hex(const std::vector<uint8_t>& bytes) {
    std::ostringstream oss;
    for (uint8_t b : bytes)
        oss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(b);
    return oss.str();
}

std::vector<Signature> SignatureDB::load_signatures(const std::string& filepath) {
    std::vector<Signature> sigs;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error opening signature file: " << filepath << std::endl;
        return sigs;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(iss, token, '|'))
            tokens.push_back(token);

        if (tokens.size() < 5) continue;

        Signature sig;
        sig.id = std::stoul(tokens[0]);
        sig.name = tokens[1];
        sig.protocol = static_cast<uint8_t>(std::stoi(tokens[2]));
        sig.dst_port = static_cast<uint16_t>(std::stoi(tokens[3]));
        sig.pattern = hex_to_bytes(tokens[4]);
        sig.description = (tokens.size() > 5) ? tokens[5] : "";
        sig.num_features = 0;
        sigs.push_back(std::move(sig));
    }
    return sigs;
}

bool SignatureDB::save_signatures(const std::string& filepath, const std::vector<Signature>& sigs) {
    std::ofstream file(filepath);
    if (!file.is_open()) return false;

    file << "# GPU Sparse IDS Signature Database\n";
    file << "# Format: ID|NAME|PROTOCOL|PORT|HEX_PATTERN|DESCRIPTION\n";
    file << "# PROTOCOL: 0=any, 6=TCP, 17=UDP  PORT: 0=any\n\n";

    for (const auto& sig : sigs) {
        file << sig.id << "|" << sig.name << "|"
             << static_cast<int>(sig.protocol) << "|" << sig.dst_port << "|"
             << bytes_to_hex(sig.pattern) << "|" << sig.description << "\n";
    }
    return true;
}

std::vector<Signature> SignatureDB::create_test_signatures() {
    std::vector<Signature> sigs;

    auto make_sig = [](uint32_t id, const std::string& name, const std::string& desc,
                       const std::string& pattern_str, uint8_t proto, uint16_t port) {
        Signature sig;
        sig.id = id;
        sig.name = name;
        sig.description = desc;
        sig.pattern.assign(pattern_str.begin(), pattern_str.end());
        sig.protocol = proto;
        sig.dst_port = port;
        sig.num_features = 0;
        return sig;
    };

    auto make_sig_bytes = [](uint32_t id, const std::string& name, const std::string& desc,
                             const std::vector<uint8_t>& pattern, uint8_t proto, uint16_t port) {
        Signature sig;
        sig.id = id;
        sig.name = name;
        sig.description = desc;
        sig.pattern = pattern;
        sig.protocol = proto;
        sig.dst_port = port;
        sig.num_features = 0;
        return sig;
    };

    sigs.push_back(make_sig(1001, "SQL_INJECTION_OR",
        "SQL injection OR 1=1", "' OR 1=1--", 6, 0));
    sigs.push_back(make_sig(1002, "XSS_SCRIPT_ALERT",
        "XSS script alert", "<script>alert(", 6, 0));
    sigs.push_back(make_sig(1003, "CMD_INJECTION_PASSWD",
        "Command injection /etc/passwd", "; cat /etc/passwd", 6, 0));
    sigs.push_back(make_sig(1004, "DIR_TRAVERSAL_SHADOW",
        "Directory traversal /etc/shadow", "../../etc/shadow", 0, 0));
    sigs.push_back(make_sig_bytes(1005, "NOP_SLED",
        "NOP sled indicator", {0x90, 0x90, 0x90, 0x90, 0x90}, 0, 0));
    sigs.push_back(make_sig(1006, "MALICIOUS_UA",
        "Nikto scanner user agent", "User-Agent: Nikto", 6, 80));
    sigs.push_back(make_sig(1007, "DNS_TUNNEL",
        "DNS tunneling indicator", "AAAA", 17, 53));
    sigs.push_back(make_sig(1008, "PHP_SHELL_EVAL",
        "PHP eval base64 decode shell", "eval(base64_decode(", 6, 0));

    return sigs;
}

} // namespace ids

