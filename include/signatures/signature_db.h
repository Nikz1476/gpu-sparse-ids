#pragma once
#include "common/types.h"
#include <string>
#include <vector>

namespace ids {

class SignatureDB {
public:
    static std::vector<Signature> load_signatures(const std::string& filepath);
    static bool save_signatures(const std::string& filepath, const std::vector<Signature>& sigs);
    static std::vector<Signature> create_test_signatures();
};

} // namespace ids

