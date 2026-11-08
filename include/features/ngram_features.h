#pragma once
#include "common/types.h"
#include <vector>
#include <cstdint>

namespace ids {

class NGramFeatureExtractor {
public:
    explicit NGramFeatureExtractor(int ngram_size = 3, int vocab_size = 65536);

    // Extract features from a byte sequence, returning sorted unique column indices
    std::vector<int> extract_features(const std::vector<uint8_t>& data) const;

    int ngram_size() const { return ngram_size_; }
    int vocab_size() const { return vocab_size_; }

    // Hash a single n-gram to a vocabulary index
    int hash_ngram(const uint8_t* data, int len) const;

private:
    int ngram_size_;
    int vocab_size_;
};

} // namespace ids

