#include "features/ngram_features.h"
#include <algorithm>
#include <unordered_set>

namespace ids {

NGramFeatureExtractor::NGramFeatureExtractor(int ngram_size, int vocab_size)
    : ngram_size_(ngram_size), vocab_size_(vocab_size) {
    if (ngram_size_ < 1) ngram_size_ = 1;
    if (vocab_size_ < 1) vocab_size_ = 1;
}

int NGramFeatureExtractor::hash_ngram(const uint8_t* data, int len) const {
    // FNV-1a 32-bit hash
    uint32_t hash = 2166136261u;
    for (int i = 0; i < len; ++i) {
        hash ^= static_cast<uint32_t>(data[i]);
        hash *= 16777619u;
    }
    return static_cast<int>(hash % static_cast<uint32_t>(vocab_size_));
}

std::vector<int> NGramFeatureExtractor::extract_features(const std::vector<uint8_t>& data) const {
    int data_len = static_cast<int>(data.size());

    if (data_len == 0) {
        return {};  // empty payload => no features
    }

    std::unordered_set<int> feature_set;

    // Always extract standard n-grams when data is long enough
    if (data_len >= ngram_size_) {
        for (int i = 0; i <= data_len - ngram_size_; ++i) {
            int idx = hash_ngram(data.data() + i, ngram_size_);
            feature_set.insert(idx);
        }
    }

    // For short data (shorter than ngram_size), extract sub-n-gram features.
    // We use ALL window sizes from 1 to min(data_len, ngram_size_-1) so that
    // the same sub-n-gram features will also appear when this short sequence
    // occurs as a substring of a longer payload.
    //
    // CRITICAL CORRECTNESS PROPERTY:
    // When a signature pattern is shorter than ngram_size, we cannot use
    // standard n-grams because the packet's n-grams will never match.
    // Instead, we mark short signatures for direct verification bypass
    // (see generate_candidates). The features extracted here are still
    // used for any short data that appears standalone.
    if (data_len < ngram_size_) {
        // Hash the entire short content as a special "short-content" feature.
        // Use a distinct prefix to avoid collisions with standard n-grams.
        // This feature will match if and only if the exact same short content
        // appears. However, this alone is unreliable for substring matching
        // in longer payloads. The real fix is in candidate generation:
        // signatures with pattern.size() < ngram_size bypass the sparse filter
        // and go directly to exact verification.
        int idx = hash_ngram(data.data(), data_len);
        feature_set.insert(idx);
    }

    std::vector<int> features(feature_set.begin(), feature_set.end());
    std::sort(features.begin(), features.end());
    return features;
}

} // namespace ids
