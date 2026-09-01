#include "sampler.h"

#include <stdexcept>

namespace minillm {

std::int64_t GreedySampler::Argmax(const std::vector<float>& logits) const
{
    if (logits.empty()) {
        throw std::runtime_error("Cannot sample from empty logits");
    }

    std::int64_t best_index = 0;
    float best_value = logits[0];
    for (std::int64_t i = 1; i < static_cast<std::int64_t>(logits.size()); ++i) {
        if (logits[static_cast<std::size_t>(i)] > best_value) {
            best_value = logits[static_cast<std::size_t>(i)];
            best_index = i;
        }
    }
    return best_index;
}

} // namespace minillm
