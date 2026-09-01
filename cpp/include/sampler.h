#pragma once

#include <cstdint>
#include <vector>

namespace minillm {

class GreedySampler {
public:
    std::int64_t Argmax(const std::vector<float>& logits) const;
};

} // namespace minillm
