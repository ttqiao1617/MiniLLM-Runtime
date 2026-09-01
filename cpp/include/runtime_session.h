#pragma once

#include "batch_plan.h"
#include "sampler.h"

namespace minillm {

class RuntimeSession {
public:
    explicit RuntimeSession(GenerationBackend& backend);
    void Warmup(const std::vector<std::int64_t>& prompt_tokens, int max_new_tokens, int runs);
    void Prefill(RequestState& request);
    void DecodeOneToken(RequestState& request);
    void DecodeBatch(const BatchPlan& plan);

private:
    GenerationBackend& backend_;
    GreedySampler sampler_;
};

} // namespace minillm
