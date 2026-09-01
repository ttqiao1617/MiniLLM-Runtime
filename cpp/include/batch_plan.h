#pragma once

#include "request_state.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace minillm {

struct BatchPlan {
    std::vector<RequestState*> requests;
    std::vector<int> slot_ids;
    std::int64_t max_past_length{0};
    std::size_t padding_tokens{0};

    int BatchSize() const { return static_cast<int>(requests.size()); }
};

BatchPlan BuildBatchPlan(const std::vector<RequestState*>& requests, int max_batch_size);

// A scheduler iteration is a round: every request that was active at the
// start of the round may advance at most once. max_tokens_per_step remains an
// upper bound, but must not make a partially filled serial cohort wrap around
// and decode the same request twice before the next arrival boundary.
int ComputeSerialRoundBudget(std::size_t active_requests, int max_tokens_per_step);

} // namespace minillm
