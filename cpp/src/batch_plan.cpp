#include "batch_plan.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace minillm {

int ComputeSerialRoundBudget(std::size_t active_requests, int max_tokens_per_step)
{
    if (max_tokens_per_step <= 0) {
        throw std::runtime_error("serial round token budget must be positive");
    }
    return static_cast<int>(std::min(
        active_requests, static_cast<std::size_t>(max_tokens_per_step)));
}

BatchPlan BuildBatchPlan(const std::vector<RequestState*>& requests, int max_batch_size)
{
    if (requests.empty()) throw std::runtime_error("cannot build an empty decode batch");
    if (max_batch_size <= 0 || static_cast<int>(requests.size()) > max_batch_size) {
        throw std::runtime_error("decode batch exceeds configured maximum");
    }

    BatchPlan plan;
    plan.requests = requests;
    std::unordered_set<const RequestState*> unique_requests;
    std::unordered_set<const KvCacheHandle*> unique_caches;
    std::unordered_set<int> unique_slots;
    std::vector<std::int64_t> past_lengths;
    past_lengths.reserve(requests.size());
    for (auto* request : requests) {
        if (request == nullptr || request->status != RequestStatus::Active
            || request->kv_cache == nullptr) {
            throw std::runtime_error("decode batch contains an invalid request");
        }
        if (!unique_requests.insert(request).second
            || !unique_caches.insert(request->kv_cache.get()).second) {
            throw std::runtime_error("decode batch contains a duplicate request/cache");
        }
        if (request->batch_slot < 0 || !unique_slots.insert(request->batch_slot).second) {
            throw std::runtime_error("decode batch contains an invalid or duplicate slot");
        }
        const auto past = static_cast<std::int64_t>(request->generated_tokens.size()) - 1;
        if (past <= 0 || past >= request->kv_cache->Capacity()) {
            throw std::runtime_error("decode batch request has an invalid past length");
        }
        past_lengths.push_back(past);
        plan.slot_ids.push_back(request->batch_slot);
        plan.max_past_length = std::max(plan.max_past_length, past);
    }
    for (const auto past : past_lengths) {
        plan.padding_tokens += static_cast<std::size_t>(plan.max_past_length - past);
    }
    return plan;
}

} // namespace minillm
