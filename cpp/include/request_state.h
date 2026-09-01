#pragma once

#include "generation_backend.h"
#include "profiler.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace minillm {

enum class RequestStatus { Waiting, Active, Finished, Failed };
enum class FinishReason { None, Length, Eos, Error };

inline const char* ToString(FinishReason reason)
{
    switch (reason) {
    case FinishReason::None: return "none";
    case FinishReason::Length: return "length";
    case FinishReason::Eos: return "eos";
    case FinishReason::Error: return "error";
    }
    return "unknown";
}

struct RequestState {
    RequestState(std::string id, std::vector<std::int64_t> prompt, int max_tokens)
        : request_id(std::move(id)), prompt_tokens(std::move(prompt)),
          generated_tokens(prompt_tokens), max_new_tokens(max_tokens) {}

    std::string request_id;
    std::vector<std::int64_t> prompt_tokens;
    std::vector<std::int64_t> generated_tokens;
    int max_new_tokens{0};
    RequestStatus status{RequestStatus::Waiting};
    FinishReason finish_reason{FinishReason::None};
    std::unique_ptr<KvCacheHandle> kv_cache;
    std::vector<StepMetric> metrics;
    double queue_wait_ms{0.0};
    double kv_init_ms{0.0};
    double completed_ms{0.0};
    std::size_t kv_allocated_bytes{0};
    int cache_capacity_tokens{0};
    std::vector<int> block_table;
    int block_committed_tokens{0};
    int block_reserved_tokens{0};
    int block_rounding_waste_tokens{0};
    double block_alloc_ms{0.0};
    double block_release_ms{0.0};
    int arrival_iteration{0};
    double arrival_ms{0.0};
    int batch_slot{-1};

    int GeneratedCount() const
    {
        return static_cast<int>(generated_tokens.size() - prompt_tokens.size());
    }
    bool IsFinished() const { return status == RequestStatus::Finished; }
};

} // namespace minillm
