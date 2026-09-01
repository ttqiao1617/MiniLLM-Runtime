#include "batch_plan.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void Expect(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

template <typename Callable>
void ExpectThrows(Callable&& callable, const std::string& message)
{
    try {
        callable();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

class TestCache final : public minillm::KvCacheHandle {
public:
    explicit TestCache(std::int64_t capacity)
    {
        capacity_ = capacity;
        allocated_bytes_ = static_cast<std::size_t>(capacity);
    }
};

minillm::RequestState MakeActive(const std::string& id, int past, int slot)
{
    // Two elements avoid a GCC 13 -Warray-bounds false positive in the
    // inlined std::vector one-element copy used by RequestState.
    std::vector<std::int64_t> prompt(2, 1);
    minillm::RequestState request{id, std::move(prompt), 16};
    request.generated_tokens.assign(static_cast<std::size_t>(past + 1), 2);
    request.status = minillm::RequestStatus::Active;
    request.kv_cache = std::make_unique<TestCache>(64);
    request.batch_slot = slot;
    return request;
}

} // namespace

int main()
{
    try {
        Expect(minillm::ComputeSerialRoundBudget(0, 8) == 0, "empty serial round");
        Expect(minillm::ComputeSerialRoundBudget(4, 8) == 4,
            "partial serial cohort must not wrap");
        Expect(minillm::ComputeSerialRoundBudget(8, 4) == 4,
            "serial token cap");
        ExpectThrows([]() { (void) minillm::ComputeSerialRoundBudget(4, 0); },
            "invalid serial token cap");

        auto a = MakeActive("a", 8, 0);
        auto b = MakeActive("b", 32, 3);
        auto c = MakeActive("c", 16, 1);
        const auto plan = minillm::BuildBatchPlan({&a, &b, &c}, 8);
        Expect(plan.BatchSize() == 3, "batch size");
        Expect(plan.max_past_length == 32, "max past length");
        Expect(plan.padding_tokens == 40, "padding token accounting");
        Expect(plan.slot_ids == std::vector<int>({0, 3, 1}), "slot mapping");
        ExpectThrows([&]() { (void) minillm::BuildBatchPlan({}, 8); }, "empty batch");
        ExpectThrows([&]() { (void) minillm::BuildBatchPlan({&a, &b}, 1); }, "batch cap");
        b.batch_slot = 0;
        ExpectThrows([&]() { (void) minillm::BuildBatchPlan({&a, &b}, 8); },
            "duplicate slot");
        b.batch_slot = 3;
        ExpectThrows([&]() { (void) minillm::BuildBatchPlan({&a, &a}, 8); },
            "duplicate request/cache");
        c.status = minillm::RequestStatus::Finished;
        ExpectThrows([&]() { (void) minillm::BuildBatchPlan({&c}, 8); },
            "finished request");
        std::cout << "batch_plan_test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "batch_plan_test failed: " << error.what() << '\n';
        return 1;
    }
}
