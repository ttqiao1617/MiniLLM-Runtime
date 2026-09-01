#include "runtime_session.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
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

class MockCache final : public minillm::KvCacheHandle {
public:
    MockCache(std::int64_t capacity, std::size_t bytes)
    {
        capacity_ = capacity;
        allocated_bytes_ = bytes;
    }
};

class MockGenerationBackend final : public minillm::GenerationBackend {
public:
    MockGenerationBackend()
    {
        spec_ = minillm::ModelSpec::Gpt2();
        spec_.vocab_size = 8;
        spec_.eos_token_ids = {7};
        spec_.pad_token_id = 0;
        spec_.engine_max_sequence_length = 8;
        spec_.Validate();
    }

    void SetGenerationSpec(const minillm::ModelSpec& spec) override { spec_ = spec; }
    const minillm::ModelSpec& GenerationSpec() const override { return spec_; }
    void LoadGenerationEngines(const std::string&, const std::string&) override {}
    minillm::GenerationCapabilities Capabilities() const override
    {
        return {8, batch_layout, true};
    }
    void SetSamplingMode(minillm::SamplingMode mode) override { sampling_mode = mode; }

    void PrepareFixedShapeBuffers(std::int64_t max_seq_len) override
    {
        prepared_lengths.push_back(max_seq_len);
    }

    minillm::ForwardOutputs ForwardFixedShape(
        const minillm::ForwardInputs& inputs, std::int64_t max_seq_len) override
    {
        Expect(static_cast<std::int64_t>(inputs.input_ids.size()) == max_seq_len,
            "prefill must use the exact prompt length");
        ++prefill_calls;
        return Outputs(3);
    }

    std::unique_ptr<minillm::KvCacheHandle> CreateKvCache(
        std::int64_t prompt_len, std::int64_t max_seq_len) override
    {
        ++cache_creations;
        last_prompt_len = prompt_len;
        last_capacity = max_seq_len;
        return std::make_unique<MockCache>(max_seq_len, static_cast<std::size_t>(max_seq_len) * 100);
    }

    minillm::ForwardOutputs ForwardKvToken(
        minillm::KvCacheHandle& cache, std::int64_t token_id, std::int64_t past_len) override
    {
        Expect(cache.Capacity() == last_capacity, "session passed the wrong cache handle");
        decode_tokens.push_back(token_id);
        decode_past_lengths.push_back(past_len);
        return Outputs(past_len == last_prompt_len ? 4 : 7);
    }

    std::uint64_t DecodeBatchExecutionCount() const override { return enqueue_count; }

    minillm::ForwardOutputs ForwardKvBatch(
        const std::vector<minillm::KvBatchInput>& inputs) override
    {
        ++batch_calls;
        ++enqueue_count;
        minillm::ForwardOutputs output;
        if (sampling_mode == minillm::SamplingMode::HostLogits) {
            output.logits_shape = {
                static_cast<std::int64_t>(inputs.size()), 1, spec_.vocab_size};
            output.logits.assign(
                inputs.size() * static_cast<std::size_t>(spec_.vocab_size), -10.0F);
        }
        std::int64_t max_past = 0;
        std::size_t padding = 0;
        for (const auto& input : inputs) max_past = std::max(max_past, input.past_length);
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            const auto token = inputs[index].past_length == 3 ? 7 : 4;
            if (sampling_mode == minillm::SamplingMode::DeviceArgmax) {
                output.sampled_token_ids.push_back(token);
            } else {
                output.logits[index * static_cast<std::size_t>(spec_.vocab_size)
                    + static_cast<std::size_t>(token)] = 10.0F;
            }
            padding += static_cast<std::size_t>(max_past - inputs[index].past_length);
        }
        output.profile.batch_size = static_cast<int>(inputs.size());
        output.profile.max_past_length = max_past;
        output.profile.padding_tokens = batch_layout == minillm::DecodeBatchLayout::Padded
            ? padding : 0;
        output.profile.enqueue_count = 1;
        return output;
    }

    minillm::ModelSpec spec_;
    int prefill_calls{0};
    int cache_creations{0};
    std::int64_t last_prompt_len{0};
    std::int64_t last_capacity{0};
    std::vector<std::int64_t> prepared_lengths;
    std::vector<std::int64_t> decode_tokens;
    std::vector<std::int64_t> decode_past_lengths;
    int batch_calls{0};
    std::uint64_t enqueue_count{0};
    minillm::SamplingMode sampling_mode{minillm::SamplingMode::HostLogits};
    minillm::DecodeBatchLayout batch_layout{minillm::DecodeBatchLayout::Padded};

private:
    minillm::ForwardOutputs Outputs(std::int64_t token) const
    {
        minillm::ForwardOutputs output;
        if (sampling_mode == minillm::SamplingMode::DeviceArgmax) {
            output.sampled_token_ids = {token};
            return output;
        }
        output.logits_shape = {1, 1, spec_.vocab_size};
        output.logits.assign(static_cast<std::size_t>(spec_.vocab_size), -10.0F);
        output.logits[static_cast<std::size_t>(token)] = 10.0F;
        return output;
    }
};

} // namespace

int main()
{
    try {
        MockGenerationBackend backend;
        minillm::RuntimeSession session(backend);
        minillm::RequestState request{"request", {1, 2}, 5};
        session.Prefill(request);
        Expect(request.status == minillm::RequestStatus::Active, "prefill should activate request");
        Expect(request.generated_tokens == std::vector<std::int64_t>({1, 2, 3}),
            "prefill token mismatch");
        Expect(request.kv_cache && request.kv_cache->Capacity() == 7, "cache capacity mismatch");
        Expect(request.kv_allocated_bytes == 700, "cache byte accounting mismatch");
        while (!request.IsFinished()) session.DecodeOneToken(request);
        Expect(request.generated_tokens == std::vector<std::int64_t>({1, 2, 3, 4, 7}),
            "decode/EOS sequence mismatch");
        Expect(request.GeneratedCount() == 3, "EOS must stop before max_new_tokens");
        Expect(request.finish_reason == minillm::FinishReason::Eos, "EOS finish reason mismatch");
        Expect(backend.decode_tokens == std::vector<std::int64_t>({3, 4}),
            "decode input-token ownership mismatch");
        Expect(backend.decode_past_lengths == std::vector<std::int64_t>({2, 3}),
            "decode position/past growth mismatch");

        minillm::RequestState one_token{"one", {2}, 1};
        session.Prefill(one_token);
        Expect(one_token.IsFinished() && one_token.GeneratedCount() == 1,
            "max_new_tokens=1 must finish after prefill");
        Expect(one_token.finish_reason == minillm::FinishReason::Length,
            "length finish reason mismatch");

        minillm::RequestState too_long{"too_long", {1, 2, 3, 4, 5, 6}, 3};
        ExpectThrows([&]() { session.Prefill(too_long); }, "engine capacity guard");

        const int calls_before_warmup = backend.prefill_calls;
        session.Warmup({1, 2}, 5, 2);
        Expect(backend.prefill_calls == calls_before_warmup + 2,
            "warmup must execute every full request");
        Expect(backend.decode_past_lengths.size() == 6,
            "warmup must execute the complete decode path");

        MockGenerationBackend batch_backend;
        batch_backend.SetSamplingMode(minillm::SamplingMode::DeviceArgmax);
        minillm::RuntimeSession batch_session(batch_backend);
        minillm::RequestState batch_a{"batch_a", {1, 2}, 2};
        minillm::RequestState batch_b{"batch_b", {1, 2, 3}, 5};
        batch_session.Prefill(batch_a);
        batch_session.Prefill(batch_b);
        batch_a.batch_slot = 0;
        batch_b.batch_slot = 1;
        const auto plan = minillm::BuildBatchPlan({&batch_a, &batch_b}, 8);
        Expect(plan.BatchSize() == 2 && plan.max_past_length == 3
                && plan.padding_tokens == 1,
            "batch plan metadata mismatch");
        batch_session.DecodeBatch(plan);
        Expect(batch_backend.batch_calls == 1 && batch_backend.enqueue_count == 1,
            "batch session must issue one backend batch call/enqueue");
        Expect(batch_a.generated_tokens.back() == 4
                && batch_a.finish_reason == minillm::FinishReason::Length,
            "batch row 0 scatter/length mismatch");
        Expect(batch_b.generated_tokens.back() == 7
                && batch_b.finish_reason == minillm::FinishReason::Eos,
            "batch row 1 scatter/EOS mismatch");

        MockGenerationBackend ragged_backend;
        ragged_backend.batch_layout = minillm::DecodeBatchLayout::Ragged;
        minillm::RuntimeSession ragged_session(ragged_backend);
        minillm::RequestState ragged_a{"ragged_a", {1, 2}, 2};
        minillm::RequestState ragged_b{"ragged_b", {1, 2, 3}, 2};
        ragged_session.Prefill(ragged_a);
        ragged_session.Prefill(ragged_b);
        ragged_a.batch_slot = 0;
        ragged_b.batch_slot = 1;
        const auto ragged_plan = minillm::BuildBatchPlan({&ragged_a, &ragged_b}, 8);
        Expect(ragged_plan.padding_tokens == 1, "ragged test requires mixed logical lengths");
        ragged_session.DecodeBatch(ragged_plan);
        Expect(ragged_backend.batch_calls == 1,
            "ragged backend must execute without reporting logical padding as physical work");

        std::cout << "runtime_session_test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "runtime_session_test failed: " << error.what() << '\n';
        return 1;
    }
}
