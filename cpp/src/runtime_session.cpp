#include "runtime_session.h"

#include "tensor_utils.h"

#include <chrono>
#include <stdexcept>

namespace minillm {
namespace {

std::vector<float> LastTokenLogits(const ForwardOutputs& outputs)
{
    if (outputs.logits_shape.size() != 3 || outputs.logits_shape[0] != 1) {
        throw std::runtime_error("RuntimeSession expected logits [1, sequence, vocab]");
    }
    const auto seq = outputs.logits_shape[1];
    const auto vocab = outputs.logits_shape[2];
    const auto offset = static_cast<std::size_t>((seq - 1) * vocab);
    if (seq <= 0 || vocab <= 0 || offset + static_cast<std::size_t>(vocab) > outputs.logits.size()) {
        throw std::runtime_error("RuntimeSession received invalid logits buffer");
    }
    return {outputs.logits.begin() + static_cast<std::ptrdiff_t>(offset), outputs.logits.end()};
}

StepMetric MakeMetric(int step, double wall_ms, double post_ms, double sampler_ms,
                      std::int64_t token, const ForwardOutputs& outputs)
{
    StepMetric metric;
    metric.step = step;
    metric.latency_ms = wall_ms;
    metric.next_token = token;
    metric.step_total_ms = wall_ms + post_ms + sampler_ms;
    metric.backend_total_ms = outputs.profile.total_ms;
    metric.input_prepare_ms = outputs.profile.input_prepare_ms;
    metric.backend_shape_ms = outputs.profile.shape_ms;
    metric.backend_buffer_bind_ms = outputs.profile.buffer_bind_ms;
    metric.h2d_ms = outputs.profile.h2d_ms;
    metric.enqueue_ms = outputs.profile.enqueue_ms;
    metric.device_sample_ms = outputs.profile.device_sample_ms;
    metric.d2h_ms = outputs.profile.d2h_ms;
    metric.sync_ms = outputs.profile.sync_ms;
    metric.kv_gather_ms = outputs.profile.kv_gather_ms;
    metric.kv_scatter_ms = outputs.profile.kv_scatter_ms;
    metric.logits_postprocess_ms = post_ms;
    metric.sampler_ms = sampler_ms;
    metric.h2d_bytes = outputs.profile.h2d_bytes;
    metric.d2h_bytes = outputs.profile.d2h_bytes;
    metric.bound_output_bytes = outputs.profile.bound_output_bytes;
    metric.kv_gather_bytes = outputs.profile.kv_gather_bytes;
    metric.kv_scatter_bytes = outputs.profile.kv_scatter_bytes;
    metric.kv_staging_zero_bytes = outputs.profile.kv_staging_zero_bytes;
    metric.padding_tokens = outputs.profile.padding_tokens;
    metric.batch_scratch_bytes = outputs.profile.batch_scratch_bytes;
    metric.batch_scratch_allocated_bytes = outputs.profile.batch_scratch_allocated_bytes;
    metric.max_past_length = outputs.profile.max_past_length;
    metric.batch_size = outputs.profile.batch_size;
    metric.enqueue_count = outputs.profile.enqueue_count;
    metric.buffer_allocation_count = outputs.profile.buffer_allocation_count;
    metric.output_tensors = outputs.profile.output_tensors;
    metric.cuda_event_timing = outputs.profile.cuda_event_timing;
    return metric;
}

double ElapsedMs(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}

std::int64_t SelectSingleToken(const ForwardOutputs& outputs,
                               const GreedySampler& sampler,
                               int vocab_size,
                               double* post_ms,
                               double* sampler_ms)
{
    if (!outputs.sampled_token_ids.empty()) {
        if (outputs.sampled_token_ids.size() != 1) {
            throw std::runtime_error("single-row generation returned multiple sampled tokens");
        }
        const auto token = outputs.sampled_token_ids.front();
        if (token < 0 || token >= vocab_size) {
            throw std::runtime_error("backend sampled token is outside the vocabulary");
        }
        *post_ms = 0.0;
        *sampler_ms = 0.0;
        return token;
    }

    const auto post_start = std::chrono::steady_clock::now();
    auto logits = LastTokenLogits(outputs);
    *post_ms = ElapsedMs(post_start);
    const auto sampler_start = std::chrono::steady_clock::now();
    const auto token = sampler.Argmax(logits);
    *sampler_ms = ElapsedMs(sampler_start);
    return token;
}

} // namespace

RuntimeSession::RuntimeSession(GenerationBackend& backend) : backend_(backend) {}

void RuntimeSession::Warmup(
    const std::vector<std::int64_t>& prompt_tokens, int max_new_tokens, int runs)
{
    for (int run = 0; run < runs; ++run) {
        RequestState request{"warmup", prompt_tokens, max_new_tokens};
        Prefill(request);
        while (!request.IsFinished()) DecodeOneToken(request);
    }
}

void RuntimeSession::Prefill(RequestState& request)
{
    if (request.status != RequestStatus::Waiting || request.prompt_tokens.empty()
        || request.max_new_tokens <= 0) {
        throw std::runtime_error("invalid request state for prefill");
    }
    const auto required_capacity = static_cast<std::int64_t>(request.prompt_tokens.size()) + request.max_new_tokens;
    const auto capacity = request.cache_capacity_tokens > 0
        ? static_cast<std::int64_t>(request.cache_capacity_tokens) : required_capacity;
    if (capacity < required_capacity) throw std::runtime_error("request KV capacity is smaller than required length");
    if (capacity > backend_.GenerationSpec().engine_max_sequence_length) {
        throw std::runtime_error("request exceeds ModelSpec engine_max_sequence_length");
    }
    backend_.PrepareFixedShapeBuffers(static_cast<std::int64_t>(request.prompt_tokens.size()));
    const auto model_start = std::chrono::steady_clock::now();
    auto outputs = backend_.ForwardFixedShape(
        ForwardInputs{request.prompt_tokens, false}, static_cast<std::int64_t>(request.prompt_tokens.size()));
    const double model_ms = ElapsedMs(model_start);
    double post_ms = 0.0;
    double sampler_ms = 0.0;
    const auto token = SelectSingleToken(outputs, sampler_,
        backend_.GenerationSpec().vocab_size, &post_ms, &sampler_ms);
    request.generated_tokens = request.prompt_tokens;
    request.generated_tokens.push_back(token);
    request.metrics.push_back(MakeMetric(0, model_ms, post_ms, sampler_ms, token, outputs));

    const auto kv_start = std::chrono::steady_clock::now();
    request.kv_cache = backend_.CreateKvCache(
        static_cast<std::int64_t>(request.prompt_tokens.size()), capacity);
    request.kv_allocated_bytes = request.kv_cache->AllocatedBytes();
    request.kv_init_ms = ElapsedMs(kv_start);
    request.metrics.front().kv_init_ms = request.kv_init_ms;
    if (backend_.GenerationSpec().IsEos(token)) {
        request.status = RequestStatus::Finished;
        request.finish_reason = FinishReason::Eos;
    } else if (request.GeneratedCount() >= request.max_new_tokens) {
        request.status = RequestStatus::Finished;
        request.finish_reason = FinishReason::Length;
    } else {
        request.status = RequestStatus::Active;
    }
}

void RuntimeSession::DecodeOneToken(RequestState& request)
{
    if (request.status != RequestStatus::Active || request.kv_cache == nullptr) {
        throw std::runtime_error("invalid request state for decode");
    }
    const auto past_len = static_cast<std::int64_t>(request.generated_tokens.size() - 1);
    const auto model_start = std::chrono::steady_clock::now();
    auto outputs = backend_.ForwardKvToken(*request.kv_cache, request.generated_tokens.back(), past_len);
    const double model_ms = ElapsedMs(model_start);
    double post_ms = 0.0;
    double sampler_ms = 0.0;
    const auto token = SelectSingleToken(outputs, sampler_,
        backend_.GenerationSpec().vocab_size, &post_ms, &sampler_ms);
    const int step = request.GeneratedCount();
    request.generated_tokens.push_back(token);
    request.metrics.push_back(MakeMetric(step, model_ms, post_ms, sampler_ms, token, outputs));
    if (backend_.GenerationSpec().IsEos(token)) {
        request.status = RequestStatus::Finished;
        request.finish_reason = FinishReason::Eos;
    } else if (request.GeneratedCount() >= request.max_new_tokens) {
        request.status = RequestStatus::Finished;
        request.finish_reason = FinishReason::Length;
    }
}

void RuntimeSession::DecodeBatch(const BatchPlan& plan)
{
    const int batch_size = plan.BatchSize();
    const auto capabilities = backend_.Capabilities();
    if (batch_size <= 0 || batch_size > capabilities.max_decode_batch_size) {
        throw std::runtime_error("batch plan exceeds backend decode profile");
    }
    std::vector<KvBatchInput> inputs;
    inputs.reserve(plan.requests.size());
    for (auto* request : plan.requests) {
        if (request == nullptr || request->status != RequestStatus::Active
            || request->kv_cache == nullptr) {
            throw std::runtime_error("invalid request state for batched decode");
        }
        const auto past_len = static_cast<std::int64_t>(request->generated_tokens.size()) - 1;
        inputs.push_back({request->kv_cache.get(), request->generated_tokens.back(), past_len});
    }

    const auto execution_before = backend_.DecodeBatchExecutionCount();
    const auto model_start = std::chrono::steady_clock::now();
    auto outputs = backend_.ForwardKvBatch(inputs);
    const double model_ms = ElapsedMs(model_start);
    const auto execution_after = backend_.DecodeBatchExecutionCount();
    if (execution_after != execution_before + 1) {
        throw std::runtime_error("one DecodeBatch call must execute exactly one logical backend batch");
    }
    const bool sampled_on_device = !outputs.sampled_token_ids.empty();
    const auto vocab = static_cast<std::size_t>(backend_.GenerationSpec().vocab_size);
    if (sampled_on_device) {
        if (outputs.sampled_token_ids.size() != static_cast<std::size_t>(batch_size)) {
            throw std::runtime_error("batched decode returned the wrong sampled-token count");
        }
    } else {
        if (outputs.logits_shape.size() != 3 || outputs.logits_shape[0] != batch_size
            || outputs.logits_shape[1] != 1
            || outputs.logits_shape[2] != backend_.GenerationSpec().vocab_size) {
            throw std::runtime_error("batched decode returned an invalid logits shape");
        }
        if (outputs.logits.size() != static_cast<std::size_t>(batch_size) * vocab) {
            throw std::runtime_error("batched decode returned an invalid logits buffer");
        }
    }
    const auto expected_padding = capabilities.decode_batch_layout == DecodeBatchLayout::Padded
        ? plan.padding_tokens : 0;
    if (outputs.profile.batch_size != batch_size
        || outputs.profile.max_past_length != plan.max_past_length
        || outputs.profile.padding_tokens != expected_padding) {
        throw std::runtime_error("batched decode profile does not match BatchPlan");
    }

    for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
        auto& request = *plan.requests[static_cast<std::size_t>(batch_index)];
        double post_ms = 0.0;
        double sampler_ms = 0.0;
        std::int64_t token = -1;
        if (sampled_on_device) {
            token = outputs.sampled_token_ids[static_cast<std::size_t>(batch_index)];
            if (token < 0 || token >= static_cast<std::int64_t>(vocab)) {
                throw std::runtime_error("backend sampled token is outside the vocabulary");
            }
        } else {
            const auto post_start = std::chrono::steady_clock::now();
            const auto begin = outputs.logits.begin()
                + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(batch_index) * vocab);
            std::vector<float> logits(begin, begin + static_cast<std::ptrdiff_t>(vocab));
            post_ms = ElapsedMs(post_start);
            const auto sampler_start = std::chrono::steady_clock::now();
            token = sampler_.Argmax(logits);
            sampler_ms = ElapsedMs(sampler_start);
        }
        const int step = request.GeneratedCount();
        request.generated_tokens.push_back(token);
        request.metrics.push_back(MakeMetric(step, model_ms, post_ms, sampler_ms, token, outputs));
        if (backend_.GenerationSpec().IsEos(token)) {
            request.status = RequestStatus::Finished;
            request.finish_reason = FinishReason::Eos;
        } else if (request.GeneratedCount() >= request.max_new_tokens) {
            request.status = RequestStatus::Finished;
            request.finish_reason = FinishReason::Length;
        }
    }
}

} // namespace minillm
