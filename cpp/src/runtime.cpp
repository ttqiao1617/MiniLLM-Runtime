#include "runtime.h"

#include "execution_backend.h"
#include "profiler.h"
#include "kv_block_manager.h"
#include "model_spec.h"
#include "request_state.h"
#include "runtime_session.h"
#include "sampler.h"
#include "tensor_utils.h"
#include "trt_backend.h"

#include <algorithm>
#include <chrono>
#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <vector>

namespace minillm {
namespace {

std::vector<float> LastTokenLogits(const ForwardOutputs& outputs)
{
    if (outputs.logits_shape.size() != 3) {
        throw std::runtime_error("Expected logits shape [batch, sequence, vocab]");
    }
    if (outputs.logits_shape[0] != 1) {
        throw std::runtime_error("Only batch size 1 is supported");
    }

    const auto seq_len = outputs.logits_shape[1];
    const auto vocab_size = outputs.logits_shape[2];
    if (seq_len <= 0 || vocab_size <= 0) {
        throw std::runtime_error("Invalid logits shape");
    }

    const auto offset = static_cast<std::size_t>((seq_len - 1) * vocab_size);
    const auto end = offset + static_cast<std::size_t>(vocab_size);
    if (end > outputs.logits.size()) {
        throw std::runtime_error("Logits buffer is smaller than logits shape");
    }

    return std::vector<float>(outputs.logits.begin() + static_cast<std::ptrdiff_t>(offset),
        outputs.logits.begin() + static_cast<std::ptrdiff_t>(end));
}

double SumLatencyMs(const std::vector<StepMetric>& metrics)
{
    double total = 0.0;
    for (const auto& metric : metrics) {
        total += metric.step_total_ms;
    }
    return total;
}

std::vector<double> TpotLatencies(const std::vector<StepMetric>& metrics)
{
    std::vector<double> values;
    if (metrics.size() <= 1) {
        return values;
    }
    values.reserve(metrics.size() - 1);
    for (std::size_t i = 1; i < metrics.size(); ++i) {
        values.push_back(metrics[i].step_total_ms);
    }
    return values;
}

double SumValues(const std::vector<double>& values)
{
    double total = 0.0;
    for (const auto value : values) {
        total += value;
    }
    return total;
}

double Percentile(std::vector<double> values, double pct)
{
    if (values.empty()) {
        return 0.0;
    }

    std::sort(values.begin(), values.end());
    const double index = (static_cast<double>(values.size() - 1) * pct) / 100.0;
    const auto lower = static_cast<std::size_t>(index);
    const auto upper = std::min(lower + 1, values.size() - 1);
    if (lower == upper) {
        return values[lower];
    }

    const double weight = index - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

double MeanMetricMs(const std::vector<StepMetric>& metrics, double StepMetric::* field)
{
    if (metrics.empty()) {
        return 0.0;
    }
    double total = 0.0;
    for (const auto& metric : metrics) {
        total += metric.*field;
    }
    return total / static_cast<double>(metrics.size());
}

double MeanMetricBytes(const std::vector<StepMetric>& metrics, std::size_t StepMetric::* field)
{
    if (metrics.empty()) {
        return 0.0;
    }
    std::size_t total = 0;
    for (const auto& metric : metrics) {
        total += metric.*field;
    }
    return static_cast<double>(total) / static_cast<double>(metrics.size());
}

double MeanMetricInt(const std::vector<StepMetric>& metrics, int StepMetric::* field)
{
    if (metrics.empty()) {
        return 0.0;
    }
    int total = 0;
    for (const auto& metric : metrics) {
        total += metric.*field;
    }
    return static_cast<double>(total) / static_cast<double>(metrics.size());
}

StepMetric MakeStepMetric(int step,
                          double model_wall_ms,
                          double logits_postprocess_ms,
                          double sampler_ms,
                          std::int64_t next_token,
                          const ForwardOutputs& outputs)
{
    StepMetric metric;
    metric.step = step;
    metric.latency_ms = model_wall_ms;
    metric.next_token = next_token;
    metric.step_total_ms = model_wall_ms + logits_postprocess_ms + sampler_ms;
    metric.backend_total_ms = outputs.profile.total_ms;
    metric.input_prepare_ms = outputs.profile.input_prepare_ms;
    metric.backend_shape_ms = outputs.profile.shape_ms;
    metric.backend_buffer_bind_ms = outputs.profile.buffer_bind_ms;
    metric.h2d_ms = outputs.profile.h2d_ms;
    metric.enqueue_ms = outputs.profile.enqueue_ms;
    metric.device_sample_ms = outputs.profile.device_sample_ms;
    metric.d2h_ms = outputs.profile.d2h_ms;
    metric.sync_ms = outputs.profile.sync_ms;
    metric.logits_postprocess_ms = logits_postprocess_ms;
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

void PrintProfileSummary(const std::vector<StepMetric>& metrics)
{
    if (metrics.empty()) {
        return;
    }

    std::cout << "  avg step total ms: " << MeanMetricMs(metrics, &StepMetric::step_total_ms) << '\n'
              << "  avg backend total ms: " << MeanMetricMs(metrics, &StepMetric::backend_total_ms) << '\n'
              << "  avg input prepare ms: " << MeanMetricMs(metrics, &StepMetric::input_prepare_ms) << '\n'
              << "  avg shape ms: " << MeanMetricMs(metrics, &StepMetric::backend_shape_ms) << '\n'
              << "  avg buffer bind ms: " << MeanMetricMs(metrics, &StepMetric::backend_buffer_bind_ms) << '\n'
              << "  avg h2d ms: " << MeanMetricMs(metrics, &StepMetric::h2d_ms) << '\n'
              << "  avg enqueue ms: " << MeanMetricMs(metrics, &StepMetric::enqueue_ms) << '\n'
              << "  avg device sample ms: " << MeanMetricMs(metrics, &StepMetric::device_sample_ms) << '\n'
              << "  avg d2h ms: " << MeanMetricMs(metrics, &StepMetric::d2h_ms) << '\n'
              << "  avg sync ms: " << MeanMetricMs(metrics, &StepMetric::sync_ms) << '\n'
              << "  avg logits postprocess ms: " << MeanMetricMs(metrics, &StepMetric::logits_postprocess_ms) << '\n'
              << "  avg sampler ms: " << MeanMetricMs(metrics, &StepMetric::sampler_ms) << '\n'
              << "  avg h2d bytes: " << MeanMetricBytes(metrics, &StepMetric::h2d_bytes) << '\n'
              << "  avg d2h bytes: " << MeanMetricBytes(metrics, &StepMetric::d2h_bytes) << '\n'
              << "  avg output bytes bound: " << MeanMetricBytes(metrics, &StepMetric::bound_output_bytes) << '\n'
              << "  avg output tensors bound: " << MeanMetricInt(metrics, &StepMetric::output_tensors) << '\n';
}

std::vector<int> DefaultShapeBuckets()
{
    return {16, 32, 64, 128, 256, 512};
}

int SelectShapeBucket(const std::vector<int>& buckets, std::size_t seq_len)
{
    for (const auto bucket : buckets) {
        if (bucket >= static_cast<int>(seq_len)) {
            return bucket;
        }
    }
    throw std::runtime_error("sequence length exceeds largest shape bucket");
}

void ValidateShapeBuckets(const std::vector<int>& buckets)
{
    if (buckets.empty()) {
        throw std::runtime_error("shape buckets must not be empty");
    }
    int previous = 0;
    for (const auto bucket : buckets) {
        if (bucket <= 0) {
            throw std::runtime_error("shape buckets must be positive");
        }
        if (bucket <= previous) {
            throw std::runtime_error("shape buckets must be strictly increasing");
        }
        previous = bucket;
    }
}

std::string BucketsToString(const std::vector<int>& buckets)
{
    std::ostringstream out;
    for (std::size_t i = 0; i < buckets.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << buckets[i];
    }
    return out.str();
}

std::unique_ptr<ExecutionBackend> CreateConfiguredBackend(
    Provider provider, const std::string& model_spec_path)
{
    auto backend = CreateBackend(provider);
    if (!model_spec_path.empty()) backend->SetModelSpec(ModelSpec::Load(model_spec_path));
    return backend;
}

std::unique_ptr<GenerationBackend> CreateConfiguredGenerationBackend(
    Provider provider, const std::string& model_spec_path)
{
    auto backend = CreateGenerationBackend(provider);
    if (!model_spec_path.empty()) {
        backend->SetGenerationSpec(ModelSpec::Load(model_spec_path));
    }
    return backend;
}

} // namespace

Runtime::Runtime(RuntimeOptions options)
    : options_(std::move(options))
{
}

int Runtime::Run()
{
    std::cout << "MiniLLM-RT\n"
              << "  mode: " << ToString(options_.mode) << '\n'
              << "  provider: " << ToString(options_.provider) << '\n'
              << "  model spec: " << (options_.model_spec_path.empty()
                    ? "built-in GPT-2" : options_.model_spec_path) << '\n';

    switch (options_.mode) {
    case Mode::Forward:
        return RunForward();
    case Mode::Naive:
        return RunNaiveDecode();
    case Mode::FixedShape:
        return RunFixedShapeDecode();
    case Mode::BucketShape:
        return RunBucketShapeDecode();
    case Mode::KvCache:
        return RunKvCacheDecode();
    case Mode::Scheduler:
        return RunScheduler(false);
    case Mode::ContinuousBatch:
        return RunScheduler(true);
    }

    throw std::runtime_error("Unhandled runtime mode");
}

int Runtime::RunForward()
{
    const auto input_ids = LoadInputIds(options_.input_ids_path);
    auto backend = CreateConfiguredBackend(options_.provider, options_.model_spec_path);
    backend->Load(options_.model_path);
    backend->Forward(ForwardInputs{input_ids, true});
    std::cout << "forward completed through backend: " << backend->Name() << '\n';
    return 0;
}

int Runtime::RunNaiveDecode()
{
    const auto input_ids = LoadInputIds(options_.input_ids_path);
    if (input_ids.empty()) {
        throw std::runtime_error("input_ids is empty");
    }
    if (options_.max_new_tokens <= 0) {
        throw std::runtime_error("max_new_tokens must be positive");
    }
    if (options_.warmup_runs < 0) {
        throw std::runtime_error("warmup_runs must be non-negative");
    }

    auto backend = CreateConfiguredBackend(options_.provider, options_.model_spec_path);
    backend->Load(options_.model_path);
    GreedySampler sampler;
    for (int run = 0; run < options_.warmup_runs; ++run) {
        auto warm_tokens = input_ids;
        for (int step = 0; step < options_.max_new_tokens; ++step) {
            auto outputs = backend->Forward(ForwardInputs{warm_tokens, false});
            const auto next_token = sampler.Argmax(LastTokenLogits(outputs));
            warm_tokens.push_back(next_token);
            if (backend->Spec().IsEos(next_token)) break;
        }
    }

    std::vector<std::int64_t> generated = input_ids;
    std::vector<StepMetric> metrics;
    metrics.reserve(static_cast<std::size_t>(options_.max_new_tokens));

    std::cout << "naive greedy decode\n"
              << "  prompt tokens: " << input_ids.size() << '\n'
              << "  warmup runs: " << options_.warmup_runs << '\n'
              << "  max new tokens: " << options_.max_new_tokens << '\n';

    for (int step = 0; step < options_.max_new_tokens; ++step) {
        double model_wall_ms = 0.0;
        double logits_postprocess_ms = 0.0;
        double sampler_ms = 0.0;
        ForwardOutputs outputs;
        {
            ScopedTimer timer(&model_wall_ms);
            outputs = backend->Forward(ForwardInputs{generated, false});
        }

        std::vector<float> last_logits;
        {
            ScopedTimer timer(&logits_postprocess_ms);
            last_logits = LastTokenLogits(outputs);
        }

        std::int64_t next_token = 0;
        {
            ScopedTimer timer(&sampler_ms);
            next_token = sampler.Argmax(last_logits);
        }
        generated.push_back(next_token);
        metrics.push_back(MakeStepMetric(step, model_wall_ms, logits_postprocess_ms, sampler_ms, next_token, outputs));
        const auto& metric = metrics.back();

        std::cout << "step " << step
                  << " next_token=" << next_token
                  << " latency_ms=" << metric.latency_ms
                  << " enqueue_ms=" << metric.enqueue_ms
                  << " d2h_ms=" << metric.d2h_ms
                  << " sampler_ms=" << metric.sampler_ms
                  << " total_tokens=" << generated.size()
                  << '\n';
        if (backend->Spec().IsEos(next_token)) break;
    }

    WriteStepCsv(options_.output_csv_path, options_.model_path, ToString(options_.provider),
        ToString(options_.mode), static_cast<int>(input_ids.size()), options_.max_new_tokens, metrics);

    const auto total_latency_ms = SumLatencyMs(metrics);
    const auto end_to_end_tokens_per_s = total_latency_ms > 0.0
        ? static_cast<double>(metrics.size()) / (total_latency_ms / 1000.0)
        : 0.0;
    const auto tpot_values = TpotLatencies(metrics);
    const auto ttft_ms = metrics.empty() ? 0.0 : metrics.front().step_total_ms;
    const auto tpot_total_ms = SumValues(tpot_values);
    const auto mean_tpot_ms = !tpot_values.empty()
        ? tpot_total_ms / static_cast<double>(tpot_values.size())
        : 0.0;
    const auto decode_tokens_per_s = tpot_total_ms > 0.0
        ? static_cast<double>(tpot_values.size()) / (tpot_total_ms / 1000.0)
        : 0.0;

    std::cout << "naive decode completed\n"
              << "  generated tokens: " << metrics.size() << '\n'
              << "  total latency ms: " << total_latency_ms << '\n'
              << "  ttft ms: " << ttft_ms << '\n'
              << "  mean tpot ms: " << mean_tpot_ms << '\n'
              << "  p50 tpot ms: " << Percentile(tpot_values, 50.0) << '\n'
              << "  p95 tpot ms: " << Percentile(tpot_values, 95.0) << '\n'
              << "  decode tokens/s: " << decode_tokens_per_s << '\n'
              << "  end-to-end tokens/s: " << end_to_end_tokens_per_s << '\n';
    PrintProfileSummary(metrics);
    if (!options_.output_csv_path.empty()) {
        std::cout << "  csv: " << options_.output_csv_path << '\n';
    }

    return 0;
}

int Runtime::RunFixedShapeDecode()
{
    const auto input_ids = LoadInputIds(options_.input_ids_path);
    if (input_ids.empty()) {
        throw std::runtime_error("input_ids is empty");
    }
    if (options_.provider != Provider::TensorRt) {
        throw std::runtime_error("fixed_shape mode is implemented for --provider tensorrt only");
    }
    if (options_.max_new_tokens <= 0) {
        throw std::runtime_error("max_new_tokens must be positive");
    }
    if (options_.warmup_runs < 0) {
        throw std::runtime_error("warmup_runs must be non-negative");
    }

    const int required_seq_len = static_cast<int>(input_ids.size()) + options_.max_new_tokens;
    const int max_seq_len = options_.max_seq_len > 0 ? options_.max_seq_len : required_seq_len;
    if (max_seq_len < required_seq_len) {
        throw std::runtime_error("max_seq_len must be at least prompt_len + max_new_tokens");
    }

    auto backend = CreateConfiguredBackend(options_.provider, options_.model_spec_path);
    backend->Load(options_.model_path);
    auto* trt_backend = dynamic_cast<TrtBackend*>(backend.get());
    if (trt_backend == nullptr) {
        throw std::runtime_error("fixed_shape mode requires TensorRT backend");
    }
    trt_backend->PrepareFixedShapeBuffers(max_seq_len);
    for (int run = 0; run < options_.warmup_runs; ++run) {
        trt_backend->ForwardFixedShape(ForwardInputs{input_ids, false}, max_seq_len);
    }

    GreedySampler sampler;
    std::vector<std::int64_t> generated = input_ids;
    std::vector<StepMetric> metrics;
    metrics.reserve(static_cast<std::size_t>(options_.max_new_tokens));

    std::cout << "fixed-shape greedy decode\n"
              << "  prompt tokens: " << input_ids.size() << '\n'
              << "  warmup runs: " << options_.warmup_runs << '\n'
              << "  max new tokens: " << options_.max_new_tokens << '\n'
              << "  fixed max seq len: " << max_seq_len << '\n';

    for (int step = 0; step < options_.max_new_tokens; ++step) {
        double model_wall_ms = 0.0;
        double logits_postprocess_ms = 0.0;
        double sampler_ms = 0.0;
        ForwardOutputs outputs;
        {
            ScopedTimer timer(&model_wall_ms);
            outputs = trt_backend->ForwardFixedShape(ForwardInputs{generated, false}, max_seq_len);
        }

        std::vector<float> last_logits;
        {
            ScopedTimer timer(&logits_postprocess_ms);
            last_logits = LastTokenLogits(outputs);
        }

        std::int64_t next_token = 0;
        {
            ScopedTimer timer(&sampler_ms);
            next_token = sampler.Argmax(last_logits);
        }
        generated.push_back(next_token);
        metrics.push_back(MakeStepMetric(step, model_wall_ms, logits_postprocess_ms, sampler_ms, next_token, outputs));
        const auto& metric = metrics.back();

        std::cout << "step " << step
                  << " next_token=" << next_token
                  << " latency_ms=" << metric.latency_ms
                  << " enqueue_ms=" << metric.enqueue_ms
                  << " d2h_ms=" << metric.d2h_ms
                  << " sampler_ms=" << metric.sampler_ms
                  << " total_tokens=" << generated.size()
                  << '\n';
        if (backend->Spec().IsEos(next_token)) break;
    }

    WriteStepCsv(options_.output_csv_path, options_.model_path, ToString(options_.provider),
        ToString(options_.mode), static_cast<int>(input_ids.size()), options_.max_new_tokens, metrics);

    const auto total_latency_ms = SumLatencyMs(metrics);
    const auto end_to_end_tokens_per_s = total_latency_ms > 0.0
        ? static_cast<double>(metrics.size()) / (total_latency_ms / 1000.0)
        : 0.0;
    const auto tpot_values = TpotLatencies(metrics);
    const auto ttft_ms = metrics.empty() ? 0.0 : metrics.front().step_total_ms;
    const auto tpot_total_ms = SumValues(tpot_values);
    const auto mean_tpot_ms = !tpot_values.empty()
        ? tpot_total_ms / static_cast<double>(tpot_values.size())
        : 0.0;
    const auto decode_tokens_per_s = tpot_total_ms > 0.0
        ? static_cast<double>(tpot_values.size()) / (tpot_total_ms / 1000.0)
        : 0.0;

    std::cout << "fixed-shape decode completed\n"
              << "  generated tokens: " << metrics.size() << '\n'
              << "  total latency ms: " << total_latency_ms << '\n'
              << "  ttft ms: " << ttft_ms << '\n'
              << "  mean tpot ms: " << mean_tpot_ms << '\n'
              << "  p50 tpot ms: " << Percentile(tpot_values, 50.0) << '\n'
              << "  p95 tpot ms: " << Percentile(tpot_values, 95.0) << '\n'
              << "  decode tokens/s: " << decode_tokens_per_s << '\n'
              << "  end-to-end tokens/s: " << end_to_end_tokens_per_s << '\n';
    PrintProfileSummary(metrics);
    if (!options_.output_csv_path.empty()) {
        std::cout << "  csv: " << options_.output_csv_path << '\n';
    }

    return 0;
}

int Runtime::RunBucketShapeDecode()
{
    const auto input_ids = LoadInputIds(options_.input_ids_path);
    if (input_ids.empty()) {
        throw std::runtime_error("input_ids is empty");
    }
    if (options_.provider != Provider::TensorRt) {
        throw std::runtime_error("bucket_shape mode is implemented for --provider tensorrt only");
    }
    if (options_.max_new_tokens <= 0) {
        throw std::runtime_error("max_new_tokens must be positive");
    }
    if (options_.warmup_runs < 0) {
        throw std::runtime_error("warmup_runs must be non-negative");
    }

    auto buckets = options_.shape_buckets.empty() ? DefaultShapeBuckets() : options_.shape_buckets;
    ValidateShapeBuckets(buckets);
    const auto final_seq_len = input_ids.size() + static_cast<std::size_t>(options_.max_new_tokens);
    SelectShapeBucket(buckets, final_seq_len);

    auto backend = CreateConfiguredBackend(options_.provider, options_.model_spec_path);
    backend->Load(options_.model_path);
    auto* trt_backend = dynamic_cast<TrtBackend*>(backend.get());
    if (trt_backend == nullptr) {
        throw std::runtime_error("bucket_shape mode requires TensorRT backend");
    }
    const int max_required_bucket = SelectShapeBucket(buckets, final_seq_len);
    trt_backend->PrepareFixedShapeBuffers(max_required_bucket);
    const int initial_bucket = SelectShapeBucket(buckets, input_ids.size());
    for (int run = 0; run < options_.warmup_runs; ++run) {
        trt_backend->ForwardFixedShape(ForwardInputs{input_ids, false}, initial_bucket);
    }

    GreedySampler sampler;
    std::vector<std::int64_t> generated = input_ids;
    std::vector<StepMetric> metrics;
    metrics.reserve(static_cast<std::size_t>(options_.max_new_tokens));

    std::cout << "bucket-shape greedy decode\n"
              << "  prompt tokens: " << input_ids.size() << '\n'
              << "  warmup runs: " << options_.warmup_runs << '\n'
              << "  max new tokens: " << options_.max_new_tokens << '\n'
              << "  shape buckets: " << BucketsToString(buckets) << '\n';

    int current_bucket = 0;
    for (int step = 0; step < options_.max_new_tokens; ++step) {
        const int bucket = SelectShapeBucket(buckets, generated.size());
        if (bucket != current_bucket) {
            std::cout << "bucket switch step=" << step
                      << " seq_len=" << generated.size()
                      << " bucket=" << bucket << '\n';
            current_bucket = bucket;
        }

        double model_wall_ms = 0.0;
        double logits_postprocess_ms = 0.0;
        double sampler_ms = 0.0;
        ForwardOutputs outputs;
        {
            ScopedTimer timer(&model_wall_ms);
            outputs = trt_backend->ForwardFixedShape(ForwardInputs{generated, false}, bucket);
        }

        std::vector<float> last_logits;
        {
            ScopedTimer timer(&logits_postprocess_ms);
            last_logits = LastTokenLogits(outputs);
        }

        std::int64_t next_token = 0;
        {
            ScopedTimer timer(&sampler_ms);
            next_token = sampler.Argmax(last_logits);
        }
        generated.push_back(next_token);
        metrics.push_back(MakeStepMetric(step, model_wall_ms, logits_postprocess_ms, sampler_ms, next_token, outputs));
        const auto& metric = metrics.back();

        std::cout << "step " << step
                  << " next_token=" << next_token
                  << " latency_ms=" << metric.latency_ms
                  << " enqueue_ms=" << metric.enqueue_ms
                  << " d2h_ms=" << metric.d2h_ms
                  << " sampler_ms=" << metric.sampler_ms
                  << " total_tokens=" << generated.size()
                  << " bucket=" << bucket
                  << '\n';
        if (backend->Spec().IsEos(next_token)) break;
    }

    WriteStepCsv(options_.output_csv_path, options_.model_path, ToString(options_.provider),
        ToString(options_.mode), static_cast<int>(input_ids.size()), options_.max_new_tokens, metrics);

    const auto total_latency_ms = SumLatencyMs(metrics);
    const auto end_to_end_tokens_per_s = total_latency_ms > 0.0
        ? static_cast<double>(metrics.size()) / (total_latency_ms / 1000.0)
        : 0.0;
    const auto tpot_values = TpotLatencies(metrics);
    const auto ttft_ms = metrics.empty() ? 0.0 : metrics.front().step_total_ms;
    const auto tpot_total_ms = SumValues(tpot_values);
    const auto mean_tpot_ms = !tpot_values.empty()
        ? tpot_total_ms / static_cast<double>(tpot_values.size())
        : 0.0;
    const auto decode_tokens_per_s = tpot_total_ms > 0.0
        ? static_cast<double>(tpot_values.size()) / (tpot_total_ms / 1000.0)
        : 0.0;

    std::cout << "bucket-shape decode completed\n"
              << "  generated tokens: " << metrics.size() << '\n'
              << "  total latency ms: " << total_latency_ms << '\n'
              << "  ttft ms: " << ttft_ms << '\n'
              << "  mean tpot ms: " << mean_tpot_ms << '\n'
              << "  p50 tpot ms: " << Percentile(tpot_values, 50.0) << '\n'
              << "  p95 tpot ms: " << Percentile(tpot_values, 95.0) << '\n'
              << "  decode tokens/s: " << decode_tokens_per_s << '\n'
              << "  end-to-end tokens/s: " << end_to_end_tokens_per_s << '\n';
    PrintProfileSummary(metrics);
    if (!options_.output_csv_path.empty()) {
        std::cout << "  csv: " << options_.output_csv_path << '\n';
    }

    return 0;
}

int Runtime::RunKvCacheDecode()
{
    const auto input_ids = LoadInputIds(options_.input_ids_path);
    if (input_ids.empty()) throw std::runtime_error("input_ids is empty");
    if (options_.decode_model_path.empty()) {
        throw std::runtime_error("kv_cache mode requires --decode-model");
    }
    if (options_.max_new_tokens <= 0 || options_.warmup_runs < 0) {
        throw std::runtime_error("invalid max_new_tokens or warmup_runs");
    }
    const auto max_seq_len = static_cast<std::int64_t>(input_ids.size()) + options_.max_new_tokens;

    auto backend = CreateConfiguredGenerationBackend(options_.provider, options_.model_spec_path);
    backend->SetSamplingMode(ParseSamplingMode(options_.sampling_mode));
    backend->LoadGenerationEngines(options_.model_path, options_.decode_model_path);
    if (max_seq_len > backend->GenerationSpec().engine_max_sequence_length) {
        throw std::runtime_error("KV cache sequence exceeds ModelSpec engine_max_sequence_length");
    }
    RuntimeSession session(*backend);
    session.Warmup(input_ids, options_.max_new_tokens, options_.warmup_runs);
    RequestState request{"request_0", input_ids, options_.max_new_tokens};
    session.Prefill(request);
    while (!request.IsFinished()) session.DecodeOneToken(request);
    for (auto& metric : request.metrics) {
        metric.managed_kv_bytes = request.kv_allocated_bytes;
    }
    const auto& metrics = request.metrics;
    const double kv_init_ms = request.kv_init_ms;

    std::cout << "KV-cache greedy decode\n"
              << "  prompt tokens: " << input_ids.size() << '\n'
              << "  max new tokens: " << options_.max_new_tokens << '\n'
              << "  warmup runs: " << options_.warmup_runs << '\n'
              << "  cache capacity: " << max_seq_len << '\n'
              << "  managed KV bytes: "
              << request.kv_allocated_bytes
              << '\n';
    std::cout << "prefill next_token=" << metrics.front().next_token
              << " backend_latency_ms=" << metrics.front().latency_ms
              << " token_ready_ms=" << metrics.front().step_total_ms
              << " kv_init_ms=" << kv_init_ms << '\n';

    WriteStepCsv(options_.output_csv_path, options_.model_path, ToString(options_.provider),
        ToString(options_.mode), static_cast<int>(input_ids.size()), options_.max_new_tokens, metrics);
    const auto total_latency_ms = SumLatencyMs(metrics);
    const auto tpot_values = TpotLatencies(metrics);
    const auto tpot_total_ms = SumValues(tpot_values);
    std::cout << "KV-cache decode completed\n"
              << "  generated tokens: " << metrics.size() << '\n'
              << "  ttft ms: " << metrics.front().step_total_ms << '\n'
              << "  kv init ms: " << kv_init_ms << '\n'
              << "  mean tpot ms: " << (tpot_values.empty() ? 0.0 : tpot_total_ms / tpot_values.size()) << '\n'
              << "  p50 tpot ms: " << Percentile(tpot_values, 50.0) << '\n'
              << "  p95 tpot ms: " << Percentile(tpot_values, 95.0) << '\n'
              << "  decode tokens/s: " << (tpot_total_ms > 0 ? tpot_values.size() / (tpot_total_ms / 1000.0) : 0.0) << '\n'
              << "  total measured latency ms: " << total_latency_ms << '\n'
              << "  total request latency incl. KV init ms: " << (total_latency_ms + kv_init_ms) << '\n';
    PrintProfileSummary(metrics);
    if (!options_.output_csv_path.empty()) std::cout << "  csv: " << options_.output_csv_path << '\n';
    return 0;
}

int Runtime::RunScheduler(bool continuous_batch)
{
    if (options_.model_path.empty() || options_.decode_model_path.empty()) {
        throw std::runtime_error("scheduler mode requires --model and --decode-model");
    }
    if (options_.request_input_paths.empty()) {
        throw std::runtime_error("scheduler mode requires --request-inputs");
    }
    if (!options_.request_max_new_tokens.empty()
        && options_.request_max_new_tokens.size() != options_.request_input_paths.size()) {
        throw std::runtime_error("--request-max-new-tokens must match --request-inputs count");
    }
    if (!options_.request_arrival_iterations.empty()
        && options_.request_arrival_iterations.size() != options_.request_input_paths.size()) {
        throw std::runtime_error("--request-arrival-iterations must match --request-inputs count");
    }
    if (!std::is_sorted(options_.request_arrival_iterations.begin(),
            options_.request_arrival_iterations.end())) {
        throw std::runtime_error("request arrival iterations must be nondecreasing for FIFO admission");
    }
    if (options_.max_active_requests <= 0 || options_.max_tokens_per_step <= 0
        || options_.max_new_tokens <= 0 || options_.warmup_runs < 0
        || options_.kv_block_size <= 0 || options_.kv_num_blocks <= 0
        || options_.max_batch_size <= 0) {
        throw std::runtime_error("invalid scheduler limits");
    }

    auto backend = CreateConfiguredGenerationBackend(options_.provider, options_.model_spec_path);
    backend->SetSamplingMode(ParseSamplingMode(options_.sampling_mode));
    backend->LoadGenerationEngines(options_.model_path, options_.decode_model_path);
    if (continuous_batch) {
        if (options_.max_tokens_per_step != 1) {
            throw std::runtime_error(
                "continuous_batch uses one batched decode per iteration; set --max-tokens-per-step 1");
        }
        if (options_.max_active_requests > options_.max_batch_size
            || options_.max_batch_size > backend->Capabilities().max_decode_batch_size) {
            throw std::runtime_error(
                "continuous batch limits exceed --max-batch-size or Engine profile");
        }
    }
    RuntimeSession session(*backend);
    KvBlockManager block_manager(options_.kv_block_size, options_.kv_num_blocks);

    std::vector<RequestState> requests;
    requests.reserve(options_.request_input_paths.size());
    for (std::size_t i = 0; i < options_.request_input_paths.size(); ++i) {
        auto tokens = LoadInputIds(options_.request_input_paths[i]);
        const int output_limit = options_.request_max_new_tokens.empty()
            ? options_.max_new_tokens : options_.request_max_new_tokens[i];
        requests.emplace_back("request_" + std::to_string(i), std::move(tokens), output_limit);
        requests.back().arrival_iteration = options_.request_arrival_iterations.empty()
            ? 0 : options_.request_arrival_iterations[i];
    }
    std::vector<std::size_t> warmup_representatives;
    for (std::size_t index = 0; index < requests.size(); ++index) {
        auto existing = std::find_if(
            warmup_representatives.begin(), warmup_representatives.end(),
            [&](std::size_t candidate) {
                return requests[candidate].prompt_tokens == requests[index].prompt_tokens;
            });
        if (existing == warmup_representatives.end()) {
            warmup_representatives.push_back(index);
        } else if (requests[index].max_new_tokens > requests[*existing].max_new_tokens) {
            *existing = index;
        }
    }
    for (int run = 0; run < options_.warmup_runs; ++run) {
        if (!continuous_batch) {
            for (const auto index : warmup_representatives) {
                session.Warmup(
                    requests[index].prompt_tokens, requests[index].max_new_tokens, 1);
            }
            continue;
        }
        auto warm_batch = [&](const std::vector<std::size_t>& indices) {
            std::vector<RequestState> warmup_requests;
            warmup_requests.reserve(indices.size());
            for (std::size_t slot = 0; slot < indices.size(); ++slot) {
                const auto index = indices[slot];
                const auto& source = requests[index];
                warmup_requests.emplace_back(
                    "batch_warmup_" + std::to_string(index),
                    source.prompt_tokens, source.max_new_tokens);
                session.Prefill(warmup_requests.back());
                warmup_requests.back().batch_slot = static_cast<int>(slot);
            }
            while (true) {
                std::vector<RequestState*> active_warmup;
                for (auto& request : warmup_requests) {
                    if (request.status == RequestStatus::Active) active_warmup.push_back(&request);
                }
                if (active_warmup.empty()) break;
                session.DecodeBatch(BuildBatchPlan(active_warmup, options_.max_batch_size));
            }
        };

        // Materialize the largest batch admitted by this runtime configuration
        // and run it through its complete output lengths.  Then warm any prompt
        // shape/output extent not covered by that first batch.  This avoids both
        // measured staging allocation and a hidden B=8 workspace in B=1 runs.
        const auto warmup_batch_size = std::min({
            requests.size(),
            static_cast<std::size_t>(options_.max_active_requests),
            static_cast<std::size_t>(options_.max_batch_size)});
        std::vector<std::size_t> primary_indices;
        primary_indices.reserve(warmup_batch_size);
        for (std::size_t index = 0; index < warmup_batch_size; ++index) {
            primary_indices.push_back(index);
        }
        warm_batch(primary_indices);
        for (const auto representative : warmup_representatives) {
            const bool covered = std::any_of(
                primary_indices.begin(), primary_indices.end(),
                [&](std::size_t index) {
                    return requests[index].prompt_tokens == requests[representative].prompt_tokens
                        && requests[index].max_new_tokens
                            >= requests[representative].max_new_tokens;
                });
            if (!covered) warm_batch({representative});
        }
    }

    struct TimelineEvent {
        int index;
        int iteration;
        std::string request_id;
        std::string phase;
        int token_step;
        std::int64_t token_id;
        int active_requests;
        std::size_t active_kv_bytes;
        int free_kv_blocks;
        int allocated_kv_blocks;
        int live_kv_tokens;
        int reserved_kv_tokens;
        int block_rounding_waste_tokens;
        int batch_id;
        int batch_slot;
        int batch_size;
        std::int64_t max_past_length;
        std::size_t padding_tokens;
        double start_ms;
        double end_ms;
    };
    struct BatchEvent {
        int batch_id;
        int iteration;
        std::vector<std::string> request_ids;
        std::vector<int> slot_ids;
        std::vector<std::int64_t> past_lengths;
        std::int64_t max_past_length;
        std::size_t padding_tokens;
        double input_prepare_ms;
        double shape_ms;
        double buffer_bind_ms;
        double gather_ms;
        double h2d_ms;
        double enqueue_ms;
        double device_sample_ms;
        double d2h_ms;
        double scatter_ms;
        double sync_ms;
        double backend_total_ms;
        double batch_service_ms;
        std::size_t h2d_bytes;
        std::size_t d2h_bytes;
        std::size_t gather_bytes;
        std::size_t scatter_bytes;
        std::size_t staging_zero_bytes;
        std::size_t scratch_bytes;
        std::size_t scratch_allocated_bytes;
        int buffer_allocation_count;
        std::uint64_t enqueue_before;
        std::uint64_t enqueue_after;
    };
    std::vector<TimelineEvent> timeline;
    std::vector<BatchEvent> batch_events;
    std::deque<std::size_t> waiting;
    std::deque<std::size_t> active;
    for (std::size_t i = 0; i < requests.size(); ++i) waiting.push_back(i);
    const auto scheduler_start = std::chrono::steady_clock::now();
    auto since_start_ms = [&]() {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - scheduler_start).count();
    };

    int event_index = 0;
    int iteration = 0;
    int next_batch_id = 0;
    int slot_reuse_events = 0;
    std::deque<int> free_slots;
    std::vector<int> slot_use_counts(static_cast<std::size_t>(options_.max_active_requests), 0);
    for (int slot = 0; slot < options_.max_active_requests; ++slot) free_slots.push_back(slot);
    std::vector<bool> arrival_recorded(requests.size(), false);
    const auto measured_execution_start = backend->DecodeBatchExecutionCount();
    auto active_kv_bytes = [&]() {
        std::size_t bytes = 0;
        for (const auto& request : requests) {
            if (request.kv_cache != nullptr) {
                bytes += request.kv_cache->AllocatedBytes();
            }
        }
        return bytes;
    };
    auto active_request_count = [&]() {
        return static_cast<int>(std::count_if(
            requests.begin(), requests.end(),
            [](const RequestState& request) { return request.kv_cache != nullptr; }));
    };
    auto record_arrivals = [&]() {
        for (const auto index : waiting) {
            if (requests[index].arrival_iteration > iteration) break;
            if (!arrival_recorded[index]) {
                requests[index].arrival_ms = since_start_ms();
                arrival_recorded[index] = true;
            }
        }
    };
    auto admit = [&]() {
        record_arrivals();
        while (!waiting.empty() && static_cast<int>(active.size()) < options_.max_active_requests) {
            const auto index = waiting.front();
            auto& request = requests[index];
            if (request.arrival_iteration > iteration) break;
            if (!arrival_recorded[index]) throw std::runtime_error("eligible request lacks arrival timestamp");
            const int committed_tokens = static_cast<int>(request.prompt_tokens.size()) + request.max_new_tokens;
            if (committed_tokens > backend->GenerationSpec().engine_max_sequence_length) {
                throw std::runtime_error("scheduler request exceeds ModelSpec engine sequence capacity");
            }
            if (!block_manager.CanReserve(committed_tokens)) {
                if (active.empty()) throw std::runtime_error("KV block pool is too small for the next FIFO request");
                break;
            }
            if (free_slots.empty()) {
                throw std::runtime_error("scheduler slot pool is inconsistent with active requests");
            }
            waiting.pop_front();
            const double start_ms = since_start_ms();
            request.queue_wait_ms = start_ms - request.arrival_ms;
            request.batch_slot = free_slots.front();
            free_slots.pop_front();
            if (slot_use_counts[static_cast<std::size_t>(request.batch_slot)]++ > 0) {
                ++slot_reuse_events;
            }
            const auto alloc_start = std::chrono::steady_clock::now();
            request.block_table = block_manager.Reserve(
                request.request_id, committed_tokens, static_cast<int>(request.prompt_tokens.size()));
            request.block_alloc_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - alloc_start).count();
            request.block_committed_tokens = committed_tokens;
            request.block_reserved_tokens = static_cast<int>(request.block_table.size()) * options_.kv_block_size;
            request.block_rounding_waste_tokens = request.block_reserved_tokens - committed_tokens;
            request.cache_capacity_tokens = std::min(
                request.block_reserved_tokens, backend->GenerationSpec().engine_max_sequence_length);
            session.Prefill(request);
            block_manager.UpdateLiveTokens(request.request_id, static_cast<int>(request.generated_tokens.size()));
            const double end_ms = since_start_ms();
            const auto pool = block_manager.Stats();
            timeline.push_back({event_index++, iteration, request.request_id, "prefill", 0,
                request.metrics.back().next_token, active_request_count(),
                active_kv_bytes(), pool.free_blocks, pool.allocated_blocks, pool.live_tokens,
                pool.reserved_tokens, pool.block_rounding_waste_tokens,
                -1, request.batch_slot, 1,
                static_cast<std::int64_t>(request.prompt_tokens.size()), 0,
                start_ms, end_ms});
            if (request.IsFinished()) {
                request.completed_ms = end_ms;
                const auto release_start = std::chrono::steady_clock::now();
                block_manager.Release(request.request_id);
                request.block_release_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - release_start).count();
                request.kv_cache.reset();
                free_slots.push_back(request.batch_slot);
                const auto released_pool = block_manager.Stats();
                timeline.push_back({event_index++, iteration, request.request_id, "release",
                    request.GeneratedCount(), request.metrics.back().next_token,
                    active_request_count(), active_kv_bytes(),
                    released_pool.free_blocks, released_pool.allocated_blocks,
                    released_pool.live_tokens, released_pool.reserved_tokens,
                    released_pool.block_rounding_waste_tokens,
                    -1, request.batch_slot, 1,
                    static_cast<std::int64_t>(request.prompt_tokens.size()), 0,
                    end_ms, since_start_ms()});
            } else {
                active.push_back(index);
            }
        }
    };

    while (!active.empty() || !waiting.empty()) {
        if (active.empty() && !waiting.empty()
            && requests[waiting.front()].arrival_iteration > iteration) {
            iteration = requests[waiting.front()].arrival_iteration;
        }
        // Arrival iterations are round-start boundaries.  Record and admit
        // eligible requests before this iteration's decode batch is formed.
        admit();
        if (active.empty()) {
            if (waiting.empty()) break;
            continue;
        }
        if (continuous_batch && !active.empty()) {
            std::vector<std::size_t> selected;
            while (!active.empty() && selected.size() < static_cast<std::size_t>(options_.max_batch_size)) {
                selected.push_back(active.front());
                active.pop_front();
            }
            std::vector<RequestState*> batch_requests;
            batch_requests.reserve(selected.size());
            std::vector<std::int64_t> past_lengths;
            for (const auto index : selected) {
                batch_requests.push_back(&requests[index]);
                past_lengths.push_back(
                    static_cast<std::int64_t>(requests[index].generated_tokens.size()) - 1);
            }
            const auto plan = BuildBatchPlan(batch_requests, options_.max_batch_size);
            const int batch_id = next_batch_id++;
            const double start_ms = since_start_ms();
            const auto enqueue_before = backend->DecodeBatchExecutionCount();
            session.DecodeBatch(plan);
            const auto enqueue_after = backend->DecodeBatchExecutionCount();
            const double end_ms = since_start_ms();
            const auto& batch_metric = requests[selected.front()].metrics.back();
            BatchEvent batch_event{batch_id, iteration, {}, plan.slot_ids, past_lengths,
                plan.max_past_length, plan.padding_tokens,
                batch_metric.input_prepare_ms, batch_metric.backend_shape_ms,
                batch_metric.backend_buffer_bind_ms, batch_metric.kv_gather_ms,
                batch_metric.h2d_ms, batch_metric.enqueue_ms,
                batch_metric.device_sample_ms, batch_metric.d2h_ms,
                batch_metric.kv_scatter_ms, batch_metric.sync_ms,
                batch_metric.backend_total_ms, end_ms - start_ms,
                batch_metric.h2d_bytes, batch_metric.d2h_bytes,
                batch_metric.kv_gather_bytes, batch_metric.kv_scatter_bytes,
                batch_metric.kv_staging_zero_bytes,
                batch_metric.batch_scratch_bytes,
                batch_metric.batch_scratch_allocated_bytes,
                batch_metric.buffer_allocation_count,
                enqueue_before, enqueue_after};
            for (const auto index : selected) {
                auto& request = requests[index];
                batch_event.request_ids.push_back(request.request_id);
                block_manager.UpdateLiveTokens(
                    request.request_id, static_cast<int>(request.generated_tokens.size()));
            }
            batch_events.push_back(std::move(batch_event));
            const auto pool = block_manager.Stats();
            const auto kv_bytes = active_kv_bytes();
            for (const auto index : selected) {
                auto& request = requests[index];
                const auto& metric = request.metrics.back();
                timeline.push_back({event_index++, iteration, request.request_id, "decode",
                    metric.step, metric.next_token, active_request_count(),
                    kv_bytes, pool.free_blocks, pool.allocated_blocks, pool.live_tokens,
                    pool.reserved_tokens, pool.block_rounding_waste_tokens,
                    batch_id, request.batch_slot, static_cast<int>(selected.size()),
                    plan.max_past_length, plan.padding_tokens, start_ms, end_ms});
            }
            for (const auto index : selected) {
                auto& request = requests[index];
                const auto& metric = request.metrics.back();
                if (request.IsFinished()) {
                    request.completed_ms = end_ms;
                    const auto release_start = std::chrono::steady_clock::now();
                    block_manager.Release(request.request_id);
                    request.block_release_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - release_start).count();
                    request.kv_cache.reset();
                    free_slots.push_back(request.batch_slot);
                    const auto released_pool = block_manager.Stats();
                    timeline.push_back({event_index++, iteration, request.request_id, "release",
                        request.GeneratedCount(), metric.next_token,
                        active_request_count(), active_kv_bytes(),
                        released_pool.free_blocks, released_pool.allocated_blocks,
                        released_pool.live_tokens, released_pool.reserved_tokens,
                        released_pool.block_rounding_waste_tokens,
                        batch_id, request.batch_slot, static_cast<int>(selected.size()),
                        plan.max_past_length, plan.padding_tokens, end_ms, since_start_ms()});
                } else {
                    active.push_back(index);
                }
            }
        } else {
            // Freeze this round's cohort size. In particular, when a
            // staggered trace has fewer live requests than max_active, do not
            // wrap the deque and advance the same request twice before the
            // next logical arrival boundary. This keeps serial and batched
            // scheduler iterations semantically comparable.
            int budget = ComputeSerialRoundBudget(
                active.size(), options_.max_tokens_per_step);
            while (budget-- > 0 && !active.empty()) {
                const auto index = active.front(); active.pop_front();
                auto& request = requests[index];
                const double start_ms = since_start_ms();
                session.DecodeOneToken(request);
                block_manager.UpdateLiveTokens(
                    request.request_id, static_cast<int>(request.generated_tokens.size()));
                const double end_ms = since_start_ms();
                const auto& metric = request.metrics.back();
                const auto pool = block_manager.Stats();
                timeline.push_back({event_index++, iteration, request.request_id, "decode",
                    metric.step, metric.next_token, active_request_count(),
                    active_kv_bytes(), pool.free_blocks, pool.allocated_blocks, pool.live_tokens,
                    pool.reserved_tokens, pool.block_rounding_waste_tokens,
                    -1, request.batch_slot, 1, metric.max_past_length, 0,
                    start_ms, end_ms});
                if (request.IsFinished()) {
                    request.completed_ms = end_ms;
                    const auto release_start = std::chrono::steady_clock::now();
                    block_manager.Release(request.request_id);
                    request.block_release_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - release_start).count();
                    request.kv_cache.reset();
                    free_slots.push_back(request.batch_slot);
                    const auto released_pool = block_manager.Stats();
                    timeline.push_back({event_index++, iteration, request.request_id, "release",
                        request.GeneratedCount(), metric.next_token,
                        active_request_count(), active_kv_bytes(),
                        released_pool.free_blocks, released_pool.allocated_blocks,
                        released_pool.live_tokens, released_pool.reserved_tokens,
                        released_pool.block_rounding_waste_tokens,
                        -1, request.batch_slot, 1, metric.max_past_length, 0,
                        end_ms, since_start_ms()});
                } else {
                    active.push_back(index);
                }
            }
        }
        ++iteration;
    }
    const double makespan_ms = since_start_ms();

    if (!options_.timeline_csv_path.empty()) {
        std::ofstream out(options_.timeline_csv_path);
        if (!out) throw std::runtime_error("failed to open scheduler timeline CSV");
        out << "event_index,iteration,request_id,phase,token_step,token_id,active_requests,active_kv_bytes,free_kv_blocks,allocated_kv_blocks,live_kv_tokens,reserved_kv_tokens,block_rounding_waste_tokens,batch_id,batch_slot,batch_size,max_past_length,padding_tokens,start_ms,end_ms,duration_ms\n";
        for (const auto& event : timeline) {
            out << event.index << ',' << event.iteration << ',' << event.request_id << ',' << event.phase << ','
                << event.token_step << ',' << event.token_id << ',' << event.active_requests << ',' << event.active_kv_bytes << ','
                << event.free_kv_blocks << ',' << event.allocated_kv_blocks << ',' << event.live_kv_tokens << ','
                << event.reserved_kv_tokens << ',' << event.block_rounding_waste_tokens << ','
                << event.batch_id << ',' << event.batch_slot << ',' << event.batch_size << ','
                << event.max_past_length << ',' << event.padding_tokens << ','
                << event.start_ms << ',' << event.end_ms << ',' << (event.end_ms - event.start_ms) << '\n';
        }
    }
    if (continuous_batch && !options_.batch_csv_path.empty()) {
        std::ofstream out(options_.batch_csv_path);
        if (!out) throw std::runtime_error("failed to open continuous-batch evidence CSV");
        out << "batch_id,iteration,batch_size,request_ids,slot_ids,past_lengths,"
            << "max_past_length,padding_tokens,padding_ratio,input_prepare_ms,shape_ms,"
            << "buffer_bind_ms,kv_gather_ms,h2d_ms,enqueue_ms,device_sample_ms,d2h_ms,kv_scatter_ms,"
            << "sync_ms,backend_total_ms,batch_service_ms,h2d_bytes,d2h_bytes,"
            << "kv_gather_bytes,kv_scatter_bytes,kv_staging_zero_bytes,"
            << "batch_scratch_bytes,batch_scratch_allocated_bytes,"
            << "buffer_allocation_count,enqueue_before,enqueue_after\n";
        for (const auto& batch : batch_events) {
            const double denominator = static_cast<double>(batch.request_ids.size())
                * static_cast<double>(batch.max_past_length);
            out << batch.batch_id << ',' << batch.iteration << ',' << batch.request_ids.size()
                << ",\"";
            for (std::size_t i = 0; i < batch.request_ids.size(); ++i) {
                if (i) out << ' ';
                out << batch.request_ids[i];
            }
            out << "\",\"";
            for (std::size_t i = 0; i < batch.slot_ids.size(); ++i) {
                if (i) out << ' ';
                out << batch.slot_ids[i];
            }
            out << "\",\"";
            for (std::size_t i = 0; i < batch.past_lengths.size(); ++i) {
                if (i) out << ' ';
                out << batch.past_lengths[i];
            }
            out << "\"," << batch.max_past_length << ',' << batch.padding_tokens << ','
                << (denominator > 0.0 ? batch.padding_tokens / denominator : 0.0) << ','
                << batch.input_prepare_ms << ',' << batch.shape_ms << ','
                << batch.buffer_bind_ms << ',' << batch.gather_ms << ','
                << batch.h2d_ms << ',' << batch.enqueue_ms << ','
                << batch.device_sample_ms << ',' << batch.d2h_ms << ','
                << batch.scatter_ms << ',' << batch.sync_ms << ','
                << batch.backend_total_ms << ',' << batch.batch_service_ms << ','
                << batch.h2d_bytes << ',' << batch.d2h_bytes << ','
                << batch.gather_bytes << ',' << batch.scatter_bytes << ','
                << batch.staging_zero_bytes << ',' << batch.scratch_bytes << ','
                << batch.scratch_allocated_bytes << ',' << batch.buffer_allocation_count << ','
                << batch.enqueue_before << ',' << batch.enqueue_after << '\n';
        }
    }
    if (!options_.request_summary_csv_path.empty()) {
        std::ofstream out(options_.request_summary_csv_path);
        if (!out) throw std::runtime_error("failed to open request summary CSV");
        out << "request_id,prompt_len,output_len,finish_reason,queue_wait_ms,estimated_ttft_ms,mean_compute_tpot_ms,p95_compute_tpot_ms,end_to_end_ms,kv_allocated_bytes,block_count,block_size,reserved_tokens,committed_tokens,block_rounding_waste_tokens,final_block_utilization,block_alloc_ms,block_release_ms,block_table,token_sequence,arrival_iteration,arrival_ms,batch_slot\n";
        for (const auto& request : requests) {
            const auto tpot = TpotLatencies(request.metrics);
            const double mean_tpot = tpot.empty() ? 0.0 : SumValues(tpot) / tpot.size();
            out << request.request_id << ',' << request.prompt_tokens.size() << ',' << request.GeneratedCount() << ','
                << ToString(request.finish_reason) << ','
                << request.queue_wait_ms << ','
                << (request.queue_wait_ms + request.block_alloc_ms + request.metrics.front().step_total_ms) << ','
                << mean_tpot << ',' << Percentile(tpot, 95.0) << ','
                << (request.completed_ms - request.arrival_ms) << ','
                << request.kv_allocated_bytes << ',' << request.block_table.size() << ',' << options_.kv_block_size << ','
                << request.block_reserved_tokens << ',' << request.block_committed_tokens << ','
                << request.block_rounding_waste_tokens << ','
                << (request.block_reserved_tokens > 0 ? static_cast<double>(request.generated_tokens.size()) / request.block_reserved_tokens : 0.0) << ','
                << request.block_alloc_ms << ',' << request.block_release_ms << ",\"";
            for (std::size_t i = 0; i < request.block_table.size(); ++i) {
                if (i) out << ' ';
                out << request.block_table[i];
            }
            out << "\",";
            for (std::size_t i = request.prompt_tokens.size(); i < request.generated_tokens.size(); ++i) {
                if (i != request.prompt_tokens.size()) out << ' ';
                out << request.generated_tokens[i];
            }
            out << ',' << request.arrival_iteration << ',' << request.arrival_ms << ','
                << request.batch_slot << '\n';
        }
    }

    std::size_t total_generated = 0;
    std::size_t decode_tokens = 0;
    for (const auto& request : requests) {
        total_generated += static_cast<std::size_t>(request.GeneratedCount());
        decode_tokens += static_cast<std::size_t>(std::max(0, request.GeneratedCount() - 1));
    }
    const auto decode_executions = backend->DecodeBatchExecutionCount() - measured_execution_start;
    std::cout << "scheduler completed\n"
              << "  execution: " << (continuous_batch ? "continuous_batch" : "serial_round_robin") << '\n'
              << "  requests: " << requests.size() << '\n'
              << "  max active requests: " << options_.max_active_requests << '\n'
              << "  max tokens per step: " << options_.max_tokens_per_step << '\n'
              << "  max batch size: " << (continuous_batch ? options_.max_batch_size : 1) << '\n'
              << "  KV block size: " << options_.kv_block_size << '\n'
              << "  KV pool blocks: " << options_.kv_num_blocks << '\n'
              << "  generated tokens: " << total_generated << '\n'
              << "  decode tokens: " << decode_tokens << '\n'
              << "  decode enqueues: " << decode_executions << '\n'
              << "  decode tokens/enqueue: "
              << (decode_executions > 0 ? static_cast<double>(decode_tokens) / decode_executions : 0.0) << '\n'
              << "  slot reuse events: " << slot_reuse_events << '\n'
              << "  makespan ms: " << makespan_ms << '\n'
              << "  aggregate tokens/s: " << (makespan_ms > 0 ? total_generated / (makespan_ms / 1000.0) : 0.0) << '\n';
    return 0;
}

Mode ParseMode(const std::string& value)
{
    if (value == "forward") {
        return Mode::Forward;
    }
    if (value == "naive") {
        return Mode::Naive;
    }
    if (value == "fixed_shape") {
        return Mode::FixedShape;
    }
    if (value == "bucket_shape") {
        return Mode::BucketShape;
    }
    if (value == "kv_cache") {
        return Mode::KvCache;
    }
    if (value == "scheduler") {
        return Mode::Scheduler;
    }
    if (value == "continuous_batch") {
        return Mode::ContinuousBatch;
    }
    throw std::runtime_error("Unknown mode: " + value);
}

Provider ParseProvider(const std::string& value)
{
    if (value == "cpu") {
        return Provider::Cpu;
    }
    if (value == "cuda") {
        return Provider::Cuda;
    }
    if (value == "tensorrt") {
        return Provider::TensorRt;
    }
    throw std::runtime_error("Unknown provider: " + value);
}

std::vector<int> ParseShapeBuckets(const std::string& value)
{
    std::vector<int> buckets;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) {
            throw std::runtime_error("Empty shape bucket in: " + value);
        }
        std::size_t consumed = 0;
        const int bucket = std::stoi(item, &consumed);
        if (consumed != item.size()) {
            throw std::runtime_error("Invalid shape bucket in: " + value);
        }
        buckets.push_back(bucket);
    }
    ValidateShapeBuckets(buckets);
    return buckets;
}

std::string ToString(Mode mode)
{
    switch (mode) {
    case Mode::Forward:
        return "forward";
    case Mode::Naive:
        return "naive";
    case Mode::FixedShape:
        return "fixed_shape";
    case Mode::BucketShape:
        return "bucket_shape";
    case Mode::KvCache:
        return "kv_cache";
    case Mode::Scheduler:
        return "scheduler";
    case Mode::ContinuousBatch:
        return "continuous_batch";
    }
    return "unknown";
}

std::string ToString(Provider provider)
{
    switch (provider) {
    case Provider::Cpu:
        return "cpu";
    case Provider::Cuda:
        return "cuda";
    case Provider::TensorRt:
        return "tensorrt";
    }
    return "unknown";
}

} // namespace minillm
