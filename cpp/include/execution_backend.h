#pragma once

#include "model_spec.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace minillm {

enum class Provider;

struct ForwardInputs {
    std::vector<std::int64_t> input_ids;
    bool verbose{true};
};

struct ForwardProfile {
    double total_ms{0.0};
    double input_prepare_ms{0.0};
    double shape_ms{0.0};
    double buffer_bind_ms{0.0};
    double h2d_ms{0.0};
    double enqueue_ms{0.0};
    double device_sample_ms{0.0};
    double d2h_ms{0.0};
    double sync_ms{0.0};
    double kv_gather_ms{0.0};
    double kv_scatter_ms{0.0};
    std::size_t h2d_bytes{0};
    std::size_t d2h_bytes{0};
    std::size_t bound_output_bytes{0};
    std::size_t kv_gather_bytes{0};
    std::size_t kv_scatter_bytes{0};
    std::size_t kv_staging_zero_bytes{0};
    std::size_t padding_tokens{0};
    std::size_t batch_scratch_bytes{0};
    std::size_t batch_scratch_allocated_bytes{0};
    std::int64_t max_past_length{0};
    int batch_size{1};
    int enqueue_count{0};
    int buffer_allocation_count{0};
    int output_tensors{0};
    bool cuda_event_timing{false};
};

struct ForwardOutputs {
    std::vector<float> logits;
    std::vector<std::int64_t> logits_shape;
    // Backends may keep logits on device and return exactly one sampled token
    // per batch row. Host logits remain available as the reference path.
    std::vector<std::int64_t> sampled_token_ids;
    ForwardProfile profile;
};

class ExecutionBackend {
public:
    virtual ~ExecutionBackend() = default;

    virtual void SetModelSpec(const ModelSpec& spec)
    {
        spec.Validate();
        model_spec_ = spec;
    }
    const ModelSpec& Spec() const { return model_spec_; }
    virtual void Load(const std::string& model_path) = 0;
    virtual ForwardOutputs Forward(const ForwardInputs& inputs) = 0;
    virtual std::string Name() const = 0;

protected:
    ModelSpec model_spec_{ModelSpec::Gpt2()};
};

std::unique_ptr<ExecutionBackend> CreateBackend(Provider provider);

} // namespace minillm
