#include "ort_backend.h"

#include "profiler.h"
#include "tensor_utils.h"

#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace minillm {
namespace {

constexpr const char* kInputIdsName = "input_ids";
constexpr const char* kAttentionMaskName = "attention_mask";
constexpr const char* kLogitsName = "logits";
constexpr std::size_t kTopK = 5;

std::vector<std::int64_t> ShapeFromTensor(const Ort::Value& tensor)
{
    return tensor.GetTensorTypeAndShapeInfo().GetShape();
}

std::int64_t NumElements(const std::vector<std::int64_t>& shape)
{
    std::int64_t count = 1;
    for (const auto dim : shape) {
        if (dim <= 0) {
            throw std::runtime_error("Unexpected dynamic or invalid output dimension at runtime");
        }
        count *= dim;
    }
    return count;
}

} // namespace

OrtBackend::OrtBackend()
    : env_(std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "minillm-ort"))
{
}

OrtBackend::~OrtBackend() = default;

void OrtBackend::SetModelSpec(const ModelSpec& spec)
{
    spec.Validate();
    const auto gpt2 = ModelSpec::Gpt2();
    if (spec.num_hidden_layers != gpt2.num_hidden_layers
        || spec.num_attention_heads != gpt2.num_attention_heads
        || spec.num_kv_heads != gpt2.num_kv_heads
        || spec.head_dim != gpt2.head_dim || spec.vocab_size != gpt2.vocab_size
        || spec.requires_position_ids || spec.logits_dtype != TensorDataType::Float32) {
        throw std::runtime_error(
            "the C++ ORT backend currently supports only the GPT-2 FP32 two-input contract; "
            "use python/verify_qwen_onnx.py for Qwen ORT correctness");
    }
    ExecutionBackend::SetModelSpec(spec);
}

void OrtBackend::Load(const std::string& model_path)
{
    model_path_ = model_path;

    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_ = std::make_unique<Ort::Session>(*env_, model_path_.c_str(), session_options);
    std::cout << "[ORT] loaded model: " << model_path_ << "\n";
}

ForwardOutputs OrtBackend::Forward(const ForwardInputs& inputs)
{
    ForwardProfile profile;
    const auto profile_start = std::chrono::steady_clock::now();

    if (session_ == nullptr) {
        throw std::runtime_error("ORT session is not loaded");
    }
    if (inputs.input_ids.empty()) {
        throw std::runtime_error("input_ids is empty");
    }

    const auto seq_len = static_cast<std::int64_t>(inputs.input_ids.size());
    std::vector<std::int64_t> input_shape;
    std::vector<std::int64_t> attention_mask;
    {
        ScopedTimer timer(&profile.input_prepare_ms);
        input_shape = {1, seq_len};
        attention_mask.assign(inputs.input_ids.size(), 1);
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // decoder_model.onnx 的两个输入都是 int64，shape 都是 [batch_size, sequence_length]。
    auto input_ids_tensor = Ort::Value::CreateTensor<std::int64_t>(memory_info,
        const_cast<std::int64_t*>(inputs.input_ids.data()), inputs.input_ids.size(), input_shape.data(),
        input_shape.size());
    auto attention_mask_tensor = Ort::Value::CreateTensor<std::int64_t>(memory_info, attention_mask.data(),
        attention_mask.size(), input_shape.data(), input_shape.size());

    std::vector<const char*> input_names{kInputIdsName, kAttentionMaskName};
    std::vector<const char*> output_names{kLogitsName};
    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(2);
    input_tensors.push_back(std::move(input_ids_tensor));
    input_tensors.push_back(std::move(attention_mask_tensor));

    auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(),
        input_tensors.size(), output_names.data(), output_names.size());

    if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
        throw std::runtime_error("ORT did not return logits tensor");
    }

    const auto logits_shape = ShapeFromTensor(output_tensors[0]);
    if (logits_shape.size() != 3) {
        throw std::runtime_error("Expected logits shape [batch, sequence, vocab]");
    }
    if (logits_shape[0] != 1 || logits_shape[1] != seq_len) {
        throw std::runtime_error("Unexpected logits batch or sequence dimension");
    }

    const auto logits_count = NumElements(logits_shape);
    const auto vocab_size = logits_shape[2];
    const float* logits_data = output_tensors[0].GetTensorData<float>();
    const float* last_token_logits = logits_data + static_cast<std::ptrdiff_t>((seq_len - 1) * vocab_size);

    if (inputs.verbose) {
        PrintShape("[ORT] input_ids shape", input_shape);
        PrintShape("[ORT] logits shape", logits_shape);
        std::cout << "[ORT] last-token top-" << kTopK << ":\n";
        const auto topk = TopK(last_token_logits, vocab_size, kTopK);
        for (std::size_t i = 0; i < topk.size(); ++i) {
            std::cout << "  " << (i + 1) << ": id=" << topk[i].first << ", logit=" << topk[i].second << "\n";
        }
    }

    ForwardOutputs outputs;
    outputs.logits_shape = logits_shape;
    outputs.logits.assign(logits_data, logits_data + logits_count);
    profile.output_tensors = 1;
    profile.bound_output_bytes = static_cast<std::size_t>(logits_count) * sizeof(float);
    profile.total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - profile_start).count();
    outputs.profile = profile;
    return outputs;
}

std::string OrtBackend::Name() const
{
    return "ort";
}

} // namespace minillm
