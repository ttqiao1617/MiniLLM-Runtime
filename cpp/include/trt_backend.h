#pragma once

#include "execution_backend.h"
#include "generation_backend.h"

#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace nvinfer1 {
class ICudaEngine;
class IExecutionContext;
class IRuntime;
} // namespace nvinfer1

namespace minillm {

class TrtLogger;
class TrtBackend;

class TrtKvCacheHandle final : public KvCacheHandle {
public:
    ~TrtKvCacheHandle() override;

private:
    friend class TrtBackend;
    TrtKvCacheHandle() = default;
    void* slab_{nullptr};
    std::vector<void*> tensors_;
};

class TrtBackend : public ExecutionBackend, public GenerationBackend {
public:
    TrtBackend();
    ~TrtBackend() override;

    void Load(const std::string& model_path) override;
    ForwardOutputs Forward(const ForwardInputs& inputs) override;
    void SetGenerationSpec(const ModelSpec& spec) override { SetModelSpec(spec); }
    const ModelSpec& GenerationSpec() const override { return model_spec_; }
    void LoadGenerationEngines(
        const std::string& prefill_model_path,
        const std::string& decode_model_path) override;
    GenerationCapabilities Capabilities() const override;
    void SetSamplingMode(SamplingMode mode) override;
    void PrepareFixedShapeBuffers(std::int64_t max_seq_len) override;
    ForwardOutputs ForwardFixedShape(
        const ForwardInputs& inputs, std::int64_t max_seq_len) override;
    void LoadKvDecodeEngine(const std::string& model_path);
    std::unique_ptr<KvCacheHandle> CreateKvCache(
        std::int64_t prompt_len, std::int64_t max_seq_len) override;
    ForwardOutputs ForwardKvToken(
        KvCacheHandle& cache, std::int64_t token_id, std::int64_t past_len) override;
    std::uint64_t DecodeBatchExecutionCount() const override { return decode_enqueue_count_; }
    ForwardOutputs ForwardKvBatch(const std::vector<KvBatchInput>& inputs) override;
    std::string Name() const override;

private:
    struct DeviceBuffer {
        void* ptr{nullptr};
        std::size_t bytes{0};
    };

    void ReleaseBuffers();
    void* EnsureBuffer(const std::string& name, std::size_t bytes);
    void* EnsureDecodeBuffer(const std::string& name, std::size_t bytes);
    void CollectLogitsOrSample(const void* logits_device,
                               int batch_size,
                               int vocab_size,
                               const std::string& buffer_name,
                               ForwardOutputs* outputs,
                               ForwardProfile* profile);

    std::string engine_path_;
    std::unique_ptr<TrtLogger> logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::unique_ptr<nvinfer1::ICudaEngine> decode_engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> decode_context_;
    cudaStream_t stream_{nullptr};
    std::unordered_map<std::string, DeviceBuffer> buffers_;
    std::unordered_map<std::string, DeviceBuffer> decode_buffers_;
    int batch_staging_capacity_{0};
    bool prefill_has_kv_outputs_{false};
    int decode_max_batch_size_{0};
    std::uint64_t decode_enqueue_count_{0};
    std::uint64_t decode_buffer_allocation_count_{0};
    SamplingMode sampling_mode_{SamplingMode::HostLogits};
};

} // namespace minillm
