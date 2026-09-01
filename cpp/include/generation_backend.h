#pragma once

#include "execution_backend.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace minillm {

enum class SamplingMode {
    HostLogits,
    DeviceArgmax,
};

enum class DecodeBatchLayout {
    Padded,
    Ragged,
};

struct GenerationCapabilities {
    int max_decode_batch_size{1};
    DecodeBatchLayout decode_batch_layout{DecodeBatchLayout::Padded};
    bool supports_device_argmax{false};
};

class KvCacheHandle {
public:
    virtual ~KvCacheHandle() = default;
    KvCacheHandle(const KvCacheHandle&) = delete;
    KvCacheHandle& operator=(const KvCacheHandle&) = delete;

    std::int64_t Capacity() const { return capacity_; }
    std::size_t AllocatedBytes() const { return allocated_bytes_; }

protected:
    KvCacheHandle() = default;
    std::int64_t capacity_{0};
    std::size_t allocated_bytes_{0};
};

struct KvBatchInput {
    KvCacheHandle* cache{nullptr};
    std::int64_t token_id{0};
    std::int64_t past_length{0};
};

// Model-independent prefill/decode contract consumed by RuntimeSession.  A
// backend owns the physical cache implementation; the session only owns its
// request lifetime and observes capacity/bytes.
class GenerationBackend {
public:
    virtual ~GenerationBackend() = default;
    virtual void SetGenerationSpec(const ModelSpec& spec) = 0;
    virtual const ModelSpec& GenerationSpec() const = 0;
    virtual void LoadGenerationEngines(
        const std::string& prefill_model_path,
        const std::string& decode_model_path) = 0;
    virtual GenerationCapabilities Capabilities() const = 0;
    virtual void SetSamplingMode(SamplingMode mode) = 0;
    virtual void PrepareFixedShapeBuffers(std::int64_t max_seq_len) = 0;
    virtual ForwardOutputs ForwardFixedShape(
        const ForwardInputs& inputs, std::int64_t max_seq_len) = 0;
    virtual std::unique_ptr<KvCacheHandle> CreateKvCache(
        std::int64_t prompt_len, std::int64_t max_seq_len) = 0;
    virtual ForwardOutputs ForwardKvToken(
        KvCacheHandle& cache, std::int64_t token_id, std::int64_t past_len) = 0;
    virtual std::uint64_t DecodeBatchExecutionCount() const = 0;
    virtual ForwardOutputs ForwardKvBatch(const std::vector<KvBatchInput>& inputs) = 0;
};

std::unique_ptr<GenerationBackend> CreateGenerationBackend(Provider provider);

SamplingMode ParseSamplingMode(const std::string& value);
std::string ToString(SamplingMode mode);

} // namespace minillm
