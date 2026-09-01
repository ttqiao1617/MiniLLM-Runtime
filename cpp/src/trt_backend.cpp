#include "trt_backend.h"

#include "device_argmax.h"
#include "kv_staging_kernels.h"
#include "profiler.h"
#include "tensor_utils.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace minillm {
namespace {

constexpr const char* kInputIdsName = "input_ids";
constexpr const char* kAttentionMaskName = "attention_mask";
constexpr const char* kPositionIdsName = "position_ids";
constexpr const char* kLogitsName = "logits";
constexpr std::size_t kTopK = 5;

void CheckCuda(cudaError_t status, const std::string& label)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(label + ": " + cudaGetErrorString(status));
    }
}

class CudaEventSpan {
public:
    explicit CudaEventSpan(std::string label)
        : label_(std::move(label))
    {
        CheckCuda(cudaEventCreate(&start_), "cudaEventCreate(" + label_ + ".start)");
        CheckCuda(cudaEventCreate(&stop_), "cudaEventCreate(" + label_ + ".stop)");
    }

    ~CudaEventSpan()
    {
        if (start_ != nullptr) {
            cudaEventDestroy(start_);
        }
        if (stop_ != nullptr) {
            cudaEventDestroy(stop_);
        }
    }

    CudaEventSpan(const CudaEventSpan&) = delete;
    CudaEventSpan& operator=(const CudaEventSpan&) = delete;

    void Start(cudaStream_t stream)
    {
        CheckCuda(cudaEventRecord(start_, stream), "cudaEventRecord(" + label_ + ".start)");
    }

    void Stop(cudaStream_t stream)
    {
        CheckCuda(cudaEventRecord(stop_, stream), "cudaEventRecord(" + label_ + ".stop)");
    }

    double ElapsedMs() const
    {
        float elapsed_ms = 0.0F;
        CheckCuda(cudaEventElapsedTime(&elapsed_ms, start_, stop_), "cudaEventElapsedTime(" + label_ + ")");
        return static_cast<double>(elapsed_ms);
    }

private:
    std::string label_;
    cudaEvent_t start_{nullptr};
    cudaEvent_t stop_{nullptr};
};

std::vector<char> ReadBinaryFile(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open TensorRT engine: " + path);
    }
    in.seekg(0, std::ios::end);
    const auto size = in.tellg();
    if (size <= 0) {
        throw std::runtime_error("TensorRT engine is empty: " + path);
    }
    in.seekg(0, std::ios::beg);
    std::vector<char> data(static_cast<std::size_t>(size));
    in.read(data.data(), size);
    if (!in) {
        throw std::runtime_error("Failed to read TensorRT engine: " + path);
    }
    return data;
}

std::string DimsToString(const nvinfer1::Dims& dims)
{
    std::string result = "[";
    for (int32_t i = 0; i < dims.nbDims; ++i) {
        if (i != 0) {
            result += ", ";
        }
        result += std::to_string(dims.d[i]);
    }
    result += "]";
    return result;
}

std::int64_t NumElements(const nvinfer1::Dims& dims)
{
    if (dims.nbDims <= 0) {
        throw std::runtime_error("Tensor has invalid rank");
    }
    std::int64_t count = 1;
    for (int32_t i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            throw std::runtime_error("Tensor has unresolved or invalid dimension: " + DimsToString(dims));
        }
        if (count > std::numeric_limits<std::int64_t>::max() / dims.d[i]) {
            throw std::runtime_error("Tensor element count overflow");
        }
        count *= dims.d[i];
    }
    return count;
}

std::size_t DataTypeSize(nvinfer1::DataType dtype)
{
    switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
        return sizeof(float);
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kINT8:
        return 1;
    case nvinfer1::DataType::kINT32:
        return sizeof(std::int32_t);
    case nvinfer1::DataType::kBOOL:
        return sizeof(bool);
    case nvinfer1::DataType::kUINT8:
        return 1;
    case nvinfer1::DataType::kFP8:
        return 1;
    case nvinfer1::DataType::kBF16:
        return 2;
    case nvinfer1::DataType::kINT64:
        return sizeof(std::int64_t);
    case nvinfer1::DataType::kINT4:
        return 1;
    case nvinfer1::DataType::kFP4:
        return 1;
    }
    throw std::runtime_error("Unsupported TensorRT data type");
}

nvinfer1::DataType ToTrtDataType(TensorDataType dtype)
{
    switch (dtype) {
    case TensorDataType::Int64: return nvinfer1::DataType::kINT64;
    case TensorDataType::Int32: return nvinfer1::DataType::kINT32;
    case TensorDataType::Float32: return nvinfer1::DataType::kFLOAT;
    case TensorDataType::Float16: return nvinfer1::DataType::kHALF;
    case TensorDataType::BFloat16: return nvinfer1::DataType::kBF16;
    }
    throw std::runtime_error("Unsupported ModelSpec dtype for TensorRT");
}

nvinfer1::Dims MakeInputDims(std::int64_t seq_len)
{
    nvinfer1::Dims dims{};
    dims.nbDims = 2;
    dims.d[0] = 1;
    dims.d[1] = seq_len;
    return dims;
}

void PrintShape(const std::string& label, const nvinfer1::Dims& dims)
{
    std::cout << label << ": " << DimsToString(dims) << "\n";
}

std::vector<std::int64_t> ShapeVector(const nvinfer1::Dims& dims)
{
    std::vector<std::int64_t> shape;
    shape.reserve(static_cast<std::size_t>(dims.nbDims));
    for (int32_t i = 0; i < dims.nbDims; ++i) {
        shape.push_back(dims.d[i]);
    }
    return shape;
}

void RequireTensor(const nvinfer1::ICudaEngine& engine,
                   const std::string& name,
                   nvinfer1::TensorIOMode expected_mode,
                   nvinfer1::DataType expected_dtype)
{
    const auto mode = engine.getTensorIOMode(name.c_str());
    if (mode != expected_mode) {
        throw std::runtime_error("TensorRT engine tensor has unexpected mode: " + name);
    }

    const auto dtype = engine.getTensorDataType(name.c_str());
    if (dtype != expected_dtype) {
        throw std::runtime_error("TensorRT engine tensor has unexpected dtype: " + name
            + " (engine=" + std::to_string(static_cast<int>(dtype))
            + ", spec=" + std::to_string(static_cast<int>(expected_dtype)) + ")");
    }
}

bool HasTensor(const nvinfer1::ICudaEngine& engine, const std::string& name)
{
    for (int32_t index = 0; index < engine.getNbIOTensors(); ++index) {
        if (name == engine.getIOTensorName(index)) return true;
    }
    return false;
}

void RequireShape(const nvinfer1::ICudaEngine& engine, const std::string& name,
                  int rank, const std::vector<std::pair<int, std::int64_t>>& fixed_dims)
{
    const auto dims = engine.getTensorShape(name.c_str());
    if (dims.nbDims != rank) {
        throw std::runtime_error("TensorRT engine tensor has unexpected rank: " + name
            + " shape=" + DimsToString(dims));
    }
    for (const auto& expected : fixed_dims) {
        if (dims.d[expected.first] != expected.second) {
            throw std::runtime_error("TensorRT engine tensor has unexpected shape: " + name
                + " shape=" + DimsToString(dims));
        }
    }
}

void RequireDecodeBatchAxis(const nvinfer1::ICudaEngine& engine, const std::string& name)
{
    const auto dims = engine.getTensorShape(name.c_str());
    if (dims.nbDims <= 0 || (dims.d[0] != 1 && dims.d[0] != -1)) {
        throw std::runtime_error("TensorRT decode tensor has invalid batch axis: " + name
            + " shape=" + DimsToString(dims));
    }
}

void ValidateProfileMaximum(const nvinfer1::ICudaEngine& engine, const std::string& name,
                            int sequence_axis, std::int64_t required_max)
{
    if (engine.getNbOptimizationProfiles() <= 0) return;
    const auto max_dims = engine.getProfileShape(
        name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
    if (max_dims.nbDims <= sequence_axis || max_dims.d[sequence_axis] < required_max) {
        throw std::runtime_error("TensorRT profile max does not cover ModelSpec engine limit for "
            + name + ": max=" + DimsToString(max_dims)
            + ", required=" + std::to_string(required_max));
    }
}

int CommonIoCount(const ModelSpec& spec)
{
    return spec.requires_position_ids ? 4 : 3;
}

void ValidateCommonContract(const nvinfer1::ICudaEngine& engine, const ModelSpec& spec,
                            bool decode_engine)
{
    RequireTensor(engine, kInputIdsName, nvinfer1::TensorIOMode::kINPUT,
        ToTrtDataType(spec.input_ids_dtype));
    RequireTensor(engine, kAttentionMaskName, nvinfer1::TensorIOMode::kINPUT,
        ToTrtDataType(spec.attention_mask_dtype));
    RequireTensor(engine, kLogitsName, nvinfer1::TensorIOMode::kOUTPUT,
        ToTrtDataType(spec.logits_dtype));
    if (decode_engine) {
        RequireShape(engine, kInputIdsName, 2, {{1, 1}});
        RequireShape(engine, kAttentionMaskName, 2, {});
        RequireShape(engine, kLogitsName, 3, {{1, 1}, {2, spec.vocab_size}});
        RequireDecodeBatchAxis(engine, kInputIdsName);
        RequireDecodeBatchAxis(engine, kAttentionMaskName);
        RequireDecodeBatchAxis(engine, kLogitsName);
    } else {
        RequireShape(engine, kInputIdsName, 2, {{0, 1}});
        RequireShape(engine, kAttentionMaskName, 2, {{0, 1}});
        RequireShape(engine, kLogitsName, 3, {{0, 1}, {2, spec.vocab_size}});
    }
    if (spec.requires_position_ids) {
        RequireTensor(engine, kPositionIdsName, nvinfer1::TensorIOMode::kINPUT,
            ToTrtDataType(spec.position_ids_dtype));
        if (decode_engine) {
            RequireShape(engine, kPositionIdsName, 2, {{1, 1}});
            RequireDecodeBatchAxis(engine, kPositionIdsName);
        } else {
            RequireShape(engine, kPositionIdsName, 2, {{0, 1}});
        }
    } else if (HasTensor(engine, kPositionIdsName)) {
        throw std::runtime_error("TensorRT engine exposes position_ids but ModelSpec disables it");
    }
    if (!decode_engine) {
        ValidateProfileMaximum(engine, kInputIdsName, 1, spec.engine_max_sequence_length);
        ValidateProfileMaximum(engine, kAttentionMaskName, 1, spec.engine_max_sequence_length);
        if (spec.requires_position_ids) {
            ValidateProfileMaximum(engine, kPositionIdsName, 1, spec.engine_max_sequence_length);
        }
    }
}

bool ValidatePrefillContract(const nvinfer1::ICudaEngine& engine, const ModelSpec& spec)
{
    ValidateCommonContract(engine, spec, false);
    int present_count = 0;
    for (int layer = 0; layer < spec.num_hidden_layers; ++layer) {
        for (const bool key : {true, false}) {
            const auto name = spec.PresentTensorName(layer, key);
            if (!HasTensor(engine, name)) continue;
            ++present_count;
            RequireTensor(engine, name, nvinfer1::TensorIOMode::kOUTPUT,
                ToTrtDataType(spec.kv_dtype));
            RequireShape(engine, name, 4,
                {{0, 1}, {1, spec.num_kv_heads}, {3, spec.head_dim}});
        }
    }
    const int expected_kv_count = 2 * spec.num_hidden_layers;
    if (present_count != 0 && present_count != expected_kv_count) {
        throw std::runtime_error("TensorRT prefill engine exposes an incomplete present KV set: found "
            + std::to_string(present_count) + ", expected " + std::to_string(expected_kv_count));
    }
    const int expected_io = CommonIoCount(spec) + present_count;
    if (engine.getNbIOTensors() != expected_io) {
        throw std::runtime_error("TensorRT prefill engine I/O count does not match ModelSpec contract: found "
            + std::to_string(engine.getNbIOTensors()) + ", expected " + std::to_string(expected_io));
    }
    return present_count != 0;
}

int ValidateDecodeContract(const nvinfer1::ICudaEngine& engine, const ModelSpec& spec)
{
    ValidateCommonContract(engine, spec, true);
    const int expected_io = CommonIoCount(spec) + 4 * spec.num_hidden_layers;
    if (engine.getNbIOTensors() != expected_io) {
        throw std::runtime_error("TensorRT decode engine I/O count does not match ModelSpec contract: found "
            + std::to_string(engine.getNbIOTensors()) + ", expected " + std::to_string(expected_io));
    }
    for (int layer = 0; layer < spec.num_hidden_layers; ++layer) {
        for (const bool key : {true, false}) {
            const auto past = spec.PastTensorName(layer, key);
            const auto present = spec.PresentTensorName(layer, key);
            RequireTensor(engine, past, nvinfer1::TensorIOMode::kINPUT,
                ToTrtDataType(spec.kv_dtype));
            RequireTensor(engine, present, nvinfer1::TensorIOMode::kOUTPUT,
                ToTrtDataType(spec.kv_dtype));
            RequireShape(engine, past, 4,
                {{1, spec.num_kv_heads}, {3, spec.head_dim}});
            RequireShape(engine, present, 4,
                {{1, spec.num_kv_heads}, {3, spec.head_dim}});
            RequireDecodeBatchAxis(engine, past);
            RequireDecodeBatchAxis(engine, present);
        }
    }
    ValidateProfileMaximum(engine, kAttentionMaskName, 1, spec.engine_max_sequence_length);
    ValidateProfileMaximum(engine, spec.PastTensorName(0, true), 2,
        spec.engine_max_sequence_length - 1);

    int max_batch_size = 1;
    if (engine.getNbOptimizationProfiles() > 0) {
        for (const auto selector : {nvinfer1::OptProfileSelector::kMIN,
                 nvinfer1::OptProfileSelector::kOPT,
                 nvinfer1::OptProfileSelector::kMAX}) {
            const auto token = engine.getProfileShape(kInputIdsName, 0, selector);
            const auto mask = engine.getProfileShape(kAttentionMaskName, 0, selector);
            const auto past = engine.getProfileShape(
                spec.PastTensorName(0, true).c_str(), 0, selector);
            if (token.nbDims != 2 || token.d[0] <= 0 || token.d[1] != 1
                || mask.nbDims != 2 || mask.d[0] != token.d[0]
                || past.nbDims != 4 || past.d[0] != token.d[0]
                || past.d[1] != spec.num_kv_heads || past.d[2] <= 0
                || past.d[3] != spec.head_dim || mask.d[1] != past.d[2] + 1) {
                throw std::runtime_error("TensorRT decode profile has incoherent batch/sequence shapes");
            }
            if (spec.requires_position_ids) {
                const auto positions = engine.getProfileShape(kPositionIdsName, 0, selector);
                if (positions.nbDims != 2 || positions.d[0] != token.d[0]
                    || positions.d[1] != 1) {
                    throw std::runtime_error("TensorRT decode profile has incoherent position_ids shape");
                }
            }
            for (int layer = 0; layer < spec.num_hidden_layers; ++layer) {
                for (const bool key : {true, false}) {
                    const auto dims = engine.getProfileShape(
                        spec.PastTensorName(layer, key).c_str(), 0, selector);
                    if (dims.nbDims != 4 || dims.d[0] != token.d[0]
                        || dims.d[1] != spec.num_kv_heads || dims.d[2] != past.d[2]
                        || dims.d[3] != spec.head_dim) {
                        throw std::runtime_error(
                            "TensorRT decode profile KV shapes are not coherent");
                    }
                }
            }
            if (selector == nvinfer1::OptProfileSelector::kMAX) {
                max_batch_size = token.d[0];
            }
        }
    }
    return max_batch_size;
}

std::vector<std::uint8_t> EncodeIntegers(
    const std::vector<std::int64_t>& values, TensorDataType dtype, const std::string& label)
{
    std::vector<std::uint8_t> bytes(values.size() * ElementSize(dtype));
    if (dtype == TensorDataType::Int64) {
        if (!values.empty()) std::memcpy(bytes.data(), values.data(), bytes.size());
        return bytes;
    }
    if (dtype != TensorDataType::Int32) {
        throw std::runtime_error(label + " must use int32 or int64 storage");
    }
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (values[index] < std::numeric_limits<std::int32_t>::min()
            || values[index] > std::numeric_limits<std::int32_t>::max()) {
            throw std::runtime_error(label + " value exceeds int32 range");
        }
        const auto value = static_cast<std::int32_t>(values[index]);
        std::memcpy(bytes.data() + index * sizeof(value), &value, sizeof(value));
    }
    return bytes;
}

std::vector<std::int64_t> PositionIds(std::size_t count, std::int64_t first = 0)
{
    std::vector<std::int64_t> positions(count);
    for (std::size_t index = 0; index < count; ++index) {
        positions[index] = first + static_cast<std::int64_t>(index);
    }
    return positions;
}

float HalfToFloat(std::uint16_t half)
{
    const std::uint32_t sign = static_cast<std::uint32_t>(half & 0x8000U) << 16U;
    std::uint32_t exponent = (half >> 10U) & 0x1FU;
    std::uint32_t mantissa = half & 0x03FFU;
    std::uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1U;
                ++shift;
            }
            mantissa &= 0x03FFU;
            const std::uint32_t float_exponent = static_cast<std::uint32_t>(127 - 14 - shift);
            bits = sign | (float_exponent << 23U) | (mantissa << 13U);
        }
    } else if (exponent == 0x1FU) {
        bits = sign | 0x7F800000U | (mantissa << 13U);
    } else {
        exponent += static_cast<std::uint32_t>(127 - 15);
        bits = sign | (exponent << 23U) | (mantissa << 13U);
    }
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::vector<float> DecodeFloats(
    const std::vector<std::uint8_t>& bytes, std::size_t count, TensorDataType dtype)
{
    if (bytes.size() != count * ElementSize(dtype)) {
        throw std::runtime_error("Logits host buffer size does not match ModelSpec dtype");
    }
    std::vector<float> output(count);
    if (dtype == TensorDataType::Float32) {
        if (!bytes.empty()) std::memcpy(output.data(), bytes.data(), bytes.size());
        return output;
    }
    for (std::size_t index = 0; index < count; ++index) {
        std::uint16_t value = 0;
        std::memcpy(&value, bytes.data() + index * sizeof(value), sizeof(value));
        if (dtype == TensorDataType::Float16) {
            output[index] = HalfToFloat(value);
        } else if (dtype == TensorDataType::BFloat16) {
            const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
            std::memcpy(&output[index], &bits, sizeof(float));
        } else {
            throw std::runtime_error("Logits dtype is not floating point");
        }
    }
    return output;
}

void ValidateRuntimeKvShape(const nvinfer1::Dims& dims, const ModelSpec& spec,
                            std::int64_t sequence_length, const std::string& name,
                            std::int64_t batch_size = 1)
{
    if (dims.nbDims != 4 || dims.d[0] != batch_size || dims.d[1] != spec.num_kv_heads
        || dims.d[2] != sequence_length || dims.d[3] != spec.head_dim) {
        throw std::runtime_error("TensorRT runtime KV shape mismatch for " + name
            + ": got " + DimsToString(dims) + ", expected ["
            + std::to_string(batch_size) + ","
            + std::to_string(spec.num_kv_heads) + ","
            + std::to_string(sequence_length) + ","
            + std::to_string(spec.head_dim) + "]");
    }
}

} // namespace

TrtKvCacheHandle::~TrtKvCacheHandle()
{
    if (slab_ != nullptr) cudaFree(slab_);
}

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TRT] " << msg << "\n";
        }
    }
};

TrtBackend::TrtBackend()
    : logger_(std::make_unique<TrtLogger>())
{
}

TrtBackend::~TrtBackend()
{
    ReleaseBuffers();
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

void TrtBackend::ReleaseBuffers()
{
    for (auto& item : buffers_) {
        if (item.second.ptr != nullptr) {
            cudaFree(item.second.ptr);
            item.second.ptr = nullptr;
            item.second.bytes = 0;
        }
    }
    buffers_.clear();
    for (auto& item : decode_buffers_) {
        if (item.second.ptr != nullptr) cudaFree(item.second.ptr);
    }
    decode_buffers_.clear();
}

void* TrtBackend::EnsureDecodeBuffer(const std::string& name, std::size_t bytes)
{
    if (bytes == 0) throw std::runtime_error("Cannot allocate zero-byte decode buffer: " + name);
    auto& buffer = decode_buffers_[name];
    if (buffer.ptr != nullptr && buffer.bytes >= bytes) return buffer.ptr;
    if (buffer.ptr != nullptr) CheckCuda(cudaFree(buffer.ptr), "cudaFree(" + name + ")");
    CheckCuda(cudaMalloc(&buffer.ptr, bytes), "cudaMalloc(" + name + ")");
    buffer.bytes = bytes;
    ++decode_buffer_allocation_count_;
    return buffer.ptr;
}

void TrtBackend::CollectLogitsOrSample(const void* logits_device,
                                       int batch_size,
                                       int vocab_size,
                                       const std::string& buffer_name,
                                       ForwardOutputs* outputs,
                                       ForwardProfile* profile)
{
    if (outputs == nullptr || profile == nullptr || logits_device == nullptr
        || batch_size <= 0 || vocab_size <= 0) {
        throw std::invalid_argument("invalid logits collection arguments");
    }
    outputs->logits_shape = {batch_size, 1, vocab_size};
    CudaEventSpan d2h("generation.d2h");
    if (sampling_mode_ == SamplingMode::DeviceArgmax) {
        auto* sampled_device = static_cast<std::int32_t*>(EnsureDecodeBuffer(
            buffer_name, static_cast<std::size_t>(batch_size) * sizeof(std::int32_t)));
        CudaEventSpan device_sample("generation.device_sample");
        device_sample.Start(stream_);
        LaunchDeviceArgmax(logits_device, model_spec_.logits_dtype,
            batch_size, vocab_size, sampled_device, stream_);
        device_sample.Stop(stream_);

        std::vector<std::int32_t> sampled_host(static_cast<std::size_t>(batch_size));
        d2h.Start(stream_);
        CheckCuda(cudaMemcpyAsync(sampled_host.data(), sampled_device,
            sampled_host.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream_),
            "sampled token IDs D2H");
        d2h.Stop(stream_);
        {
            ScopedTimer timer(&profile->sync_ms);
            CheckCuda(cudaStreamSynchronize(stream_), "device sampling synchronize");
        }
        profile->device_sample_ms = device_sample.ElapsedMs();
        profile->d2h_bytes = sampled_host.size() * sizeof(std::int32_t);
        outputs->sampled_token_ids.assign(sampled_host.begin(), sampled_host.end());
    } else {
        const auto element_count = static_cast<std::size_t>(batch_size)
            * static_cast<std::size_t>(vocab_size);
        const auto logits_bytes = element_count * ElementSize(model_spec_.logits_dtype);
        std::vector<std::uint8_t> logits_host(logits_bytes);
        d2h.Start(stream_);
        CheckCuda(cudaMemcpyAsync(logits_host.data(), logits_device, logits_bytes,
            cudaMemcpyDeviceToHost, stream_), "generation logits D2H");
        d2h.Stop(stream_);
        {
            ScopedTimer timer(&profile->sync_ms);
            CheckCuda(cudaStreamSynchronize(stream_), "host logits synchronize");
        }
        profile->d2h_bytes = logits_bytes;
        outputs->logits = DecodeFloats(
            logits_host, element_count, model_spec_.logits_dtype);
    }
    profile->d2h_ms = d2h.ElapsedMs();
    profile->cuda_event_timing = true;
}

void* TrtBackend::EnsureBuffer(const std::string& name, std::size_t bytes)
{
    if (bytes == 0) {
        throw std::runtime_error("Cannot allocate zero-byte TensorRT buffer: " + name);
    }

    auto& buffer = buffers_[name];
    if (buffer.ptr != nullptr && buffer.bytes >= bytes) {
        if (!context_->setTensorAddress(name.c_str(), buffer.ptr)) {
            throw std::runtime_error("Failed to bind TensorRT tensor address: " + name);
        }
        return buffer.ptr;
    }

    if (buffer.ptr != nullptr) {
        CheckCuda(cudaFree(buffer.ptr), "cudaFree(" + name + ")");
        buffer.ptr = nullptr;
        buffer.bytes = 0;
    }

    CheckCuda(cudaMalloc(&buffer.ptr, bytes), "cudaMalloc(" + name + ")");
    buffer.bytes = bytes;
    if (!context_->setTensorAddress(name.c_str(), buffer.ptr)) {
        throw std::runtime_error("Failed to bind TensorRT tensor address: " + name);
    }
    return buffer.ptr;
}

void TrtBackend::Load(const std::string& model_path)
{
    model_spec_.Validate();
    engine_path_ = model_path;
    const auto engine_bytes = ReadBinaryFile(engine_path_);

    runtime_.reset(nvinfer1::createInferRuntime(*logger_));
    if (runtime_ == nullptr) {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }

    engine_.reset(runtime_->deserializeCudaEngine(engine_bytes.data(), engine_bytes.size()));
    if (engine_ == nullptr) {
        throw std::runtime_error("Failed to deserialize TensorRT engine: " + engine_path_);
    }

    context_.reset(engine_->createExecutionContext());
    if (context_ == nullptr) {
        throw std::runtime_error("Failed to create TensorRT execution context");
    }

    prefill_has_kv_outputs_ = ValidatePrefillContract(*engine_, model_spec_);

    if (stream_ == nullptr) CheckCuda(cudaStreamCreate(&stream_), "cudaStreamCreate");

    std::cout << "[TensorRT] loaded engine: " << engine_path_ << "\n";
    std::cout << "[TensorRT] model spec: " << model_spec_.model_name
              << " layers=" << model_spec_.num_hidden_layers
              << " kv_heads=" << model_spec_.num_kv_heads
              << " head_dim=" << model_spec_.head_dim
              << " engine_max_seq=" << model_spec_.engine_max_sequence_length << "\n";
    std::cout << "[TensorRT] io tensors: " << engine_->getNbIOTensors() << "\n";
    for (int32_t i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* name = engine_->getIOTensorName(i);
        const auto mode = engine_->getTensorIOMode(name);
        std::cout << "  " << i << ": " << name
                  << " mode=" << (mode == nvinfer1::TensorIOMode::kINPUT ? "input" : "output")
                  << " shape=" << DimsToString(engine_->getTensorShape(name)) << "\n";
    }
}

ForwardOutputs TrtBackend::Forward(const ForwardInputs& inputs)
{
    ForwardProfile profile;
    const auto profile_start = std::chrono::steady_clock::now();

    if (engine_ == nullptr || context_ == nullptr) {
        throw std::runtime_error("TensorRT engine is not loaded");
    }
    if (inputs.input_ids.empty()) {
        throw std::runtime_error("input_ids is empty");
    }

    const auto seq_len = static_cast<std::int64_t>(inputs.input_ids.size());
    if (seq_len > model_spec_.engine_max_sequence_length) {
        throw std::runtime_error("input_ids length exceeds ModelSpec engine_max_sequence_length");
    }
    const auto input_dims = MakeInputDims(seq_len);
    {
        ScopedTimer timer(&profile.shape_ms);
        if (!context_->setInputShape(kInputIdsName, input_dims)) {
            throw std::runtime_error("Failed to set TensorRT input_ids shape: " + DimsToString(input_dims));
        }
        if (!context_->setInputShape(kAttentionMaskName, input_dims)) {
            throw std::runtime_error("Failed to set TensorRT attention_mask shape: " + DimsToString(input_dims));
        }
        if (model_spec_.requires_position_ids
            && !context_->setInputShape(kPositionIdsName, input_dims)) {
            throw std::runtime_error("Failed to set TensorRT position_ids shape: " + DimsToString(input_dims));
        }

        const int32_t missing_shapes = context_->inferShapes(0, nullptr);
        if (missing_shapes != 0) {
            throw std::runtime_error("TensorRT shape inference failed or missing input specs: " + std::to_string(missing_shapes));
        }
    }

    std::vector<std::uint8_t> input_host;
    std::vector<std::uint8_t> mask_host;
    std::vector<std::uint8_t> position_host;
    {
        ScopedTimer timer(&profile.input_prepare_ms);
        input_host = EncodeIntegers(inputs.input_ids, model_spec_.input_ids_dtype, kInputIdsName);
        mask_host = EncodeIntegers(
            std::vector<std::int64_t>(inputs.input_ids.size(), 1),
            model_spec_.attention_mask_dtype, kAttentionMaskName);
        if (model_spec_.requires_position_ids) {
            position_host = EncodeIntegers(PositionIds(inputs.input_ids.size()),
                model_spec_.position_ids_dtype, kPositionIdsName);
        }
    }

    auto allocate_and_bind = [&](const std::string& name, std::size_t bytes) {
        void* ptr = nullptr;
        CheckCuda(cudaMalloc(&ptr, bytes), "cudaMalloc(" + name + ")");
        buffers_[name] = DeviceBuffer{ptr, bytes};
        if (!context_->setTensorAddress(name.c_str(), ptr)) {
            throw std::runtime_error("Failed to bind TensorRT tensor address: " + name);
        }
        return ptr;
    };

    void* input_ids_device = nullptr;
    void* attention_mask_device = nullptr;
    void* position_ids_device = nullptr;
    nvinfer1::Dims logits_dims{};
    std::size_t logits_bytes = 0;
    {
        ScopedTimer timer(&profile.buffer_bind_ms);
        ReleaseBuffers();
        input_ids_device = allocate_and_bind(kInputIdsName, input_host.size());
        attention_mask_device = allocate_and_bind(kAttentionMaskName, mask_host.size());
        if (model_spec_.requires_position_ids) {
            position_ids_device = allocate_and_bind(kPositionIdsName, position_host.size());
        }

        for (int32_t i = 0; i < engine_->getNbIOTensors(); ++i) {
            const std::string name = engine_->getIOTensorName(i);
            if (engine_->getTensorIOMode(name.c_str()) != nvinfer1::TensorIOMode::kOUTPUT) {
                continue;
            }

            const auto dims = context_->getTensorShape(name.c_str());
            const auto elements = NumElements(dims);
            const auto bytes = static_cast<std::size_t>(elements) * DataTypeSize(engine_->getTensorDataType(name.c_str()));
            allocate_and_bind(name, bytes);
            profile.output_tensors += 1;
            profile.bound_output_bytes += bytes;
            if (name == kLogitsName) {
                logits_dims = dims;
                logits_bytes = bytes;
            }
        }
    }

    profile.h2d_bytes = input_host.size() + mask_host.size() + position_host.size();
    CudaEventSpan h2d_timer("trt.h2d");
    h2d_timer.Start(stream_);
    CheckCuda(cudaMemcpyAsync(input_ids_device, input_host.data(), input_host.size(), cudaMemcpyHostToDevice, stream_),
        "cudaMemcpyAsync(input_ids H2D)");
    CheckCuda(cudaMemcpyAsync(attention_mask_device, mask_host.data(), mask_host.size(), cudaMemcpyHostToDevice, stream_),
        "cudaMemcpyAsync(attention_mask H2D)");
    if (model_spec_.requires_position_ids) {
        CheckCuda(cudaMemcpyAsync(position_ids_device, position_host.data(), position_host.size(),
            cudaMemcpyHostToDevice, stream_), "cudaMemcpyAsync(position_ids H2D)");
    }
    h2d_timer.Stop(stream_);

    if (logits_bytes == 0 || buffers_.find(kLogitsName) == buffers_.end()) {
        throw std::runtime_error("TensorRT engine does not expose logits output");
    }
    CudaEventSpan enqueue_timer("trt.enqueue");
    enqueue_timer.Start(stream_);
    if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("TensorRT enqueueV3 failed");
    }
    enqueue_timer.Stop(stream_);

    const auto logits_count = NumElements(logits_dims);
    std::vector<std::uint8_t> logits_host(logits_bytes);
    profile.d2h_bytes = logits_bytes;
    CudaEventSpan d2h_timer("trt.d2h");
    d2h_timer.Start(stream_);
    CheckCuda(cudaMemcpyAsync(logits_host.data(), buffers_.at(kLogitsName).ptr, logits_bytes, cudaMemcpyDeviceToHost, stream_),
        "cudaMemcpyAsync(logits D2H)");
    d2h_timer.Stop(stream_);
    {
        ScopedTimer timer(&profile.sync_ms);
        CheckCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    }
    profile.h2d_ms = h2d_timer.ElapsedMs();
    profile.enqueue_ms = enqueue_timer.ElapsedMs();
    profile.d2h_ms = d2h_timer.ElapsedMs();
    profile.cuda_event_timing = true;
    auto logits = DecodeFloats(logits_host, static_cast<std::size_t>(logits_count), model_spec_.logits_dtype);

    if (logits_dims.nbDims != 3) {
        throw std::runtime_error("Expected TensorRT logits shape [batch, sequence, vocab]");
    }
    if (logits_dims.d[0] != 1
        || (logits_dims.d[1] != seq_len && logits_dims.d[1] != 1)) {
        throw std::runtime_error("Unexpected TensorRT logits batch or sequence dimension");
    }

    const auto vocab_size = logits_dims.d[2];
    const auto sequence_index = logits_dims.d[1] == 1 ? 0 : seq_len - 1;
    const float* last_token_logits = logits.data()
        + static_cast<std::ptrdiff_t>(sequence_index * vocab_size);

    if (inputs.verbose) {
        PrintShape("[TensorRT] input_ids shape", input_dims);
        PrintShape("[TensorRT] logits shape", logits_dims);
        std::cout << "[TensorRT] last-token top-" << kTopK << ":\n";
        const auto topk = TopK(last_token_logits, vocab_size, kTopK);
        for (std::size_t i = 0; i < topk.size(); ++i) {
            std::cout << "  " << (i + 1) << ": id=" << topk[i].first << ", logit=" << topk[i].second << "\n";
        }
    }

    ForwardOutputs outputs;
    outputs.logits_shape = ShapeVector(logits_dims);
    outputs.logits = std::move(logits);
    profile.total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - profile_start).count();
    outputs.profile = profile;
    return outputs;
}

void TrtBackend::PrepareFixedShapeBuffers(std::int64_t max_seq_len)
{
    if (engine_ == nullptr || context_ == nullptr) {
        throw std::runtime_error("TensorRT engine is not loaded");
    }
    if (max_seq_len <= 0 || max_seq_len > model_spec_.engine_max_sequence_length) {
        throw std::runtime_error("max_seq_len is outside ModelSpec engine capacity");
    }

    const auto input_dims = MakeInputDims(max_seq_len);
    if (!context_->setInputShape(kInputIdsName, input_dims)) {
        throw std::runtime_error("Failed to set TensorRT input_ids shape: " + DimsToString(input_dims));
    }
    if (!context_->setInputShape(kAttentionMaskName, input_dims)) {
        throw std::runtime_error("Failed to set TensorRT attention_mask shape: " + DimsToString(input_dims));
    }
    if (model_spec_.requires_position_ids
        && !context_->setInputShape(kPositionIdsName, input_dims)) {
        throw std::runtime_error("Failed to set TensorRT position_ids shape: " + DimsToString(input_dims));
    }

    const int32_t missing_shapes = context_->inferShapes(0, nullptr);
    if (missing_shapes != 0) {
        throw std::runtime_error("TensorRT shape inference failed or missing input specs: " + std::to_string(missing_shapes));
    }

    const auto count = static_cast<std::size_t>(max_seq_len);
    EnsureBuffer(kInputIdsName, count * ElementSize(model_spec_.input_ids_dtype));
    EnsureBuffer(kAttentionMaskName, count * ElementSize(model_spec_.attention_mask_dtype));
    if (model_spec_.requires_position_ids) {
        EnsureBuffer(kPositionIdsName, count * ElementSize(model_spec_.position_ids_dtype));
    }

    for (int32_t i = 0; i < engine_->getNbIOTensors(); ++i) {
        const std::string name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name.c_str()) != nvinfer1::TensorIOMode::kOUTPUT) {
            continue;
        }

        const auto dims = context_->getTensorShape(name.c_str());
        const auto elements = NumElements(dims);
        const auto bytes = static_cast<std::size_t>(elements) * DataTypeSize(engine_->getTensorDataType(name.c_str()));
        EnsureBuffer(name, bytes);
    }
}

ForwardOutputs TrtBackend::ForwardFixedShape(const ForwardInputs& inputs, std::int64_t max_seq_len)
{
    ForwardProfile profile;
    const auto profile_start = std::chrono::steady_clock::now();

    if (engine_ == nullptr || context_ == nullptr) {
        throw std::runtime_error("TensorRT engine is not loaded");
    }
    if (inputs.input_ids.empty()) {
        throw std::runtime_error("input_ids is empty");
    }
    if (max_seq_len <= 0 || max_seq_len > model_spec_.engine_max_sequence_length) {
        throw std::runtime_error("max_seq_len is outside ModelSpec engine capacity");
    }
    if (static_cast<std::int64_t>(inputs.input_ids.size()) > max_seq_len) {
        throw std::runtime_error("input_ids length exceeds fixed max_seq_len");
    }

    const auto actual_seq_len = static_cast<std::int64_t>(inputs.input_ids.size());
    const auto input_dims = MakeInputDims(max_seq_len);
    {
        ScopedTimer timer(&profile.shape_ms);
        if (!context_->setInputShape(kInputIdsName, input_dims)) {
            throw std::runtime_error("Failed to set TensorRT input_ids shape: " + DimsToString(input_dims));
        }
        if (!context_->setInputShape(kAttentionMaskName, input_dims)) {
            throw std::runtime_error("Failed to set TensorRT attention_mask shape: " + DimsToString(input_dims));
        }
        if (model_spec_.requires_position_ids
            && !context_->setInputShape(kPositionIdsName, input_dims)) {
            throw std::runtime_error("Failed to set TensorRT position_ids shape: " + DimsToString(input_dims));
        }

        const int32_t missing_shapes = context_->inferShapes(0, nullptr);
        if (missing_shapes != 0) {
            throw std::runtime_error("TensorRT shape inference failed or missing input specs: " + std::to_string(missing_shapes));
        }
    }

    std::vector<std::uint8_t> input_host;
    std::vector<std::uint8_t> mask_host;
    std::vector<std::uint8_t> position_host;
    {
        ScopedTimer timer(&profile.input_prepare_ms);
        std::vector<std::int64_t> padded_input(
            static_cast<std::size_t>(max_seq_len), model_spec_.pad_token_id);
        std::vector<std::int64_t> attention_mask(static_cast<std::size_t>(max_seq_len), 0);
        for (std::size_t i = 0; i < inputs.input_ids.size(); ++i) {
            padded_input[i] = inputs.input_ids[i];
            attention_mask[i] = 1;
        }
        input_host = EncodeIntegers(padded_input, model_spec_.input_ids_dtype, kInputIdsName);
        mask_host = EncodeIntegers(attention_mask, model_spec_.attention_mask_dtype, kAttentionMaskName);
        if (model_spec_.requires_position_ids) {
            position_host = EncodeIntegers(PositionIds(static_cast<std::size_t>(max_seq_len)),
                model_spec_.position_ids_dtype, kPositionIdsName);
        }
    }

    void* input_ids_device = nullptr;
    void* attention_mask_device = nullptr;
    void* position_ids_device = nullptr;
    nvinfer1::Dims logits_dims{};
    {
        ScopedTimer timer(&profile.buffer_bind_ms);
        input_ids_device = EnsureBuffer(kInputIdsName, input_host.size());
        attention_mask_device = EnsureBuffer(kAttentionMaskName, mask_host.size());
        if (model_spec_.requires_position_ids) {
            position_ids_device = EnsureBuffer(kPositionIdsName, position_host.size());
        }

        for (int32_t i = 0; i < engine_->getNbIOTensors(); ++i) {
            const std::string name = engine_->getIOTensorName(i);
            if (engine_->getTensorIOMode(name.c_str()) != nvinfer1::TensorIOMode::kOUTPUT) {
                continue;
            }

            const auto dims = context_->getTensorShape(name.c_str());
            const auto elements = NumElements(dims);
            const auto bytes = static_cast<std::size_t>(elements) * DataTypeSize(engine_->getTensorDataType(name.c_str()));
            EnsureBuffer(name, bytes);
            profile.output_tensors += 1;
            profile.bound_output_bytes += bytes;
            if (name == kLogitsName) {
                logits_dims = dims;
            }
        }
    }

    profile.h2d_bytes = input_host.size() + mask_host.size() + position_host.size();
    CudaEventSpan h2d_timer("trt.fixed.h2d");
    h2d_timer.Start(stream_);
    CheckCuda(cudaMemcpyAsync(input_ids_device, input_host.data(), input_host.size(), cudaMemcpyHostToDevice, stream_),
        "cudaMemcpyAsync(input_ids H2D)");
    CheckCuda(cudaMemcpyAsync(attention_mask_device, mask_host.data(), mask_host.size(), cudaMemcpyHostToDevice, stream_),
        "cudaMemcpyAsync(attention_mask H2D)");
    if (model_spec_.requires_position_ids) {
        CheckCuda(cudaMemcpyAsync(position_ids_device, position_host.data(), position_host.size(),
            cudaMemcpyHostToDevice, stream_), "cudaMemcpyAsync(position_ids H2D)");
    }
    h2d_timer.Stop(stream_);

    if (buffers_.find(kLogitsName) == buffers_.end()) {
        throw std::runtime_error("TensorRT engine does not expose logits output");
    }
    if (logits_dims.nbDims != 3 || logits_dims.d[0] != 1
        || (logits_dims.d[1] != max_seq_len && logits_dims.d[1] != 1)) {
        throw std::runtime_error("Unexpected TensorRT fixed-shape logits dimensions");
    }
    if (logits_dims.d[1] == 1 && actual_seq_len != max_seq_len) {
        throw std::runtime_error(
            "last-token-only TensorRT graph cannot sample a right-padded fixed/bucket shape; "
            "use dynamic naive mode or an exact sequence length");
    }

    CudaEventSpan enqueue_timer("trt.fixed.enqueue");
    enqueue_timer.Start(stream_);
    if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("TensorRT enqueueV3 failed");
    }
    enqueue_timer.Stop(stream_);

    const auto vocab_size = logits_dims.d[2];
    const auto vocab_bytes = static_cast<std::size_t>(vocab_size) * ElementSize(model_spec_.logits_dtype);
    const auto sequence_index = logits_dims.d[1] == 1 ? 0 : actual_seq_len - 1;
    const auto last_token_offset = static_cast<std::size_t>(sequence_index) * vocab_bytes;
    const auto* logits_base = static_cast<const char*>(buffers_.at(kLogitsName).ptr);
    ForwardOutputs outputs;
    CollectLogitsOrSample(logits_base + last_token_offset, 1, vocab_size,
        "sampling.fixed.token_ids", &outputs, &profile);
    profile.h2d_ms = h2d_timer.ElapsedMs();
    profile.enqueue_ms = enqueue_timer.ElapsedMs();

    if (inputs.verbose) {
        PrintShape("[TensorRT fixed] input_ids shape", input_dims);
        std::vector<std::int64_t> last_logits_shape{1, 1, vocab_size};
        PrintShape("[TensorRT fixed] last logits shape", last_logits_shape);
        if (!outputs.logits.empty()) {
            std::cout << "[TensorRT fixed] last-token top-" << kTopK << ":\n";
            const auto topk = TopK(outputs.logits.data(), vocab_size, kTopK);
            for (std::size_t i = 0; i < topk.size(); ++i) {
                std::cout << "  " << (i + 1) << ": id=" << topk[i].first
                          << ", logit=" << topk[i].second << "\n";
            }
        } else {
            std::cout << "[TensorRT fixed] sampled token id: "
                      << outputs.sampled_token_ids.front() << "\n";
        }
    }

    profile.total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - profile_start).count();
    outputs.profile = profile;
    return outputs;
}

void TrtBackend::LoadKvDecodeEngine(const std::string& model_path)
{
    if (runtime_ == nullptr || engine_ == nullptr) {
        throw std::runtime_error("Load the prefill TensorRT engine before the KV decode engine");
    }
    if (!prefill_has_kv_outputs_) {
        throw std::runtime_error("KV decode requires a prefill engine with every ModelSpec present tensor");
    }
    if (model_spec_.engine_max_sequence_length < 2) {
        throw std::runtime_error("KV decode requires engine_max_sequence_length >= 2");
    }
    const auto bytes = ReadBinaryFile(model_path);
    decode_engine_.reset(runtime_->deserializeCudaEngine(bytes.data(), bytes.size()));
    if (decode_engine_ == nullptr) {
        throw std::runtime_error("Failed to deserialize KV decode engine: " + model_path);
    }
    decode_context_.reset(decode_engine_->createExecutionContext());
    if (decode_context_ == nullptr) {
        throw std::runtime_error("Failed to create KV decode execution context");
    }
    decode_max_batch_size_ = ValidateDecodeContract(*decode_engine_, model_spec_);
    decode_enqueue_count_ = 0;
    batch_staging_capacity_ = 0;
    std::cout << "[TensorRT] loaded KV decode engine: " << model_path << "\n"
              << "[TensorRT] decode max batch size: " << decode_max_batch_size_ << "\n";
}

void TrtBackend::LoadGenerationEngines(
    const std::string& prefill_model_path, const std::string& decode_model_path)
{
    if (prefill_model_path.empty() || decode_model_path.empty()) {
        throw std::runtime_error("generation requires both prefill and decode engine paths");
    }
    Load(prefill_model_path);
    LoadKvDecodeEngine(decode_model_path);
}

GenerationCapabilities TrtBackend::Capabilities() const
{
    return {decode_max_batch_size_, DecodeBatchLayout::Padded, true};
}

void TrtBackend::SetSamplingMode(SamplingMode mode)
{
    sampling_mode_ = mode;
}

std::unique_ptr<KvCacheHandle> TrtBackend::CreateKvCache(
    std::int64_t prompt_len, std::int64_t max_seq_len)
{
    if (decode_context_ == nullptr) throw std::runtime_error("KV decode engine is not loaded");
    if (prompt_len <= 0 || max_seq_len < prompt_len
        || max_seq_len > model_spec_.engine_max_sequence_length) {
        throw std::runtime_error("Invalid KV cache prompt/capacity length");
    }
    auto cache = std::unique_ptr<TrtKvCacheHandle>(new TrtKvCacheHandle());
    const auto tensor_count = static_cast<std::size_t>(model_spec_.num_hidden_layers) * 2;
    cache->tensors_.assign(tensor_count, nullptr);
    const auto capacity_bytes = model_spec_.KvTensorBytes(max_seq_len);
    const auto prompt_bytes = model_spec_.KvTensorBytes(prompt_len);
    const auto slab_bytes = model_spec_.KvCacheBytes(max_seq_len, 1);
    CheckCuda(cudaMalloc(&cache->slab_, slab_bytes), "cudaMalloc(kv_slab)");
    auto* slab = static_cast<std::uint8_t*>(cache->slab_);
    const auto row_bytes = static_cast<std::size_t>(model_spec_.head_dim)
        * ElementSize(model_spec_.kv_dtype);
    for (int layer = 0; layer < model_spec_.num_hidden_layers; ++layer) {
        for (int kind = 0; kind < 2; ++kind) {
            const int index = layer * 2 + kind;
            cache->tensors_[index] = slab + static_cast<std::size_t>(index) * capacity_bytes;
            const std::string present = model_spec_.PresentTensorName(layer, kind == 0);
            const auto present_dims = context_->getTensorShape(present.c_str());
            ValidateRuntimeKvShape(present_dims, model_spec_, prompt_len, present);
            const auto source = buffers_.find(present);
            if (source == buffers_.end() || source->second.ptr == nullptr) {
                throw std::runtime_error("Prefill engine output is missing: " + present);
            }
            if (source->second.bytes < prompt_bytes) {
                throw std::runtime_error("Prefill engine KV output is smaller than ModelSpec: " + present);
            }
            CheckCuda(cudaMemcpy2DAsync(cache->tensors_[index],
                    static_cast<std::size_t>(max_seq_len) * row_bytes,
                    source->second.ptr, static_cast<std::size_t>(prompt_len) * row_bytes,
                    static_cast<std::size_t>(prompt_len) * row_bytes,
                    static_cast<std::size_t>(model_spec_.num_kv_heads),
                    cudaMemcpyDeviceToDevice, stream_),
                "cudaMemcpy2DAsync(prefill KV to fixed-stride cache)");
        }
    }
    CheckCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize(KV init)");
    cache->capacity_ = max_seq_len;
    cache->allocated_bytes_ = slab_bytes;
    return cache;
}

ForwardOutputs TrtBackend::ForwardKvToken(
    KvCacheHandle& cache, std::int64_t token_id, std::int64_t past_len)
{
    return ForwardKvBatch({KvBatchInput{&cache, token_id, past_len}});
}

ForwardOutputs TrtBackend::ForwardKvBatch(const std::vector<KvBatchInput>& inputs)
{
    ForwardProfile profile;
    const auto total_start = std::chrono::steady_clock::now();
    const auto allocations_before = decode_buffer_allocation_count_;
    const auto batch_size = static_cast<std::int64_t>(inputs.size());
    const auto tensor_count = static_cast<std::size_t>(model_spec_.num_hidden_layers) * 2;
    if (inputs.empty() || batch_size > decode_max_batch_size_) {
        throw std::runtime_error("KV decode batch is empty or exceeds the Engine profile");
    }
    if (decode_engine_ == nullptr || decode_context_ == nullptr) {
        throw std::runtime_error("KV decode backend is not initialized");
    }

    std::vector<void*> cache_slabs_host;
    std::vector<std::int64_t> past_lengths_host;
    std::vector<std::int64_t> capacities_host;
    cache_slabs_host.reserve(inputs.size());
    past_lengths_host.reserve(inputs.size());
    capacities_host.reserve(inputs.size());
    std::unordered_set<const TrtKvCacheHandle*> unique_caches;
    std::int64_t max_past_len = 0;
    std::size_t padding_tokens = 0;
    for (const auto& input : inputs) {
        auto* cache = dynamic_cast<TrtKvCacheHandle*>(input.cache);
        if (cache == nullptr || cache->slab_ == nullptr
            || cache->tensors_.size() != tensor_count) {
            throw std::runtime_error("batched KV cache belongs to a different/uninitialized backend");
        }
        if (!unique_caches.insert(cache).second) {
            throw std::runtime_error("batched KV decode contains a duplicate cache handle");
        }
        if (input.past_length <= 0 || input.past_length >= cache->capacity_) {
            throw std::runtime_error("batched KV past length is outside cache capacity");
        }
        cache_slabs_host.push_back(cache->slab_);
        past_lengths_host.push_back(input.past_length);
        capacities_host.push_back(cache->capacity_);
        max_past_len = std::max(max_past_len, input.past_length);
    }
    if (max_past_len >= model_spec_.engine_max_sequence_length) {
        throw std::runtime_error("batched KV past length exceeds ModelSpec Engine limit");
    }
    for (const auto& input : inputs) {
        padding_tokens += static_cast<std::size_t>(max_past_len - input.past_length);
    }
    if (!model_spec_.requires_position_ids && padding_tokens != 0) {
        throw std::runtime_error(
            "mixed-length batched KV decode requires explicit per-row position_ids");
    }

    nvinfer1::Dims token_dims{};
    token_dims.nbDims = 2;
    token_dims.d[0] = static_cast<int32_t>(batch_size);
    token_dims.d[1] = 1;
    nvinfer1::Dims mask_dims{};
    mask_dims.nbDims = 2;
    mask_dims.d[0] = static_cast<int32_t>(batch_size);
    mask_dims.d[1] = static_cast<int32_t>(max_past_len + 1);
    nvinfer1::Dims kv_dims{};
    kv_dims.nbDims = 4;
    kv_dims.d[0] = static_cast<int32_t>(batch_size);
    kv_dims.d[1] = model_spec_.num_kv_heads;
    kv_dims.d[2] = static_cast<int32_t>(max_past_len);
    kv_dims.d[3] = model_spec_.head_dim;
    {
        ScopedTimer timer(&profile.shape_ms);
        if (!decode_context_->setInputShape(kInputIdsName, token_dims)
            || !decode_context_->setInputShape(kAttentionMaskName, mask_dims)) {
            throw std::runtime_error("failed to set batched KV token/mask shape");
        }
        if (model_spec_.requires_position_ids
            && !decode_context_->setInputShape(kPositionIdsName, token_dims)) {
            throw std::runtime_error("failed to set batched KV position_ids shape");
        }
        for (int layer = 0; layer < model_spec_.num_hidden_layers; ++layer) {
            for (const bool key : {true, false}) {
                const auto past = model_spec_.PastTensorName(layer, key);
                if (!decode_context_->setInputShape(past.c_str(), kv_dims)) {
                    throw std::runtime_error("failed to set batched KV input shape: " + past);
                }
            }
        }
        const int32_t missing = decode_context_->inferShapes(0, nullptr);
        if (missing != 0) {
            throw std::runtime_error("batched KV shape inference failed: "
                + std::to_string(missing));
        }
        for (int layer = 0; layer < model_spec_.num_hidden_layers; ++layer) {
            for (const bool key : {true, false}) {
                const auto present = model_spec_.PresentTensorName(layer, key);
                ValidateRuntimeKvShape(decode_context_->getTensorShape(present.c_str()),
                    model_spec_, max_past_len + 1, present, batch_size);
            }
        }
    }

    std::vector<std::uint8_t> token_host;
    std::vector<std::uint8_t> mask_host;
    std::vector<std::uint8_t> position_host;
    {
        ScopedTimer timer(&profile.input_prepare_ms);
        std::vector<std::int64_t> token_values;
        std::vector<std::int64_t> position_values;
        std::vector<std::int64_t> mask_values(
            static_cast<std::size_t>(batch_size * (max_past_len + 1)), 0);
        token_values.reserve(inputs.size());
        position_values.reserve(inputs.size());
        for (std::size_t batch_index = 0; batch_index < inputs.size(); ++batch_index) {
            const auto& input = inputs[batch_index];
            token_values.push_back(input.token_id);
            position_values.push_back(input.past_length);
            const auto row = batch_index * static_cast<std::size_t>(max_past_len + 1);
            std::fill(mask_values.begin() + static_cast<std::ptrdiff_t>(row),
                mask_values.begin() + static_cast<std::ptrdiff_t>(
                    row + static_cast<std::size_t>(input.past_length)), 1);
            mask_values[row + static_cast<std::size_t>(max_past_len)] = 1;
        }
        token_host = EncodeIntegers(
            token_values, model_spec_.input_ids_dtype, kInputIdsName);
        mask_host = EncodeIntegers(
            mask_values, model_spec_.attention_mask_dtype, kAttentionMaskName);
        if (model_spec_.requires_position_ids) {
            position_host = EncodeIntegers(
                position_values, model_spec_.position_ids_dtype, kPositionIdsName);
        }
    }

    void* token_device = nullptr;
    void* mask_device = nullptr;
    void* position_device = nullptr;
    void* logits_device = nullptr;
    void* packed_past_slab = nullptr;
    void* packed_present_slab = nullptr;
    void* cache_slabs_device = nullptr;
    void* past_lengths_device = nullptr;
    void* capacities_device = nullptr;
    nvinfer1::Dims logits_dims{};
    std::vector<void*> past_buffers(tensor_count, nullptr);
    std::vector<void*> present_buffers(tensor_count, nullptr);
    const auto per_batch_past_bytes = model_spec_.KvTensorBytes(max_past_len);
    const auto per_batch_present_bytes = model_spec_.KvTensorBytes(max_past_len + 1);
    // Keep two contiguous staging slabs at the largest warmed batch. TensorRT
    // sees one dense region per layer/KV kind, while the fused kernels use a
    // stable tensor stride even when a later active batch becomes smaller.
    batch_staging_capacity_ = std::max(batch_staging_capacity_, static_cast<int>(batch_size));
    const auto staging_batch_capacity = static_cast<std::size_t>(batch_staging_capacity_);
    const auto staging_past_capacity_bytes = model_spec_.KvTensorBytes(
        model_spec_.engine_max_sequence_length - 1);
    const auto staging_present_capacity_bytes = model_spec_.KvTensorBytes(
        model_spec_.engine_max_sequence_length);
    const auto token_capacity_bytes = staging_batch_capacity
        * ElementSize(model_spec_.input_ids_dtype);
    const auto mask_capacity_bytes = staging_batch_capacity
        * static_cast<std::size_t>(model_spec_.engine_max_sequence_length)
        * ElementSize(model_spec_.attention_mask_dtype);
    const auto position_capacity_bytes = staging_batch_capacity
        * ElementSize(model_spec_.position_ids_dtype);
    const auto logits_capacity_bytes = staging_batch_capacity
        * static_cast<std::size_t>(model_spec_.vocab_size)
        * ElementSize(model_spec_.logits_dtype);
    const auto packed_past_tensor_capacity_bytes = staging_batch_capacity
        * staging_past_capacity_bytes;
    const auto packed_present_tensor_capacity_bytes = staging_batch_capacity
        * staging_present_capacity_bytes;
    {
        ScopedTimer timer(&profile.buffer_bind_ms);
        token_device = EnsureDecodeBuffer("batch.input_ids", token_capacity_bytes);
        mask_device = EnsureDecodeBuffer("batch.attention_mask", mask_capacity_bytes);
        if (model_spec_.requires_position_ids) {
            position_device = EnsureDecodeBuffer(
                "batch.position_ids", position_capacity_bytes);
        }
        if (!decode_context_->setTensorAddress(kInputIdsName, token_device)
            || !decode_context_->setTensorAddress(kAttentionMaskName, mask_device)) {
            throw std::runtime_error("failed to bind batched KV token/mask buffers");
        }
        if (model_spec_.requires_position_ids
            && !decode_context_->setTensorAddress(kPositionIdsName, position_device)) {
            throw std::runtime_error("failed to bind batched KV position_ids buffer");
        }
        packed_past_slab = EnsureDecodeBuffer("batch.past.slab",
            tensor_count * packed_past_tensor_capacity_bytes);
        packed_present_slab = EnsureDecodeBuffer("batch.present.slab",
            tensor_count * packed_present_tensor_capacity_bytes);
        cache_slabs_device = EnsureDecodeBuffer("batch.cache_slabs",
            staging_batch_capacity * sizeof(void*));
        past_lengths_device = EnsureDecodeBuffer("batch.past_lengths",
            staging_batch_capacity * sizeof(std::int64_t));
        capacities_device = EnsureDecodeBuffer("batch.capacities",
            staging_batch_capacity * sizeof(std::int64_t));
        for (int layer = 0; layer < model_spec_.num_hidden_layers; ++layer) {
            for (int kind = 0; kind < 2; ++kind) {
                const auto index = static_cast<std::size_t>(layer * 2 + kind);
                const auto past = model_spec_.PastTensorName(layer, kind == 0);
                const auto present = model_spec_.PresentTensorName(layer, kind == 0);
                past_buffers[index] = static_cast<std::uint8_t*>(packed_past_slab)
                    + index * packed_past_tensor_capacity_bytes;
                present_buffers[index] = static_cast<std::uint8_t*>(packed_present_slab)
                    + index * packed_present_tensor_capacity_bytes;
                if (!decode_context_->setTensorAddress(past.c_str(), past_buffers[index])
                    || !decode_context_->setTensorAddress(present.c_str(), present_buffers[index])) {
                    throw std::runtime_error("failed to bind batched KV staging buffers");
                }
                profile.output_tensors += 1;
                profile.bound_output_bytes +=
                    static_cast<std::size_t>(batch_size) * per_batch_present_bytes;
            }
        }
        logits_dims = decode_context_->getTensorShape(kLogitsName);
        if (logits_dims.nbDims != 3 || logits_dims.d[0] != batch_size
            || logits_dims.d[1] != 1 || logits_dims.d[2] != model_spec_.vocab_size) {
            throw std::runtime_error("batched KV logits shape does not match ModelSpec");
        }
        const auto logits_bytes = static_cast<std::size_t>(NumElements(logits_dims))
            * ElementSize(model_spec_.logits_dtype);
        logits_device = EnsureDecodeBuffer("batch.logits", logits_capacity_bytes);
        if (!decode_context_->setTensorAddress(kLogitsName, logits_device)) {
            throw std::runtime_error("failed to bind batched KV logits output");
        }
        profile.output_tensors += 1;
        profile.bound_output_bytes += logits_bytes;
    }
    profile.batch_scratch_bytes = token_host.size() + mask_host.size() + position_host.size()
        + static_cast<std::size_t>(batch_size)
            * (sizeof(void*) + 2 * sizeof(std::int64_t))
        + tensor_count * static_cast<std::size_t>(batch_size)
            * (per_batch_past_bytes + per_batch_present_bytes)
        + static_cast<std::size_t>(batch_size)
            * static_cast<std::size_t>(model_spec_.vocab_size)
            * ElementSize(model_spec_.logits_dtype);
    for (const auto& entry : decode_buffers_) {
        if (entry.first.rfind("batch.", 0) == 0) {
            profile.batch_scratch_allocated_bytes += entry.second.bytes;
        }
    }

    const auto element_size = ElementSize(model_spec_.kv_dtype);
    CudaEventSpan gather("batch.gather");
    gather.Start(stream_);
    const auto descriptor_count = static_cast<std::size_t>(batch_size);
    CheckCuda(cudaMemcpyAsync(cache_slabs_device, cache_slabs_host.data(),
        descriptor_count * sizeof(void*), cudaMemcpyHostToDevice, stream_),
        "batch cache slab pointers H2D");
    CheckCuda(cudaMemcpyAsync(past_lengths_device, past_lengths_host.data(),
        descriptor_count * sizeof(std::int64_t), cudaMemcpyHostToDevice, stream_),
        "batch past lengths H2D");
    CheckCuda(cudaMemcpyAsync(capacities_device, capacities_host.data(),
        descriptor_count * sizeof(std::int64_t), cudaMemcpyHostToDevice, stream_),
        "batch cache capacities H2D");
    LaunchGatherKvStaging(static_cast<const void* const*>(cache_slabs_device),
        static_cast<const std::int64_t*>(past_lengths_device),
        static_cast<const std::int64_t*>(capacities_device), packed_past_slab,
        static_cast<int>(tensor_count), static_cast<int>(batch_size),
        model_spec_.num_kv_heads, model_spec_.head_dim,
        static_cast<int>(max_past_len), static_cast<int>(element_size),
        packed_past_tensor_capacity_bytes, stream_);
    for (const auto& input : inputs) {
        profile.kv_gather_bytes += tensor_count * model_spec_.KvTensorBytes(input.past_length);
        const auto padding = max_past_len - input.past_length;
        if (padding > 0) {
            profile.kv_staging_zero_bytes += tensor_count * model_spec_.KvTensorBytes(padding);
        }
    }
    gather.Stop(stream_);

    profile.h2d_bytes = token_host.size() + mask_host.size() + position_host.size()
        + descriptor_count * (sizeof(void*) + 2 * sizeof(std::int64_t));
    CudaEventSpan h2d("batch.h2d");
    h2d.Start(stream_);
    CheckCuda(cudaMemcpyAsync(token_device, token_host.data(), token_host.size(),
        cudaMemcpyHostToDevice, stream_), "batch token H2D");
    CheckCuda(cudaMemcpyAsync(mask_device, mask_host.data(), mask_host.size(),
        cudaMemcpyHostToDevice, stream_), "batch mask H2D");
    if (model_spec_.requires_position_ids) {
        CheckCuda(cudaMemcpyAsync(position_device, position_host.data(), position_host.size(),
            cudaMemcpyHostToDevice, stream_), "batch position_ids H2D");
    }
    h2d.Stop(stream_);

    CudaEventSpan enqueue("batch.enqueue");
    enqueue.Start(stream_);
    if (!decode_context_->enqueueV3(stream_)) {
        throw std::runtime_error("batched KV decode enqueueV3 failed");
    }
    ++decode_enqueue_count_;
    enqueue.Stop(stream_);

    CudaEventSpan scatter("batch.scatter");
    scatter.Start(stream_);
    LaunchScatterNewKv(packed_present_slab,
        static_cast<void* const*>(cache_slabs_device),
        static_cast<const std::int64_t*>(past_lengths_device),
        static_cast<const std::int64_t*>(capacities_device),
        static_cast<int>(tensor_count), static_cast<int>(batch_size),
        model_spec_.num_kv_heads, model_spec_.head_dim,
        static_cast<int>(max_past_len), static_cast<int>(element_size),
        packed_present_tensor_capacity_bytes, stream_);
    profile.kv_scatter_bytes = tensor_count * static_cast<std::size_t>(batch_size)
        * model_spec_.KvTensorBytes(1);
    scatter.Stop(stream_);

    ForwardOutputs outputs;
    CollectLogitsOrSample(logits_device, static_cast<int>(batch_size),
        model_spec_.vocab_size, "sampling.batch.token_ids", &outputs, &profile);
    profile.kv_gather_ms = gather.ElapsedMs();
    profile.h2d_ms = h2d.ElapsedMs();
    profile.enqueue_ms = enqueue.ElapsedMs();
    profile.kv_scatter_ms = scatter.ElapsedMs();
    profile.batch_size = static_cast<int>(batch_size);
    profile.max_past_length = max_past_len;
    profile.padding_tokens = padding_tokens;
    profile.enqueue_count = 1;
    profile.buffer_allocation_count = static_cast<int>(
        decode_buffer_allocation_count_ - allocations_before);
    profile.total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - total_start).count();
    outputs.profile = profile;
    return outputs;
}

std::string TrtBackend::Name() const
{
    return "tensorrt";
}

} // namespace minillm
