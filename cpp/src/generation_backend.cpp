#include "generation_backend.h"

#include "runtime.h"
#include "trt_backend.h"

#include <memory>
#include <stdexcept>

namespace minillm {

std::unique_ptr<GenerationBackend> CreateGenerationBackend(Provider provider)
{
    switch (provider) {
    case Provider::TensorRt:
        return std::make_unique<TrtBackend>();
    case Provider::Cpu:
        throw std::runtime_error(
            "generation modes require a KV-capable backend; CPU/ONNX Runtime is forward-only");
    case Provider::Cuda:
        throw std::runtime_error(
            "the native CUDA generation backend is not enabled in this build");
    }
    throw std::runtime_error("Unhandled generation backend provider");
}

SamplingMode ParseSamplingMode(const std::string& value)
{
    if (value == "host") return SamplingMode::HostLogits;
    if (value == "device") return SamplingMode::DeviceArgmax;
    throw std::runtime_error("Unknown sampling mode: " + value);
}

std::string ToString(SamplingMode mode)
{
    switch (mode) {
    case SamplingMode::HostLogits: return "host";
    case SamplingMode::DeviceArgmax: return "device";
    }
    throw std::runtime_error("Unhandled sampling mode");
}

} // namespace minillm
