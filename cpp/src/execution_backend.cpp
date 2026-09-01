#include "execution_backend.h"

#include "ort_backend.h"
#include "runtime.h"
#include "trt_backend.h"

#include <memory>
#include <stdexcept>

namespace minillm {

std::unique_ptr<ExecutionBackend> CreateBackend(Provider provider)
{
    switch (provider) {
    case Provider::Cpu:
        return std::make_unique<OrtBackend>();
    case Provider::Cuda:
        throw std::runtime_error("C++ ONNX Runtime CUDA provider is not implemented; use --provider cpu or --provider tensorrt");
    case Provider::TensorRt:
        return std::make_unique<TrtBackend>();
    }

    throw std::runtime_error("Unhandled backend provider");
}

} // namespace minillm
