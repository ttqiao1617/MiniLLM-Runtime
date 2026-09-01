#include "device_argmax.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace minillm {
namespace {

template <typename T>
__device__ float ToFloat(T value)
{
    return static_cast<float>(value);
}

template <>
__device__ float ToFloat<__half>(__half value)
{
    return __half2float(value);
}

template <>
__device__ float ToFloat<__nv_bfloat16>(__nv_bfloat16 value)
{
    return __bfloat162float(value);
}

struct Candidate {
    float value;
    int index;
};

__device__ Candidate Better(Candidate lhs, Candidate rhs)
{
    if (rhs.value > lhs.value || (rhs.value == lhs.value && rhs.index < lhs.index)) {
        return rhs;
    }
    return lhs;
}

template <typename T>
__global__ void ArgmaxRowsKernel(
    const T* logits, int vocab_size, std::int32_t* output_token_ids)
{
    constexpr int kThreads = 256;
    __shared__ float values[kThreads];
    __shared__ int indices[kThreads];
    __shared__ int first_is_nan;

    const int row = blockIdx.x;
    const auto* row_logits = logits + static_cast<std::size_t>(row) * vocab_size;
    if (threadIdx.x == 0) first_is_nan = isnan(ToFloat(row_logits[0]));
    __syncthreads();
    if (first_is_nan) {
        if (threadIdx.x == 0) output_token_ids[row] = 0;
        return;
    }
    Candidate best{-CUDART_INF_F, 0x7fffffff};
    for (int token = threadIdx.x; token < vocab_size; token += blockDim.x) {
        const float value = ToFloat(row_logits[token]);
        if (!isnan(value)) best = Better(best, {value, token});
    }
    values[threadIdx.x] = best.value;
    indices[threadIdx.x] = best.index;
    __syncthreads();

    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const Candidate lhs{values[threadIdx.x], indices[threadIdx.x]};
            const Candidate rhs{values[threadIdx.x + stride], indices[threadIdx.x + stride]};
            const auto reduced = Better(lhs, rhs);
            values[threadIdx.x] = reduced.value;
            indices[threadIdx.x] = reduced.index;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) output_token_ids[row] = indices[0];
}

template <typename T>
void LaunchTyped(const void* logits,
                 int batch_size,
                 int vocab_size,
                 std::int32_t* output_token_ids,
                 cudaStream_t stream)
{
    ArgmaxRowsKernel<<<batch_size, 256, 0, stream>>>(
        static_cast<const T*>(logits), vocab_size, output_token_ids);
}

} // namespace

void LaunchDeviceArgmax(const void* logits,
                        TensorDataType dtype,
                        int batch_size,
                        int vocab_size,
                        std::int32_t* output_token_ids,
                        cudaStream_t stream)
{
    if (logits == nullptr || output_token_ids == nullptr || batch_size <= 0 || vocab_size <= 0) {
        throw std::invalid_argument("device argmax received invalid buffer or shape");
    }
    switch (dtype) {
    case TensorDataType::Float32:
        LaunchTyped<float>(logits, batch_size, vocab_size, output_token_ids, stream);
        break;
    case TensorDataType::Float16:
        LaunchTyped<__half>(logits, batch_size, vocab_size, output_token_ids, stream);
        break;
    case TensorDataType::BFloat16:
        LaunchTyped<__nv_bfloat16>(logits, batch_size, vocab_size, output_token_ids, stream);
        break;
    default:
        throw std::invalid_argument("device argmax requires a floating-point logits dtype");
    }
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("device argmax launch failed: ") + cudaGetErrorString(status));
    }
}

} // namespace minillm
