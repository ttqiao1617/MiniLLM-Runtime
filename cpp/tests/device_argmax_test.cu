#include "device_argmax.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void CheckCuda(cudaError_t status, const std::string& label)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(label + ": " + cudaGetErrorString(status));
    }
}

void Expect(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

template <typename T>
std::vector<std::int32_t> Run(
    const std::vector<T>& logits, minillm::TensorDataType dtype, int batch, int vocab)
{
    T* device_logits = nullptr;
    std::int32_t* device_tokens = nullptr;
    CheckCuda(cudaMalloc(&device_logits, logits.size() * sizeof(T)), "cudaMalloc(logits)");
    CheckCuda(cudaMalloc(&device_tokens, static_cast<std::size_t>(batch) * sizeof(std::int32_t)),
        "cudaMalloc(tokens)");
    CheckCuda(cudaMemcpy(device_logits, logits.data(), logits.size() * sizeof(T),
        cudaMemcpyHostToDevice), "cudaMemcpy(logits)");
    minillm::LaunchDeviceArgmax(
        device_logits, dtype, batch, vocab, device_tokens, nullptr);
    std::vector<std::int32_t> tokens(static_cast<std::size_t>(batch));
    CheckCuda(cudaMemcpy(tokens.data(), device_tokens, tokens.size() * sizeof(std::int32_t),
        cudaMemcpyDeviceToHost), "cudaMemcpy(tokens)");
    CheckCuda(cudaFree(device_tokens), "cudaFree(tokens)");
    CheckCuda(cudaFree(device_logits), "cudaFree(logits)");
    return tokens;
}

} // namespace

int main()
{
    try {
        int device_count = 0;
        const auto count_status = cudaGetDeviceCount(&device_count);
        if (count_status == cudaErrorNoDevice || device_count == 0) {
            std::cout << "device_argmax_test skipped: no CUDA device\n";
            return 0;
        }
        CheckCuda(count_status, "cudaGetDeviceCount");

        const std::vector<float> values{
            -4.0F, 3.0F, 1.0F, 3.0F, -2.0F,
            -8.0F, -7.0F, -1.0F, -1.5F, -3.0F,
            std::numeric_limits<float>::quiet_NaN(), 10.0F, 9.0F, 8.0F, 7.0F,
        };
        const auto fp32 = Run(values, minillm::TensorDataType::Float32, 3, 5);
        Expect(fp32 == std::vector<std::int32_t>({1, 2, 0}),
            "FP32 argmax, tie break, or first-NaN semantics mismatch");

        std::vector<__half> half_values;
        half_values.reserve(values.size());
        for (const auto value : values) half_values.push_back(__float2half(value));
        const auto fp16 = Run(half_values, minillm::TensorDataType::Float16, 3, 5);
        Expect(fp16 == fp32, "FP16 argmax mismatch");

        std::cout << "device_argmax_test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "device_argmax_test failed: " << error.what() << '\n';
        return 1;
    }
}
