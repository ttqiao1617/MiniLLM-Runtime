#include "kv_staging_kernels.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
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

std::uint16_t PrefixValue(int batch, int tensor, int head, int sequence, int dim)
{
    return static_cast<std::uint16_t>(
        1000 * batch + 300 * tensor + 100 * head + 10 * sequence + dim);
}

std::uint16_t NewValue(int batch, int tensor, int head, int dim)
{
    return static_cast<std::uint16_t>(40000 + 1000 * batch + 300 * tensor + 100 * head + dim);
}

} // namespace

int main()
{
    try {
        int device_count = 0;
        const auto count_status = cudaGetDeviceCount(&device_count);
        if (count_status == cudaErrorNoDevice || device_count == 0) {
            std::cout << "kv_staging_kernels_test skipped: no CUDA device\n";
            return 0;
        }
        CheckCuda(count_status, "cudaGetDeviceCount");

        constexpr int kTensorCount = 2;
        constexpr int kBatch = 2;
        constexpr int kHeads = 2;
        constexpr int kHeadDim = 8;
        constexpr int kMaxPast = 3;
        constexpr int kElementSize = sizeof(std::uint16_t);
        constexpr std::uint16_t kSentinel = 0xEEEE;
        const std::vector<std::int64_t> capacities{5, 4};
        const std::vector<std::int64_t> past_lengths{2, 3};

        std::vector<void*> host_device_slabs(kBatch, nullptr);
        std::vector<std::vector<std::uint16_t>> host_slabs;
        host_slabs.reserve(kBatch);
        for (int batch = 0; batch < kBatch; ++batch) {
            const auto elements = static_cast<std::size_t>(kTensorCount) * kHeads
                * static_cast<std::size_t>(capacities[batch]) * kHeadDim;
            host_slabs.emplace_back(elements, kSentinel);
            auto& slab = host_slabs.back();
            for (int tensor = 0; tensor < kTensorCount; ++tensor) {
                for (int head = 0; head < kHeads; ++head) {
                    for (int sequence = 0; sequence < past_lengths[batch]; ++sequence) {
                        for (int dim = 0; dim < kHeadDim; ++dim) {
                            const auto offset = (((static_cast<std::size_t>(tensor) * kHeads + head)
                                * capacities[batch] + sequence) * kHeadDim) + dim;
                            slab[offset] = PrefixValue(batch, tensor, head, sequence, dim);
                        }
                    }
                }
            }
            CheckCuda(cudaMalloc(&host_device_slabs[batch], elements * kElementSize),
                "cudaMalloc(cache slab)");
            CheckCuda(cudaMemcpy(host_device_slabs[batch], slab.data(), elements * kElementSize,
                cudaMemcpyHostToDevice), "cudaMemcpy(cache slab)");
        }

        void** device_slabs = nullptr;
        std::int64_t* device_past = nullptr;
        std::int64_t* device_capacities = nullptr;
        CheckCuda(cudaMalloc(&device_slabs, kBatch * sizeof(void*)), "cudaMalloc(slab pointers)");
        CheckCuda(cudaMalloc(&device_past, kBatch * sizeof(std::int64_t)), "cudaMalloc(past)");
        CheckCuda(cudaMalloc(&device_capacities, kBatch * sizeof(std::int64_t)),
            "cudaMalloc(capacities)");
        CheckCuda(cudaMemcpy(device_slabs, host_device_slabs.data(), kBatch * sizeof(void*),
            cudaMemcpyHostToDevice), "cudaMemcpy(slab pointers)");
        CheckCuda(cudaMemcpy(device_past, past_lengths.data(), kBatch * sizeof(std::int64_t),
            cudaMemcpyHostToDevice), "cudaMemcpy(past)");
        CheckCuda(cudaMemcpy(device_capacities, capacities.data(), kBatch * sizeof(std::int64_t),
            cudaMemcpyHostToDevice), "cudaMemcpy(capacities)");

        const auto packed_past_tensor_elements = static_cast<std::size_t>(kBatch) * kHeads
            * kMaxPast * kHeadDim;
        const auto packed_past_tensor_bytes = packed_past_tensor_elements * kElementSize;
        std::uint16_t* device_packed_past = nullptr;
        CheckCuda(cudaMalloc(&device_packed_past,
            kTensorCount * packed_past_tensor_bytes), "cudaMalloc(packed past)");
        minillm::LaunchGatherKvStaging(
            reinterpret_cast<const void* const*>(device_slabs), device_past,
            device_capacities, device_packed_past, kTensorCount, kBatch, kHeads,
            kHeadDim, kMaxPast, kElementSize, packed_past_tensor_bytes, nullptr);
        std::vector<std::uint16_t> packed_past(
            static_cast<std::size_t>(kTensorCount) * packed_past_tensor_elements);
        CheckCuda(cudaMemcpy(packed_past.data(), device_packed_past,
            packed_past.size() * kElementSize, cudaMemcpyDeviceToHost),
            "cudaMemcpy(packed past)");
        for (int tensor = 0; tensor < kTensorCount; ++tensor) {
            for (int batch = 0; batch < kBatch; ++batch) {
                for (int head = 0; head < kHeads; ++head) {
                    for (int sequence = 0; sequence < kMaxPast; ++sequence) {
                        for (int dim = 0; dim < kHeadDim; ++dim) {
                            const auto offset = ((((static_cast<std::size_t>(tensor) * kBatch + batch)
                                * kHeads + head) * kMaxPast + sequence) * kHeadDim) + dim;
                            const auto expected = sequence < past_lengths[batch]
                                ? PrefixValue(batch, tensor, head, sequence, dim) : 0;
                            Expect(packed_past[offset] == expected,
                                "fused gather value/padding mismatch");
                        }
                    }
                }
            }
        }

        const auto packed_present_tensor_elements = static_cast<std::size_t>(kBatch) * kHeads
            * (kMaxPast + 1) * kHeadDim;
        const auto packed_present_tensor_bytes = packed_present_tensor_elements * kElementSize;
        std::vector<std::uint16_t> packed_present(
            static_cast<std::size_t>(kTensorCount) * packed_present_tensor_elements, 0);
        for (int tensor = 0; tensor < kTensorCount; ++tensor) {
            for (int batch = 0; batch < kBatch; ++batch) {
                for (int head = 0; head < kHeads; ++head) {
                    for (int dim = 0; dim < kHeadDim; ++dim) {
                        const auto offset = ((((static_cast<std::size_t>(tensor) * kBatch + batch)
                            * kHeads + head) * (kMaxPast + 1) + kMaxPast) * kHeadDim) + dim;
                        packed_present[offset] = NewValue(batch, tensor, head, dim);
                    }
                }
            }
        }
        std::uint16_t* device_packed_present = nullptr;
        CheckCuda(cudaMalloc(&device_packed_present,
            kTensorCount * packed_present_tensor_bytes), "cudaMalloc(packed present)");
        CheckCuda(cudaMemcpy(device_packed_present, packed_present.data(),
            packed_present.size() * kElementSize, cudaMemcpyHostToDevice),
            "cudaMemcpy(packed present)");
        minillm::LaunchScatterNewKv(device_packed_present, device_slabs, device_past,
            device_capacities, kTensorCount, kBatch, kHeads, kHeadDim, kMaxPast,
            kElementSize, packed_present_tensor_bytes, nullptr);
        CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        for (int batch = 0; batch < kBatch; ++batch) {
            auto actual = host_slabs[batch];
            CheckCuda(cudaMemcpy(actual.data(), host_device_slabs[batch],
                actual.size() * kElementSize, cudaMemcpyDeviceToHost),
                "cudaMemcpy(scattered cache)");
            for (int tensor = 0; tensor < kTensorCount; ++tensor) {
                for (int head = 0; head < kHeads; ++head) {
                    for (int sequence = 0; sequence < capacities[batch]; ++sequence) {
                        for (int dim = 0; dim < kHeadDim; ++dim) {
                            const auto offset = (((static_cast<std::size_t>(tensor) * kHeads + head)
                                * capacities[batch] + sequence) * kHeadDim) + dim;
                            std::uint16_t expected = kSentinel;
                            if (sequence < past_lengths[batch]) {
                                expected = PrefixValue(batch, tensor, head, sequence, dim);
                            } else if (sequence == past_lengths[batch]) {
                                expected = NewValue(batch, tensor, head, dim);
                            }
                            Expect(actual[offset] == expected,
                                "fused scatter changed prefix or missed new KV");
                        }
                    }
                }
            }
        }

        CheckCuda(cudaFree(device_packed_present), "cudaFree(packed present)");
        CheckCuda(cudaFree(device_packed_past), "cudaFree(packed past)");
        CheckCuda(cudaFree(device_capacities), "cudaFree(capacities)");
        CheckCuda(cudaFree(device_past), "cudaFree(past)");
        CheckCuda(cudaFree(device_slabs), "cudaFree(slab pointers)");
        for (auto* slab : host_device_slabs) CheckCuda(cudaFree(slab), "cudaFree(cache slab)");

        std::cout << "kv_staging_kernels_test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "kv_staging_kernels_test failed: " << error.what() << '\n';
        return 1;
    }
}
