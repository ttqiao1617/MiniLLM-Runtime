#include "kv_staging_kernels.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace minillm {
namespace {

constexpr int kThreads = 256;
constexpr int kVectorBytes = 16;

__global__ void GatherKvKernel(const std::uint8_t* const* cache_slabs,
                               const std::int64_t* past_lengths,
                               const std::int64_t* capacities,
                               std::uint8_t* packed_past,
                               int tensor_count,
                               int batch_size,
                               int num_kv_heads,
                               int max_past_length,
                               int chunks_per_token,
                               std::size_t row_bytes,
                               std::size_t packed_tensor_stride_bytes,
                               std::size_t total_chunks)
{
    for (std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x
             + threadIdx.x;
         linear < total_chunks;
         linear += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        auto index = linear;
        const int chunk = static_cast<int>(index % chunks_per_token);
        index /= chunks_per_token;
        const int sequence = static_cast<int>(index % max_past_length);
        index /= max_past_length;
        const int head = static_cast<int>(index % num_kv_heads);
        index /= num_kv_heads;
        const int batch = static_cast<int>(index % batch_size);
        const int tensor = static_cast<int>(index / batch_size);

        auto* destination = packed_past
            + static_cast<std::size_t>(tensor) * packed_tensor_stride_bytes
            + (((static_cast<std::size_t>(batch) * num_kv_heads + head)
                    * max_past_length + sequence) * row_bytes)
            + static_cast<std::size_t>(chunk) * kVectorBytes;
        auto* destination_vector = reinterpret_cast<uint4*>(destination);
        if (sequence >= past_lengths[batch]) {
            *destination_vector = make_uint4(0, 0, 0, 0);
            continue;
        }

        const auto capacity = static_cast<std::size_t>(capacities[batch]);
        const auto tensor_stride = static_cast<std::size_t>(num_kv_heads)
            * capacity * row_bytes;
        const auto* source = cache_slabs[batch]
            + static_cast<std::size_t>(tensor) * tensor_stride
            + (static_cast<std::size_t>(head) * capacity + sequence) * row_bytes
            + static_cast<std::size_t>(chunk) * kVectorBytes;
        *destination_vector = *reinterpret_cast<const uint4*>(source);
    }
}

__global__ void ScatterNewKvKernel(const std::uint8_t* packed_present,
                                   std::uint8_t* const* cache_slabs,
                                   const std::int64_t* past_lengths,
                                   const std::int64_t* capacities,
                                   int tensor_count,
                                   int batch_size,
                                   int num_kv_heads,
                                   int max_past_length,
                                   int chunks_per_token,
                                   std::size_t row_bytes,
                                   std::size_t packed_tensor_stride_bytes,
                                   std::size_t total_chunks)
{
    for (std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x
             + threadIdx.x;
         linear < total_chunks;
         linear += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        auto index = linear;
        const int chunk = static_cast<int>(index % chunks_per_token);
        index /= chunks_per_token;
        const int head = static_cast<int>(index % num_kv_heads);
        index /= num_kv_heads;
        const int batch = static_cast<int>(index % batch_size);
        const int tensor = static_cast<int>(index / batch_size);

        const auto capacity = static_cast<std::size_t>(capacities[batch]);
        const auto destination_sequence = static_cast<std::size_t>(past_lengths[batch]);
        const auto tensor_stride = static_cast<std::size_t>(num_kv_heads)
            * capacity * row_bytes;
        auto* destination = cache_slabs[batch]
            + static_cast<std::size_t>(tensor) * tensor_stride
            + (static_cast<std::size_t>(head) * capacity + destination_sequence) * row_bytes
            + static_cast<std::size_t>(chunk) * kVectorBytes;
        const auto* source = packed_present
            + static_cast<std::size_t>(tensor) * packed_tensor_stride_bytes
            + (((static_cast<std::size_t>(batch) * num_kv_heads + head)
                    * (max_past_length + 1) + max_past_length) * row_bytes)
            + static_cast<std::size_t>(chunk) * kVectorBytes;
        *reinterpret_cast<uint4*>(destination) = *reinterpret_cast<const uint4*>(source);
    }
}

void ValidateArguments(const void* cache_slabs,
                       const std::int64_t* past_lengths,
                       const std::int64_t* capacities,
                       const void* packed,
                       int tensor_count,
                       int batch_size,
                       int num_kv_heads,
                       int head_dim,
                       int max_past_length,
                       int element_size,
                       std::size_t packed_tensor_stride_bytes)
{
    if (cache_slabs == nullptr || past_lengths == nullptr || capacities == nullptr
        || packed == nullptr || tensor_count <= 0 || batch_size <= 0
        || num_kv_heads <= 0 || head_dim <= 0 || max_past_length <= 0
        || (element_size != 2 && element_size != 4)
        || (static_cast<std::size_t>(head_dim) * element_size) % kVectorBytes != 0
        || packed_tensor_stride_bytes == 0) {
        throw std::invalid_argument("invalid fused KV staging arguments");
    }
}

void CheckLaunch(const char* label)
{
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(status));
    }
}

} // namespace

void LaunchGatherKvStaging(const void* const* cache_slabs,
                           const std::int64_t* past_lengths,
                           const std::int64_t* capacities,
                           void* packed_past,
                           int tensor_count,
                           int batch_size,
                           int num_kv_heads,
                           int head_dim,
                           int max_past_length,
                           int element_size,
                           std::size_t packed_tensor_stride_bytes,
                           cudaStream_t stream)
{
    ValidateArguments(cache_slabs, past_lengths, capacities, packed_past,
        tensor_count, batch_size, num_kv_heads, head_dim, max_past_length,
        element_size, packed_tensor_stride_bytes);
    const auto row_bytes = static_cast<std::size_t>(head_dim) * element_size;
    const int chunks_per_token = static_cast<int>(row_bytes / kVectorBytes);
    const auto total_chunks = static_cast<std::size_t>(tensor_count) * batch_size
        * num_kv_heads * max_past_length * chunks_per_token;
    const auto blocks = static_cast<unsigned int>((total_chunks + kThreads - 1) / kThreads);
    GatherKvKernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const std::uint8_t* const*>(cache_slabs),
        past_lengths, capacities, static_cast<std::uint8_t*>(packed_past),
        tensor_count, batch_size, num_kv_heads, max_past_length,
        chunks_per_token, row_bytes, packed_tensor_stride_bytes, total_chunks);
    CheckLaunch("fused KV gather launch failed");
}

void LaunchScatterNewKv(const void* packed_present,
                        void* const* cache_slabs,
                        const std::int64_t* past_lengths,
                        const std::int64_t* capacities,
                        int tensor_count,
                        int batch_size,
                        int num_kv_heads,
                        int head_dim,
                        int max_past_length,
                        int element_size,
                        std::size_t packed_tensor_stride_bytes,
                        cudaStream_t stream)
{
    ValidateArguments(cache_slabs, past_lengths, capacities, packed_present,
        tensor_count, batch_size, num_kv_heads, head_dim, max_past_length,
        element_size, packed_tensor_stride_bytes);
    const auto row_bytes = static_cast<std::size_t>(head_dim) * element_size;
    const int chunks_per_token = static_cast<int>(row_bytes / kVectorBytes);
    const auto total_chunks = static_cast<std::size_t>(tensor_count) * batch_size
        * num_kv_heads * chunks_per_token;
    const auto blocks = static_cast<unsigned int>((total_chunks + kThreads - 1) / kThreads);
    ScatterNewKvKernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const std::uint8_t*>(packed_present),
        reinterpret_cast<std::uint8_t* const*>(cache_slabs),
        past_lengths, capacities, tensor_count, batch_size, num_kv_heads,
        max_past_length, chunks_per_token, row_bytes,
        packed_tensor_stride_bytes, total_chunks);
    CheckLaunch("fused new-KV scatter launch failed");
}

} // namespace minillm
