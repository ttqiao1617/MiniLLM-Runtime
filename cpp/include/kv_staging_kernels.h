#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace minillm {

// Packs fixed-stride per-request KV slabs into TensorRT's dense
// [B, H, max_past, D] tensor layout and writes zero for logical padding.
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
                           cudaStream_t stream);

// Appends only the new-token row from TensorRT's dense present tensors back to
// the fixed-stride request slabs. Existing prefix rows remain in place.
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
                        cudaStream_t stream);

} // namespace minillm
