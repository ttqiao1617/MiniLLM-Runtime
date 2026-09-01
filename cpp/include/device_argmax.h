#pragma once

#include "model_spec.h"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace minillm {

// Launches one deterministic argmax reduction per row. Ties are resolved in
// favor of the lowest token ID, matching GreedySampler's host semantics.
void LaunchDeviceArgmax(const void* logits,
                        TensorDataType dtype,
                        int batch_size,
                        int vocab_size,
                        std::int32_t* output_token_ids,
                        cudaStream_t stream);

} // namespace minillm
