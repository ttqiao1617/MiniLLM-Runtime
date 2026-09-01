#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace minillm {

struct StepMetric {
    int step{0};
    double latency_ms{0.0};
    std::int64_t next_token{-1};
    double step_total_ms{0.0};
    double backend_total_ms{0.0};
    double input_prepare_ms{0.0};
    double backend_shape_ms{0.0};
    double backend_buffer_bind_ms{0.0};
    double h2d_ms{0.0};
    double enqueue_ms{0.0};
    double device_sample_ms{0.0};
    double d2h_ms{0.0};
    double sync_ms{0.0};
    double kv_gather_ms{0.0};
    double kv_scatter_ms{0.0};
    double logits_postprocess_ms{0.0};
    double sampler_ms{0.0};
    double kv_init_ms{0.0};
    std::size_t h2d_bytes{0};
    std::size_t d2h_bytes{0};
    std::size_t bound_output_bytes{0};
    std::size_t kv_gather_bytes{0};
    std::size_t kv_scatter_bytes{0};
    std::size_t kv_staging_zero_bytes{0};
    std::size_t padding_tokens{0};
    std::size_t batch_scratch_bytes{0};
    std::size_t batch_scratch_allocated_bytes{0};
    std::int64_t max_past_length{0};
    int batch_size{1};
    int enqueue_count{0};
    int buffer_allocation_count{0};
    std::size_t managed_kv_bytes{0};
    int output_tensors{0};
    bool cuda_event_timing{false};
};

class ScopedTimer {
public:
    explicit ScopedTimer(double* elapsed_ms);
    ~ScopedTimer();

private:
    double* elapsed_ms_;
    std::chrono::steady_clock::time_point start_;
};

void WriteStepCsv(const std::string& path,
                  const std::string& model,
                  const std::string& provider,
                  const std::string& mode,
                  int prompt_len,
                  int output_len,
                  const std::vector<StepMetric>& metrics);

} // namespace minillm
