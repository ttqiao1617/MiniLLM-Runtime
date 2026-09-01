#include "profiler.h"

#include <fstream>
#include <stdexcept>

namespace minillm {

ScopedTimer::ScopedTimer(double* elapsed_ms)
    : elapsed_ms_(elapsed_ms)
    , start_(std::chrono::steady_clock::now())
{
}

ScopedTimer::~ScopedTimer()
{
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration<double, std::milli>(end - start_).count();
    if (elapsed_ms_ != nullptr) {
        *elapsed_ms_ = elapsed;
    }
}

void WriteStepCsv(const std::string& path,
                  const std::string& model,
                  const std::string& provider,
                  const std::string& mode,
                  int prompt_len,
                  int output_len,
                  const std::vector<StepMetric>& metrics)
{
    if (path.empty()) {
        return;
    }

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open CSV for writing: " + path);
    }

    out << "model,provider,mode,prompt_len,output_len,step,latency_ms,next_token,"
        << "step_total_ms,backend_total_ms,input_prepare_ms,backend_shape_ms,"
        << "backend_buffer_bind_ms,h2d_ms,enqueue_ms,device_sample_ms,d2h_ms,sync_ms,"
        << "kv_gather_ms,kv_scatter_ms,logits_postprocess_ms,sampler_ms,kv_init_ms,"
        << "h2d_bytes,d2h_bytes,bound_output_bytes,kv_gather_bytes,kv_scatter_bytes,"
        << "kv_staging_zero_bytes,"
        << "padding_tokens,batch_scratch_bytes,batch_scratch_allocated_bytes,"
        << "max_past_length,batch_size,enqueue_count,buffer_allocation_count,"
        << "managed_kv_bytes,output_tensors,cuda_event_timing\n";
    for (const auto& metric : metrics) {
        out << model << ','
            << provider << ','
            << mode << ','
            << prompt_len << ','
            << output_len << ','
            << metric.step << ','
            << metric.latency_ms << ','
            << metric.next_token << ','
            << metric.step_total_ms << ','
            << metric.backend_total_ms << ','
            << metric.input_prepare_ms << ','
            << metric.backend_shape_ms << ','
            << metric.backend_buffer_bind_ms << ','
            << metric.h2d_ms << ','
            << metric.enqueue_ms << ','
            << metric.device_sample_ms << ','
            << metric.d2h_ms << ','
            << metric.sync_ms << ','
            << metric.kv_gather_ms << ','
            << metric.kv_scatter_ms << ','
            << metric.logits_postprocess_ms << ','
            << metric.sampler_ms << ','
            << metric.kv_init_ms << ','
            << metric.h2d_bytes << ','
            << metric.d2h_bytes << ','
            << metric.bound_output_bytes << ','
            << metric.kv_gather_bytes << ','
            << metric.kv_scatter_bytes << ','
            << metric.kv_staging_zero_bytes << ','
            << metric.padding_tokens << ','
            << metric.batch_scratch_bytes << ','
            << metric.batch_scratch_allocated_bytes << ','
            << metric.max_past_length << ','
            << metric.batch_size << ','
            << metric.enqueue_count << ','
            << metric.buffer_allocation_count << ','
            << metric.managed_kv_bytes << ','
            << metric.output_tensors << ','
            << (metric.cuda_event_timing ? 1 : 0) << '\n';
    }
}

} // namespace minillm
