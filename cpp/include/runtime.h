#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace minillm {

enum class Mode {
    Forward,
    Naive,
    FixedShape,
    BucketShape,
    KvCache,
    Scheduler,
    ContinuousBatch,
};

enum class Provider {
    Cpu,
    Cuda,
    TensorRt,
};

struct RuntimeOptions {
    std::string model_path;
    std::string decode_model_path;
    std::string model_spec_path;
    std::string input_ids_path;
    std::string output_csv_path;
    std::string timeline_csv_path;
    std::string request_summary_csv_path;
    std::string batch_csv_path;
    std::string sampling_mode{"device"};
    std::vector<std::string> request_input_paths;
    std::vector<int> request_max_new_tokens;
    std::vector<int> request_arrival_iterations;
    Mode mode{Mode::Forward};
    Provider provider{Provider::Cpu};
    int max_new_tokens{32};
    int warmup_runs{0};
    int max_seq_len{0};
    int max_active_requests{2};
    int max_tokens_per_step{1};
    int max_batch_size{8};
    int kv_block_size{16};
    int kv_num_blocks{64};
    std::vector<int> shape_buckets;
};

class Runtime {
public:
    explicit Runtime(RuntimeOptions options);

    int Run();

private:
    int RunForward();
    int RunNaiveDecode();
    int RunFixedShapeDecode();
    int RunBucketShapeDecode();
    int RunKvCacheDecode();
    int RunScheduler(bool continuous_batch = false);

    RuntimeOptions options_;
};

Mode ParseMode(const std::string& value);
Provider ParseProvider(const std::string& value);
std::vector<int> ParseShapeBuckets(const std::string& value);
std::string ToString(Mode mode);
std::string ToString(Provider provider);

} // namespace minillm
