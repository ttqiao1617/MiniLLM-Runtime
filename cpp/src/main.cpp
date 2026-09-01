#include "runtime.h"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void PrintHelp(const char* argv0)
{
    std::cout
        << "Usage: " << argv0 << " [options]\n\n"
        << "Options:\n"
        << "  --model PATH             ONNX model path\n"
        << "  --decode-model PATH      TensorRT with-past engine for kv_cache mode\n"
        << "  --model-spec PATH        key=value model/engine contract (default: built-in GPT-2)\n"
        << "  --input-ids PATH         Text file containing token IDs\n"
        << "  --request-inputs LIST    Comma-separated token ID files for scheduler mode\n"
        << "  --request-max-new-tokens LIST  Optional per-request output limits\n"
        << "  --request-arrival-iterations LIST  Optional nondecreasing scheduler arrivals\n"
        << "  --max-new-tokens N       Number of tokens to generate, default 32\n"
        << "  --warmup-runs N          Unmeasured backend forwards before decode, default 0\n"
        << "  --max-seq-len N          Fixed sequence length for fixed_shape mode\n"
        << "  --shape-buckets LIST     Comma-separated bucket lengths for bucket_shape mode\n"
        << "  --mode MODE              forward | naive | fixed_shape | bucket_shape | kv_cache | scheduler | continuous_batch\n"
        << "  --provider PROVIDER      cpu | tensorrt (cuda is reserved, not implemented)\n"
        << "  --sampling MODE          host | device, default device\n"
        << "  --output-csv PATH        Optional per-step metric CSV\n"
        << "  --timeline-csv PATH      Scheduler event timeline CSV\n"
        << "  --request-summary-csv P  Scheduler per-request summary CSV\n"
        << "  --batch-csv PATH         Continuous-batch per-enqueue evidence CSV\n"
        << "  --max-active-requests N  Scheduler active request limit, default 2\n"
        << "  --max-tokens-per-step N  Decode budget per scheduler iteration, default 1\n"
        << "  --max-batch-size N       Continuous decode batch cap, default 8\n"
        << "  --kv-block-size N        Logical KV block size in tokens, default 16\n"
        << "  --kv-num-blocks N        Logical KV block pool size, default 64\n"
        << "  --help                   Show this help\n";
}

int ParseInteger(const std::string& value, const std::string& option)
{
    std::size_t consumed = 0;
    int parsed = 0;
    try {
        parsed = std::stoi(value, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid integer for " + option + ": " + value);
    }
    if (consumed != value.size()) {
        throw std::runtime_error("Invalid integer for " + option + ": " + value);
    }
    return parsed;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        minillm::RuntimeOptions options;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto require_value = [&](const std::string& name) -> std::string {
                if (i + 1 >= argc) {
                    throw std::runtime_error("Missing value for " + name);
                }
                return argv[++i];
            };

            if (arg == "--help" || arg == "-h") {
                PrintHelp(argv[0]);
                return 0;
            } else if (arg == "--model") {
                options.model_path = require_value(arg);
            } else if (arg == "--decode-model") {
                options.decode_model_path = require_value(arg);
            } else if (arg == "--model-spec") {
                options.model_spec_path = require_value(arg);
            } else if (arg == "--input-ids") {
                options.input_ids_path = require_value(arg);
            } else if (arg == "--request-inputs") {
                std::string value = require_value(arg);
                std::size_t begin = 0;
                while (begin <= value.size()) {
                    const auto end = value.find(',', begin);
                    const auto item = value.substr(begin, end == std::string::npos ? end : end - begin);
                    if (item.empty()) throw std::runtime_error("empty path in --request-inputs");
                    options.request_input_paths.push_back(item);
                    if (end == std::string::npos) break;
                    begin = end + 1;
                }
            } else if (arg == "--request-max-new-tokens") {
                std::string value = require_value(arg);
                std::size_t begin = 0;
                while (begin <= value.size()) {
                    const auto end = value.find(',', begin);
                    const auto item = value.substr(begin, end == std::string::npos ? end : end - begin);
                    if (item.empty()) throw std::runtime_error("empty value in --request-max-new-tokens");
                    const int limit = ParseInteger(item, arg);
                    if (limit <= 0) throw std::runtime_error("request output limits must be positive");
                    options.request_max_new_tokens.push_back(limit);
                    if (end == std::string::npos) break;
                    begin = end + 1;
                }
            } else if (arg == "--request-arrival-iterations") {
                std::string value = require_value(arg);
                std::size_t begin = 0;
                while (begin <= value.size()) {
                    const auto end = value.find(',', begin);
                    const auto item = value.substr(begin, end == std::string::npos ? end : end - begin);
                    if (item.empty()) throw std::runtime_error("empty value in --request-arrival-iterations");
                    const int iteration = ParseInteger(item, arg);
                    if (iteration < 0) throw std::runtime_error("request arrivals must be nonnegative");
                    options.request_arrival_iterations.push_back(iteration);
                    if (end == std::string::npos) break;
                    begin = end + 1;
                }
            } else if (arg == "--max-new-tokens") {
                options.max_new_tokens = ParseInteger(require_value(arg), arg);
            } else if (arg == "--warmup-runs") {
                options.warmup_runs = ParseInteger(require_value(arg), arg);
            } else if (arg == "--max-seq-len") {
                options.max_seq_len = ParseInteger(require_value(arg), arg);
            } else if (arg == "--shape-buckets") {
                options.shape_buckets = minillm::ParseShapeBuckets(require_value(arg));
            } else if (arg == "--mode") {
                options.mode = minillm::ParseMode(require_value(arg));
            } else if (arg == "--provider") {
                options.provider = minillm::ParseProvider(require_value(arg));
            } else if (arg == "--sampling") {
                options.sampling_mode = require_value(arg);
            } else if (arg == "--output-csv") {
                options.output_csv_path = require_value(arg);
            } else if (arg == "--timeline-csv") {
                options.timeline_csv_path = require_value(arg);
            } else if (arg == "--request-summary-csv") {
                options.request_summary_csv_path = require_value(arg);
            } else if (arg == "--batch-csv") {
                options.batch_csv_path = require_value(arg);
            } else if (arg == "--max-active-requests") {
                options.max_active_requests = ParseInteger(require_value(arg), arg);
            } else if (arg == "--max-tokens-per-step") {
                options.max_tokens_per_step = ParseInteger(require_value(arg), arg);
            } else if (arg == "--max-batch-size") {
                options.max_batch_size = ParseInteger(require_value(arg), arg);
            } else if (arg == "--kv-block-size") {
                options.kv_block_size = ParseInteger(require_value(arg), arg);
            } else if (arg == "--kv-num-blocks") {
                options.kv_num_blocks = ParseInteger(require_value(arg), arg);
            } else {
                throw std::runtime_error("Unknown argument: " + arg);
            }
        }

        minillm::Runtime runtime(options);
        return runtime.Run();
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
