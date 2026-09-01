#include "model_spec.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void Expect(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

template <typename Callable>
void ExpectThrows(Callable&& callable, const std::string& message)
{
    try {
        callable();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

} // namespace

int main()
{
    try {
        const auto gpt2 = minillm::ModelSpec::Gpt2();
        gpt2.Validate();
        Expect(gpt2.PastTensorName(11, false) == "past_key_values.11.value", "GPT-2 past naming");
        Expect(gpt2.PresentTensorName(0, true) == "present.0.key", "GPT-2 present naming");
        Expect(gpt2.KvTensorBytes(512) == 12ULL * 512 * 64 * 4, "GPT-2 tensor bytes");
        Expect(gpt2.KvCacheBytes(512) == 12ULL * 2 * 2 * 12 * 512 * 64 * 4,
            "GPT-2 ping-pong bytes");
        Expect(gpt2.IsEos(50256) && !gpt2.IsEos(42), "GPT-2 EOS detection");

        const std::string path = "/tmp/minillm_model_spec_test.spec";
        {
            std::ofstream out(path);
            out << "# Qwen test manifest\n"
                << "model_name=Qwen2.5-0.5B-Instruct\n"
                << "num_hidden_layers=24\nnum_attention_heads=14\nnum_kv_heads=2\n"
                << "head_dim=64\nvocab_size=151936\n"
                << "model_max_position_embeddings=32768\nengine_max_sequence_length=512\n"
                << "input_ids_dtype=int64\nattention_mask_dtype=int64\nposition_ids_dtype=int64\n"
                << "logits_dtype=float16\nkv_dtype=float16\nrequires_position_ids=true\n"
                << "eos_token_ids=151645, 151643\npad_token_id=151643\n"
                << "past_prefix=past_key_values.\npresent_prefix=present.\nkv_layout=bhsd\n";
        }
        const auto qwen = minillm::ModelSpec::Load(path);
        std::remove(path.c_str());
        Expect(qwen.num_hidden_layers == 24 && qwen.num_kv_heads == 2, "Qwen structure parsing");
        Expect(qwen.requires_position_ids, "Qwen position IDs");
        Expect(qwen.logits_dtype == minillm::TensorDataType::Float16, "Qwen logits dtype");
        Expect(qwen.KvTensorBytes(512) == 2ULL * 512 * 64 * 2, "Qwen GQA tensor bytes");
        Expect(qwen.KvCacheBytes(512) == 24ULL * 2 * 2 * 2 * 512 * 64 * 2,
            "Qwen ping-pong bytes");
        Expect(qwen.IsEos(151645) && qwen.IsEos(151643), "Qwen multi-EOS detection");

        auto bad = qwen;
        bad.engine_max_sequence_length = 65536;
        ExpectThrows([&]() { bad.Validate(); }, "native/engine limit validation");
        bad = qwen;
        bad.num_kv_heads = 3;
        ExpectThrows([&]() { bad.Validate(); }, "GQA divisibility validation");
        ExpectThrows([&]() { (void) minillm::ParseTensorDataType("fp16"); }, "strict dtype parsing");

        std::cout << "model_spec_test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "model_spec_test failed: " << error.what() << '\n';
        return 1;
    }
}
