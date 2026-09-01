#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace minillm {

enum class TensorDataType {
    Int64,
    Int32,
    Float32,
    Float16,
    BFloat16,
};

enum class KvLayout {
    Bhsd,
};

struct ModelSpec {
    std::string model_name;
    int num_hidden_layers{0};
    int num_attention_heads{0};
    int num_kv_heads{0};
    int head_dim{0};
    int vocab_size{0};
    int model_max_position_embeddings{0};
    int engine_max_sequence_length{0};
    TensorDataType input_ids_dtype{TensorDataType::Int64};
    TensorDataType attention_mask_dtype{TensorDataType::Int64};
    TensorDataType position_ids_dtype{TensorDataType::Int64};
    TensorDataType logits_dtype{TensorDataType::Float32};
    TensorDataType kv_dtype{TensorDataType::Float32};
    bool requires_position_ids{false};
    std::vector<std::int64_t> eos_token_ids;
    std::int64_t pad_token_id{0};
    std::string past_prefix;
    std::string present_prefix;
    KvLayout kv_layout{KvLayout::Bhsd};

    static ModelSpec Gpt2();
    static ModelSpec Load(const std::string& path);

    void Validate() const;
    bool IsEos(std::int64_t token_id) const;
    std::string PastTensorName(int layer, bool key) const;
    std::string PresentTensorName(int layer, bool key) const;
    std::size_t KvTensorBytes(std::int64_t sequence_length) const;
    std::size_t KvCacheBytes(std::int64_t capacity, int copies = 2) const;
};

TensorDataType ParseTensorDataType(const std::string& value);
std::string ToString(TensorDataType dtype);
std::size_t ElementSize(TensorDataType dtype);
KvLayout ParseKvLayout(const std::string& value);
std::string ToString(KvLayout layout);

} // namespace minillm
