#include "model_spec.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace minillm {
namespace {

std::string Trim(std::string value)
{
    const auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

int ParsePositiveInt(const std::unordered_map<std::string, std::string>& values,
                     const std::string& key)
{
    const auto it = values.find(key);
    if (it == values.end()) throw std::runtime_error("ModelSpec missing required key: " + key);
    std::size_t consumed = 0;
    int value = 0;
    try {
        value = std::stoi(it->second, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error("ModelSpec key is not an integer: " + key);
    }
    if (consumed != it->second.size() || value <= 0) {
        throw std::runtime_error("ModelSpec key must be a positive integer: " + key);
    }
    return value;
}

std::int64_t ParseTokenId(const std::string& text, const std::string& key)
{
    std::size_t consumed = 0;
    std::int64_t value = 0;
    try {
        value = std::stoll(text, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error("ModelSpec key is not an integer: " + key);
    }
    if (consumed != text.size()) throw std::runtime_error("ModelSpec key is not an integer: " + key);
    return value;
}

const std::string& Required(const std::unordered_map<std::string, std::string>& values,
                            const std::string& key)
{
    const auto it = values.find(key);
    if (it == values.end()) throw std::runtime_error("ModelSpec missing required key: " + key);
    return it->second;
}

std::size_t CheckedMultiply(std::size_t lhs, std::size_t rhs, const std::string& label)
{
    if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs) {
        throw std::overflow_error("ModelSpec size overflow while computing " + label);
    }
    return lhs * rhs;
}

} // namespace

TensorDataType ParseTensorDataType(const std::string& value)
{
    if (value == "int64") return TensorDataType::Int64;
    if (value == "int32") return TensorDataType::Int32;
    if (value == "float32") return TensorDataType::Float32;
    if (value == "float16") return TensorDataType::Float16;
    if (value == "bfloat16") return TensorDataType::BFloat16;
    throw std::runtime_error("Unsupported ModelSpec dtype: " + value);
}

std::string ToString(TensorDataType dtype)
{
    switch (dtype) {
    case TensorDataType::Int64: return "int64";
    case TensorDataType::Int32: return "int32";
    case TensorDataType::Float32: return "float32";
    case TensorDataType::Float16: return "float16";
    case TensorDataType::BFloat16: return "bfloat16";
    }
    throw std::runtime_error("Unhandled ModelSpec dtype");
}

std::size_t ElementSize(TensorDataType dtype)
{
    switch (dtype) {
    case TensorDataType::Int64: return 8;
    case TensorDataType::Int32: return 4;
    case TensorDataType::Float32: return 4;
    case TensorDataType::Float16:
    case TensorDataType::BFloat16: return 2;
    }
    throw std::runtime_error("Unhandled ModelSpec dtype");
}

KvLayout ParseKvLayout(const std::string& value)
{
    if (value == "bhsd") return KvLayout::Bhsd;
    throw std::runtime_error("Unsupported ModelSpec KV layout: " + value);
}

std::string ToString(KvLayout layout)
{
    if (layout == KvLayout::Bhsd) return "bhsd";
    throw std::runtime_error("Unhandled ModelSpec KV layout");
}

ModelSpec ModelSpec::Gpt2()
{
    ModelSpec spec;
    spec.model_name = "gpt2";
    spec.num_hidden_layers = 12;
    spec.num_attention_heads = 12;
    spec.num_kv_heads = 12;
    spec.head_dim = 64;
    spec.vocab_size = 50257;
    spec.model_max_position_embeddings = 1024;
    spec.engine_max_sequence_length = 512;
    spec.input_ids_dtype = TensorDataType::Int64;
    spec.attention_mask_dtype = TensorDataType::Int64;
    spec.position_ids_dtype = TensorDataType::Int64;
    spec.logits_dtype = TensorDataType::Float32;
    spec.kv_dtype = TensorDataType::Float32;
    spec.requires_position_ids = false;
    spec.eos_token_ids = {50256};
    spec.pad_token_id = 50256;
    spec.past_prefix = "past_key_values.";
    spec.present_prefix = "present.";
    spec.kv_layout = KvLayout::Bhsd;
    return spec;
}

ModelSpec ModelSpec::Load(const std::string& path)
{
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Failed to open ModelSpec: " + path);

    static const std::unordered_set<std::string> allowed{
        "model_name", "num_hidden_layers", "num_attention_heads", "num_kv_heads",
        "head_dim", "vocab_size", "model_max_position_embeddings",
        "engine_max_sequence_length", "input_ids_dtype", "attention_mask_dtype",
        "position_ids_dtype", "logits_dtype", "kv_dtype", "requires_position_ids",
        "eos_token_ids", "pad_token_id", "past_prefix", "present_prefix", "kv_layout",
    };
    std::unordered_map<std::string, std::string> values;
    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const auto comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        line = Trim(std::move(line));
        if (line.empty()) continue;
        const auto separator = line.find('=');
        if (separator == std::string::npos) {
            throw std::runtime_error("Invalid ModelSpec line " + std::to_string(line_number) + ": expected key=value");
        }
        auto key = Trim(line.substr(0, separator));
        auto value = Trim(line.substr(separator + 1));
        if (key.empty() || value.empty()) {
            throw std::runtime_error("Invalid ModelSpec line " + std::to_string(line_number) + ": empty key/value");
        }
        if (allowed.count(key) == 0) throw std::runtime_error("Unknown ModelSpec key: " + key);
        if (!values.emplace(key, value).second) throw std::runtime_error("Duplicate ModelSpec key: " + key);
    }

    ModelSpec spec;
    spec.model_name = Required(values, "model_name");
    spec.num_hidden_layers = ParsePositiveInt(values, "num_hidden_layers");
    spec.num_attention_heads = ParsePositiveInt(values, "num_attention_heads");
    spec.num_kv_heads = ParsePositiveInt(values, "num_kv_heads");
    spec.head_dim = ParsePositiveInt(values, "head_dim");
    spec.vocab_size = ParsePositiveInt(values, "vocab_size");
    spec.model_max_position_embeddings = ParsePositiveInt(values, "model_max_position_embeddings");
    spec.engine_max_sequence_length = ParsePositiveInt(values, "engine_max_sequence_length");
    spec.input_ids_dtype = ParseTensorDataType(Required(values, "input_ids_dtype"));
    spec.attention_mask_dtype = ParseTensorDataType(Required(values, "attention_mask_dtype"));
    spec.position_ids_dtype = ParseTensorDataType(Required(values, "position_ids_dtype"));
    spec.logits_dtype = ParseTensorDataType(Required(values, "logits_dtype"));
    spec.kv_dtype = ParseTensorDataType(Required(values, "kv_dtype"));
    const auto requires_positions = Required(values, "requires_position_ids");
    if (requires_positions == "true") spec.requires_position_ids = true;
    else if (requires_positions == "false") spec.requires_position_ids = false;
    else throw std::runtime_error("ModelSpec requires_position_ids must be true or false");

    const auto eos_values = Required(values, "eos_token_ids");
    std::size_t begin = 0;
    while (begin <= eos_values.size()) {
        const auto end = eos_values.find(',', begin);
        const auto item = Trim(eos_values.substr(begin, end == std::string::npos ? end : end - begin));
        if (item.empty()) throw std::runtime_error("ModelSpec eos_token_ids contains an empty item");
        spec.eos_token_ids.push_back(ParseTokenId(item, "eos_token_ids"));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    spec.pad_token_id = ParseTokenId(Required(values, "pad_token_id"), "pad_token_id");
    spec.past_prefix = Required(values, "past_prefix");
    spec.present_prefix = Required(values, "present_prefix");
    spec.kv_layout = ParseKvLayout(Required(values, "kv_layout"));
    spec.Validate();
    return spec;
}

void ModelSpec::Validate() const
{
    if (model_name.empty()) throw std::runtime_error("ModelSpec model_name must not be empty");
    if (num_hidden_layers <= 0 || num_attention_heads <= 0 || num_kv_heads <= 0
        || head_dim <= 0 || vocab_size <= 0 || model_max_position_embeddings <= 0
        || engine_max_sequence_length <= 0) {
        throw std::runtime_error("ModelSpec dimensions must be positive");
    }
    if (num_attention_heads % num_kv_heads != 0) {
        throw std::runtime_error("ModelSpec num_attention_heads must be divisible by num_kv_heads");
    }
    if (engine_max_sequence_length > model_max_position_embeddings) {
        throw std::runtime_error("ModelSpec engine sequence limit exceeds the model-native limit");
    }
    if (input_ids_dtype != TensorDataType::Int64 && input_ids_dtype != TensorDataType::Int32) {
        throw std::runtime_error("ModelSpec input_ids_dtype must be int32 or int64");
    }
    if (attention_mask_dtype != TensorDataType::Int64 && attention_mask_dtype != TensorDataType::Int32) {
        throw std::runtime_error("ModelSpec attention_mask_dtype must be int32 or int64");
    }
    if (position_ids_dtype != TensorDataType::Int64 && position_ids_dtype != TensorDataType::Int32) {
        throw std::runtime_error("ModelSpec position_ids_dtype must be int32 or int64");
    }
    const auto is_float = [](TensorDataType dtype) {
        return dtype == TensorDataType::Float32 || dtype == TensorDataType::Float16
            || dtype == TensorDataType::BFloat16;
    };
    if (!is_float(logits_dtype) || !is_float(kv_dtype)) {
        throw std::runtime_error("ModelSpec logits_dtype and kv_dtype must be floating point");
    }
    if (eos_token_ids.empty()) throw std::runtime_error("ModelSpec eos_token_ids must not be empty");
    if (pad_token_id < 0 || pad_token_id >= vocab_size) {
        throw std::runtime_error("ModelSpec pad_token_id is outside vocabulary");
    }
    std::unordered_set<std::int64_t> unique_eos;
    for (const auto token : eos_token_ids) {
        if (token < 0 || token >= vocab_size) throw std::runtime_error("ModelSpec EOS token is outside vocabulary");
        if (!unique_eos.insert(token).second) throw std::runtime_error("ModelSpec contains duplicate EOS token IDs");
    }
    if (past_prefix.empty() || present_prefix.empty() || past_prefix == present_prefix) {
        throw std::runtime_error("ModelSpec past/present prefixes must be non-empty and different");
    }
    if (kv_layout != KvLayout::Bhsd) throw std::runtime_error("ModelSpec only supports bhsd KV layout");
}

bool ModelSpec::IsEos(std::int64_t token_id) const
{
    return std::find(eos_token_ids.begin(), eos_token_ids.end(), token_id) != eos_token_ids.end();
}

std::string ModelSpec::PastTensorName(int layer, bool key) const
{
    if (layer < 0 || layer >= num_hidden_layers) throw std::out_of_range("KV layer is outside ModelSpec");
    return past_prefix + std::to_string(layer) + (key ? ".key" : ".value");
}

std::string ModelSpec::PresentTensorName(int layer, bool key) const
{
    if (layer < 0 || layer >= num_hidden_layers) throw std::out_of_range("KV layer is outside ModelSpec");
    return present_prefix + std::to_string(layer) + (key ? ".key" : ".value");
}

std::size_t ModelSpec::KvTensorBytes(std::int64_t sequence_length) const
{
    if (sequence_length <= 0) throw std::invalid_argument("KV sequence length must be positive");
    auto elements = CheckedMultiply(static_cast<std::size_t>(num_kv_heads),
        static_cast<std::size_t>(sequence_length), "KV tensor elements");
    elements = CheckedMultiply(elements, static_cast<std::size_t>(head_dim), "KV tensor elements");
    return CheckedMultiply(elements, ElementSize(kv_dtype), "KV tensor bytes");
}

std::size_t ModelSpec::KvCacheBytes(std::int64_t capacity, int copies) const
{
    if (copies <= 0) throw std::invalid_argument("KV cache copies must be positive");
    auto bytes = KvTensorBytes(capacity);
    bytes = CheckedMultiply(bytes, static_cast<std::size_t>(num_hidden_layers), "KV cache bytes");
    bytes = CheckedMultiply(bytes, 2, "KV key/value bytes");
    return CheckedMultiply(bytes, static_cast<std::size_t>(copies), "KV cache copies");
}

} // namespace minillm
