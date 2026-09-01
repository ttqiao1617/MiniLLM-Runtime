#include "tensor_utils.h"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace minillm {

std::vector<std::int64_t> LoadInputIds(const std::string& path)
{
    if (path.empty()) {
        return {};
    }

    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open input IDs file: " + path);
    }

    std::vector<std::int64_t> ids;
    std::string token;
    while (in >> token) {
        const std::string raw_token = token;
        if (!token.empty() && token.back() == ',') {
            token.pop_back();
        }
        if (token.empty()) {
            throw std::runtime_error("Invalid token ID in " + path + ": " + raw_token);
        }

        std::size_t consumed = 0;
        std::int64_t token_id = 0;
        try {
            token_id = std::stoll(token, &consumed);
        } catch (const std::exception&) {
            throw std::runtime_error("Invalid token ID in " + path + ": " + raw_token);
        }
        if (consumed != token.size() || token_id < 0) {
            throw std::runtime_error("Invalid token ID in " + path + ": " + raw_token);
        }
        ids.push_back(token_id);
    }
    return ids;
}

std::string ShapeToString(const std::vector<std::int64_t>& shape)
{
    std::string result = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            result += ", ";
        }
        result += std::to_string(shape[i]);
    }
    result += "]";
    return result;
}

void PrintShape(const std::string& label, const std::vector<std::int64_t>& shape)
{
    std::cout << label << ": " << ShapeToString(shape) << "\n";
}

std::vector<std::pair<std::int64_t, float>> TopK(const float* values, std::int64_t size, std::size_t k)
{
    if (values == nullptr || size <= 0) {
        throw std::runtime_error("Cannot compute top-k over empty logits");
    }

    std::vector<std::pair<std::int64_t, float>> ranked;
    ranked.reserve(static_cast<std::size_t>(size));
    for (std::int64_t i = 0; i < size; ++i) {
        ranked.emplace_back(i, values[static_cast<std::size_t>(i)]);
    }

    const auto top_count = std::min(k, ranked.size());
    std::partial_sort(ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(top_count), ranked.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    ranked.resize(top_count);
    return ranked;
}

} // namespace minillm
