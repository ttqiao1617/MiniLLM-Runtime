#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace minillm {

std::vector<std::int64_t> LoadInputIds(const std::string& path);
std::string ShapeToString(const std::vector<std::int64_t>& shape);
void PrintShape(const std::string& label, const std::vector<std::int64_t>& shape);
std::vector<std::pair<std::int64_t, float>> TopK(const float* values, std::int64_t size, std::size_t k);

} // namespace minillm
