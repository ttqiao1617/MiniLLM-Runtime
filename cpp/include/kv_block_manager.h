#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace minillm {

struct KvBlockPoolStats {
    int total_blocks{0};
    int free_blocks{0};
    int allocated_blocks{0};
    int live_tokens{0};
    int committed_tokens{0};
    int reserved_tokens{0};
    int block_rounding_waste_tokens{0};
};

class KvBlockManager {
public:
    KvBlockManager(int block_size, int num_blocks);
    bool CanReserve(int max_tokens) const;
    std::vector<int> Reserve(const std::string& request_id, int max_tokens, int live_tokens);
    void UpdateLiveTokens(const std::string& request_id, int live_tokens);
    void Release(const std::string& request_id);
    const std::vector<int>& BlockTable(const std::string& request_id) const;
    KvBlockPoolStats Stats() const;
    int BlockSize() const { return block_size_; }

private:
    struct Allocation {
        std::vector<int> blocks;
        int committed_tokens{0};
        int live_tokens{0};
    };
    int BlocksFor(int tokens) const;
    int block_size_;
    int num_blocks_;
    std::vector<int> free_blocks_;
    std::unordered_map<std::string, Allocation> allocations_;
};

} // namespace minillm
