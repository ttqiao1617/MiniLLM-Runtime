#include "kv_block_manager.h"

#include <stdexcept>

namespace minillm {

KvBlockManager::KvBlockManager(int block_size, int num_blocks)
    : block_size_(block_size), num_blocks_(num_blocks)
{
    if (block_size <= 0 || num_blocks <= 0) throw std::runtime_error("KV block size/count must be positive");
    free_blocks_.reserve(static_cast<std::size_t>(num_blocks));
    for (int block = num_blocks - 1; block >= 0; --block) free_blocks_.push_back(block);
}

int KvBlockManager::BlocksFor(int tokens) const
{
    if (tokens <= 0) throw std::runtime_error("KV token reservation must be positive");
    return (tokens + block_size_ - 1) / block_size_;
}

bool KvBlockManager::CanReserve(int max_tokens) const
{
    return max_tokens > 0 && BlocksFor(max_tokens) <= static_cast<int>(free_blocks_.size());
}

std::vector<int> KvBlockManager::Reserve(
    const std::string& request_id, int max_tokens, int live_tokens)
{
    if (request_id.empty() || allocations_.count(request_id)) throw std::runtime_error("duplicate/empty KV request id");
    if (live_tokens < 0 || live_tokens > max_tokens) throw std::runtime_error("invalid KV live token count");
    const int count = BlocksFor(max_tokens);
    if (count > static_cast<int>(free_blocks_.size())) throw std::runtime_error("KV block pool exhausted");
    Allocation allocation;
    allocation.committed_tokens = max_tokens;
    allocation.live_tokens = live_tokens;
    allocation.blocks.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        allocation.blocks.push_back(free_blocks_.back());
        free_blocks_.pop_back();
    }
    auto [it, inserted] = allocations_.emplace(request_id, std::move(allocation));
    if (!inserted) throw std::runtime_error("failed to insert KV allocation");
    return it->second.blocks;
}

void KvBlockManager::UpdateLiveTokens(const std::string& request_id, int live_tokens)
{
    auto it = allocations_.find(request_id);
    if (it == allocations_.end()) throw std::runtime_error("unknown KV request id");
    if (live_tokens < 0 || live_tokens > it->second.committed_tokens) throw std::runtime_error("KV live tokens exceed reservation");
    it->second.live_tokens = live_tokens;
}

void KvBlockManager::Release(const std::string& request_id)
{
    auto it = allocations_.find(request_id);
    if (it == allocations_.end()) throw std::runtime_error("release of unknown KV request id");
    for (auto block : it->second.blocks) free_blocks_.push_back(block);
    allocations_.erase(it);
}

const std::vector<int>& KvBlockManager::BlockTable(const std::string& request_id) const
{
    auto it = allocations_.find(request_id);
    if (it == allocations_.end()) throw std::runtime_error("unknown KV request id");
    return it->second.blocks;
}

KvBlockPoolStats KvBlockManager::Stats() const
{
    KvBlockPoolStats stats;
    stats.total_blocks = num_blocks_;
    stats.free_blocks = static_cast<int>(free_blocks_.size());
    stats.allocated_blocks = num_blocks_ - stats.free_blocks;
    for (const auto& [_, allocation] : allocations_) {
        stats.live_tokens += allocation.live_tokens;
        stats.committed_tokens += allocation.committed_tokens;
        stats.reserved_tokens += static_cast<int>(allocation.blocks.size()) * block_size_;
    }
    stats.block_rounding_waste_tokens = stats.reserved_tokens - stats.committed_tokens;
    return stats;
}

} // namespace minillm
