#include "kv_block_manager.h"

#include <stdexcept>
#include <vector>

int main()
{
    minillm::KvBlockManager manager(16, 6);
    auto a = manager.Reserve("a", 17, 8);
    auto b = manager.Reserve("b", 32, 20);
    if (a != std::vector<int>({0, 1}) || b != std::vector<int>({2, 3})) return 1;
    auto stats = manager.Stats();
    if (stats.allocated_blocks != 4 || stats.free_blocks != 2 || stats.live_tokens != 28
        || stats.committed_tokens != 49 || stats.reserved_tokens != 64
        || stats.block_rounding_waste_tokens != 15) return 2;
    manager.UpdateLiveTokens("a", 17);
    if (manager.Stats().live_tokens != 37) return 3;
    if (manager.CanReserve(33)) return 4;
    manager.Release("a");
    auto c = manager.Reserve("c", 33, 1);
    if (c.size() != 3 || manager.Stats().free_blocks != 1) return 5;
    manager.Release("b"); manager.Release("c");
    if (manager.Stats().free_blocks != 6) return 6;
    bool duplicate_failed = false;
    manager.Reserve("x", 1, 1);
    try { manager.Reserve("x", 1, 1); } catch (const std::runtime_error&) { duplicate_failed = true; }
    return duplicate_failed ? 0 : 7;
}
