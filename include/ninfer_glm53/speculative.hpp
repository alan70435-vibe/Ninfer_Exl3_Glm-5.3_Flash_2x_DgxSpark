#pragma once
#include <cstdint>

namespace ninfer::glm53 {
inline constexpr std::uint32_t kDflashProposals = 7;
struct RankRecord {
    std::int32_t tokens[16]{};
    std::uint32_t committed = 0;
};
struct CommitView {
    const std::int32_t* tokens = nullptr;
    std::uint32_t count = 0;
};
// Host-only local records, NOT distributed acknowledgements or cache commits.
// Invalid/mismatched inputs change neither record. Records must be distinct.
[[nodiscard]] bool commit_same_tokens(const CommitView& rank0_view, const CommitView& rank1_view,
                                      RankRecord* rank0, RankRecord* rank1);
struct SpecResult {
    std::int32_t tokens[16]{};
    std::uint32_t committed = 0;
    bool ranks_agree = false; // compatibility: local record equality only, not NCCL
    std::uint32_t accepted_drafts = 0;
};
// Greedy host oracle, with no EOS/stop/budget policy: matching draft prefix plus
// target[matched] correction (or target[k] bonus). draft has k entries, target
// has k+1; k is in 1..7. Does NOT install KV/KDA state or execute either model.
[[nodiscard]] SpecResult commit_dflash2(const std::int32_t* draft, std::uint32_t k, const std::int32_t* target,
                                        RankRecord* rank0, RankRecord* rank1);
}  // namespace ninfer::glm53
