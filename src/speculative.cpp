#include "ninfer_glm53/speculative.hpp"
#include <algorithm>

namespace ninfer::glm53 {
bool commit_same_tokens(const CommitView& a, const CommitView& b, RankRecord* rank0, RankRecord* rank1) {
    if (!rank0 || !rank1 || rank0 == rank1 || !a.tokens || !b.tokens ||
        !a.count || a.count != b.count || a.count > 16u) return false;
    RankRecord staged;
    for (std::uint32_t i = 0; i < a.count; ++i) {
        if (a.tokens[i] < 0 || a.tokens[i] != b.tokens[i]) return false;
        staged.tokens[i] = a.tokens[i];
    }
    // Copy before mutating either destination, including when views refer to a
    // previous record. These plain assignments cannot throw; tail is cleared.
    staged.committed = a.count;
    *rank0 = staged;
    *rank1 = staged;
    return true;
}
SpecResult commit_dflash2(const std::int32_t* draft, std::uint32_t k, const std::int32_t* target,
                          RankRecord* rank0, RankRecord* rank1) {
    SpecResult result;
    if (!draft || !target || !rank0 || !rank1 || rank0 == rank1 || k == 0u || k > kDflashProposals) return result;
    for (std::uint32_t i = 0; i < k; ++i) if (draft[i] < 0 || target[i] < 0) return result;
    if (target[k] < 0) return result;
    std::uint32_t matched = 0;
    while (matched < k && draft[matched] == target[matched]) ++matched;
    std::int32_t accepted[16]{};
    std::copy_n(draft, matched, accepted);
    accepted[matched] = target[matched];
    const CommitView view{accepted, matched + 1};
    result.ranks_agree = commit_same_tokens(view, view, rank0, rank1);
    if (result.ranks_agree) {
        result.committed = view.count;
        result.accepted_drafts = matched;
        std::copy_n(accepted, view.count, result.tokens);
    }
    return result;
}
}  // namespace ninfer::glm53
