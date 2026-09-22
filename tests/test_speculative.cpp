#include "ninfer_glm53/speculative.hpp"
#include <algorithm>
#include <cstdlib>
#include <iostream>

static void expect(bool ok, const char* message) {
    if (!ok) { std::cerr << message << '\n'; std::exit(1); }
}
int main() {
    using namespace ninfer::glm53;
    RankRecord r0, r1;
    const std::int32_t draft[]{10,11,12,13,14,15,16};
    for (std::uint32_t k = 1; k <= 7; ++k) {
        for (std::uint32_t matched = 0; matched <= k; ++matched) {
            std::int32_t target[8]; std::copy_n(draft, k, target); target[k] = 42;
            if (matched < k) target[matched] = 99;
            const auto result = commit_dflash2(draft, k, target, &r0, &r1);
            expect(result.ranks_agree && result.committed == matched + 1, "prefix plus correction/bonus");
            expect(result.accepted_drafts == matched, "accepted count differs from emitted count");
            for (std::uint32_t i = 0; i < matched; ++i) expect(result.tokens[i] == draft[i], "accepted token");
            expect(result.tokens[matched] == target[matched], "target correction/bonus missing");
            expect(r0.committed == result.committed && r1.committed == result.committed, "local record count");
            for (unsigned i = 0; i < 16; ++i) expect(r0.tokens[i] == r1.tokens[i], "local records differ");
        }
    }
    const auto saved0 = r0, saved1 = r1;
    const std::int32_t different[]{1,2};
    expect(!commit_same_tokens({draft,2},{different,2},&r0,&r1), "mismatch accepted");
    expect(!commit_same_tokens({draft,2},{draft,2},&r0,&r0), "same record passed as both ranks");
    expect(!commit_same_tokens({nullptr,2},{draft,2},&r0,&r1), "null view");
    expect(!commit_same_tokens({draft,0},{draft,0},&r0,&r1), "empty commit");
    expect(!commit_dflash2(draft,8,draft,&r0,&r1).ranks_agree, "out-of-bound k");
    expect(!commit_dflash2(draft,7,draft,nullptr,&r1).ranks_agree, "null rank");
    expect(r0.committed == saved0.committed && r1.committed == saved1.committed, "failure changed count");
    for (unsigned i=0;i<16;++i) expect(r0.tokens[i]==saved0.tokens[i] && r1.tokens[i]==saved1.tokens[i], "failure changed data");
    expect(commit_same_tokens({r0.tokens,2},{r1.tokens,2},&r0,&r1), "input aliases prior records");
    expect(r0.tokens[2] == 0 && r1.tokens[2] == 0, "stale record tail");
    const std::int32_t bad[]{-1};
    expect(!commit_same_tokens({bad,1},{bad,1},&r0,&r1), "negative token accepted");
    std::cout << "35 prefix/correction cases and invalid/alias cases passed\n";
}
