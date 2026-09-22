#include "ninfer_glm53/dflash_loop.hpp"
#include "ninfer_glm53/speculative.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

static void expect(bool ok, const char* message) {
    if (!ok) { std::cerr << message << '\n'; std::exit(1); }
}

namespace {

struct CountModel {
    std::vector<std::int32_t> stepped;
    int eos_at = -1;
    std::int32_t eos_id = 154820;
    void set_capture(bool) {}
    void step(std::int32_t token) { stepped.push_back(token); }
    [[nodiscard]] std::int32_t argmax() const {
        if (static_cast<int>(stepped.size()) == eos_at) return eos_id;
        return static_cast<std::int32_t>(1000 + static_cast<int>(stepped.size()));
    }
    [[nodiscard]] std::vector<std::int32_t> save() const { return stepped; }
    void restore(const std::vector<std::int32_t>& snap) { stepped = snap; }
};

struct OracleRun {
    std::vector<std::int32_t> tokens;
    std::vector<std::int32_t> stepped;
};

OracleRun greedy_oracle(const std::vector<std::int32_t>& prompt, int new_tokens, int eos_at, std::int32_t eos_id) {
    CountModel model;
    model.eos_at = eos_at;
    model.eos_id = eos_id;
    for (const auto token : prompt) model.step(token);
    OracleRun out;
    while (static_cast<int>(out.tokens.size()) < new_tokens) {
        const auto id = model.argmax();
        out.tokens.push_back(id);
        if (static_cast<int>(out.tokens.size()) < new_tokens && !ninfer::glm53::is_text_eos(id)) model.step(id);
        else break;
    }
    out.stepped = std::move(model.stepped);
    return out;
}

}  // namespace
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

    const std::vector<std::int32_t> prompt{3, 9};
    for (int accept = 0; accept <= 7; ++accept) {
        for (int budget = 1; budget <= 8; ++budget) {
            int calls = 0;
            CountModel model;
            const auto output = generate_dflash_continuation(model, [&](std::int32_t anchor) {
                ++calls;
                std::vector<std::int32_t> proposals(static_cast<std::size_t>(kDflashProposals));
                for (int index = 0; index < static_cast<int>(kDflashProposals); ++index) {
                    proposals[static_cast<std::size_t>(index)] =
                        index < accept ? static_cast<std::int32_t>(anchor + 1 + index) : static_cast<std::int32_t>(999);
                }
                return proposals;
            }, prompt, budget);
            const auto oracle = greedy_oracle(prompt, budget, -1, 154820);
            if (output.tokens != oracle.tokens || model.stepped != oracle.stepped || output.finish != "budget" ||
                static_cast<int>(output.tokens.size()) != budget) {
                std::cerr << "dflash/greedy mismatch accept=" << accept << " budget=" << budget << '\n';
                std::exit(1);
            }
            expect(output.materialized_generated == static_cast<int>(model.stepped.size()) - static_cast<int>(prompt.size()),
                   "materialized count");
            expect(std::find(model.stepped.begin(), model.stepped.end(), 999) == model.stepped.end(), "rejected draft remained");
            if (budget == 1) expect(calls == 0 && output.verify_rounds == 0 && output.last_proposals.empty(), "budget 1 proposed");
            else expect(output.last_proposals.size() == static_cast<std::size_t>(kDflashProposals), "proposal width");
            if (accept == 0 && budget == 8) {
                expect(output.verify_rounds == 4 && output.accepted_drafts == 0, "accept 0 several rounds");
            }
            if (accept == 7 && budget == 8) {
                expect(output.verify_rounds == 1 && static_cast<int>(output.tokens.size()) == 8 && output.accepted_drafts == 7,
                       "accept 7 one round");
            }
        }
    }

    {
        CountModel model;
        model.eos_at = static_cast<int>(prompt.size()) + 3;
        int calls = 0;
        const auto output = generate_dflash_continuation(model, [&](std::int32_t anchor) {
            ++calls;
            std::vector<std::int32_t> proposals(static_cast<std::size_t>(kDflashProposals));
            for (int index = 0; index < static_cast<int>(kDflashProposals); ++index) {
                proposals[static_cast<std::size_t>(index)] = static_cast<std::int32_t>(anchor + 1 + index);
            }
            return proposals;
        }, prompt, 8);
        const auto oracle = greedy_oracle(prompt, 8, model.eos_at, 154820);
        expect(output.tokens == oracle.tokens && model.stepped == oracle.stepped, "eos continuation diverged");
        expect(!output.tokens.empty() && output.tokens.back() == 154820 && output.finish == "eos", "eos ending");
        expect(std::find(model.stepped.begin(), model.stepped.end(), 154820) == model.stepped.end(), "eos was stepped");
        expect(calls > 0, "eos verify did not propose");
        expect(output.materialized_generated == static_cast<int>(model.stepped.size()) - static_cast<int>(prompt.size()),
               "eos materialized");
    }

    for (const std::int32_t eos_id : {154820, 154827, 154829}) {
        CountModel model;
        model.eos_at = static_cast<int>(prompt.size());
        model.eos_id = eos_id;
        int calls = 0;
        const auto output = generate_dflash_continuation(model, [&](std::int32_t) {
            ++calls;
            return std::vector<std::int32_t>(static_cast<std::size_t>(kDflashProposals), 999);
        }, prompt, 8);
        const auto oracle = greedy_oracle(prompt, 8, model.eos_at, eos_id);
        expect(calls == 0, "anchor eos proposed");
        expect(output.finish == "eos" && output.tokens.size() == 1U && output.tokens[0] == eos_id, "anchor eos token");
        expect(model.stepped == prompt && model.stepped == oracle.stepped, "anchor eos stepped");
        expect(output.materialized_generated == 0 && output.verify_rounds == 0, "anchor eos counts");
    }

    {
        CountModel model;
        model.eos_at = static_cast<int>(prompt.size()) + 3;
        model.eos_id = 154822;
        const auto output = generate_dflash_continuation(model, [&](std::int32_t anchor) {
            std::vector<std::int32_t> proposals(static_cast<std::size_t>(kDflashProposals));
            for (int index = 0; index < static_cast<int>(kDflashProposals); ++index) {
                proposals[static_cast<std::size_t>(index)] = static_cast<std::int32_t>(anchor + 1 + index);
            }
            return proposals;
        }, prompt, 8);
        const auto oracle = greedy_oracle(prompt, 8, model.eos_at, 154822);
        expect(output.tokens == oracle.tokens && model.stepped == oracle.stepped, "154822 stopped early");
        expect(output.finish == "budget" && static_cast<int>(output.tokens.size()) == 8, "154822 is not eos");
        expect(std::find(model.stepped.begin(), model.stepped.end(), 154822) != model.stepped.end(), "154822 was not stepped");
    }
    std::cout << "dflash continuation matches greedy across accept counts, budgets, and eos\n";
}
