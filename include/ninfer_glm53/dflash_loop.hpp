#pragma once

#include "ninfer_glm53/speculative.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::glm53 {

struct DflashOutput {
    std::vector<std::int32_t> tokens;
    std::string finish;
    int accepted_drafts = 0;
    int verify_rounds = 0;
    std::vector<std::int32_t> last_proposals;
    int materialized_generated = 0;
};

// generation_config.json EOS ids. 154822 and 13041 are ordinary tokens here.
[[nodiscard]] inline bool is_text_eos(std::int32_t token) {
    return token == 154820 || token == 154827 || token == 154829;
}

// Prompt is stepped under capture. Each verify round is rolled back with
// Model::restore before the truncated commit is stepped. The last emitted
// token of a finished generation is not stepped, matching greedy_tokens.
template <typename Model, typename Propose>
DflashOutput generate_dflash_continuation(Model& model, Propose&& propose, const std::vector<std::int32_t>& prompt,
                                         int new_tokens) {
    model.set_capture(true);
    for (const std::int32_t token : prompt) model.step(token);
    model.set_capture(false);

    DflashOutput output;
    while (static_cast<int>(output.tokens.size()) < new_tokens) {
        const std::int32_t anchor = model.argmax();
        if (is_text_eos(anchor)) {
            output.tokens.push_back(anchor);
            output.finish = "eos";
            break;
        }
        const int remaining = new_tokens - static_cast<int>(output.tokens.size());
        if (remaining == 1) {
            output.tokens.push_back(anchor);
            output.finish = "budget";
            break;
        }

        std::vector<std::int32_t> proposals = propose(anchor);
        if (proposals.size() != static_cast<std::size_t>(kDflashProposals)) {
            throw std::runtime_error("DFlash proposal count");
        }
        output.last_proposals = proposals;
        ++output.verify_rounds;

        const auto snap = model.save();
        SpecResult commit;
        try {
            std::vector<std::int32_t> posterior;
            posterior.reserve(static_cast<std::size_t>(kDflashProposals) + 1U);
            model.step(anchor);
            posterior.push_back(model.argmax());
            for (const std::int32_t draft : proposals) {
                model.step(draft);
                posterior.push_back(model.argmax());
            }
            RankRecord rank0;
            RankRecord rank1;
            commit = commit_dflash2(proposals.data(), kDflashProposals, posterior.data(), &rank0, &rank1);
            if (!commit.ranks_agree) throw std::runtime_error("DFlash rank commit failed");
        } catch (...) {
            model.restore(snap);
            throw;
        }
        // Drop every rejected proposal before building the commit. A later throw must not keep them.
        model.restore(snap);

        output.accepted_drafts += static_cast<int>(commit.accepted_drafts);
        std::vector<std::int32_t> round;
        round.reserve(static_cast<std::size_t>(commit.committed) + 1U);
        round.push_back(anchor);
        for (std::uint32_t index = 0; index < commit.committed; ++index) {
            round.push_back(commit.tokens[index]);
        }

        std::vector<std::int32_t> taken;
        taken.reserve(round.size());
        bool saw_eos = false;
        for (const std::int32_t token : round) {
            if (static_cast<int>(taken.size()) >= remaining) break;
            taken.push_back(token);
            if (is_text_eos(token)) {
                saw_eos = true;
                break;
            }
        }
        const bool stop = saw_eos || static_cast<int>(taken.size()) == remaining ||
                          static_cast<int>(output.tokens.size()) + static_cast<int>(taken.size()) >= new_tokens;

        output.tokens.insert(output.tokens.end(), taken.begin(), taken.end());
        const bool more = !output.tokens.empty() && static_cast<int>(output.tokens.size()) < new_tokens &&
                          !is_text_eos(output.tokens.back());
        const int round_count = static_cast<int>(taken.size());
        const int step_count = more ? round_count : round_count - 1;
        for (int index = 0; index < step_count; ++index) {
            model.step(taken[static_cast<std::size_t>(index)]);
        }
        if (step_count > 0) output.materialized_generated += step_count;

        if (saw_eos) {
            output.finish = "eos";
            break;
        }
        if (stop || static_cast<int>(output.tokens.size()) == new_tokens) {
            output.finish = "budget";
            break;
        }
    }
    return output;
}

}  // namespace ninfer::glm53
