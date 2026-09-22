#include "ninfer_glm53/text_forward.hpp"

#include <cmath>
#include <cstdlib>
#include <dirent.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace {

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

bool near(float actual, float expected) { return std::fabs(actual - expected) < 1e-4f; }

}  // namespace

int main() {
    using namespace ninfer::glm53;
    float hadamard[128] = {};
    hadamard[3] = 1.5f;
    hadamard[90] = -0.25f;
    float original[128];
    for (int i = 0; i < 128; ++i) original[i] = hadamard[i];
    block_hadamard_128(hadamard, 128);
    block_hadamard_128(hadamard, 128);
    for (int i = 0; i < 128; ++i) expect(near(hadamard[i], original[i]), "hadamard is an involution");

    const float gate[] = {0.f, 100.f, -2.f};
    const float up[] = {1.f, 100.f, -20.f};
    float hidden[3];
    swiglu_clamp(gate, up, 3, 10.f, hidden);
    expect(near(hidden[0], 0.f) && near(hidden[1], 99.995460213f) && near(hidden[2], 2.384058440f), "clamped swiglu");

    const float logits[] = {0.f, 10.f, 3.f, 10.f, 1.f};
    const float bias[] = {0.f, 0.f, 0.f, 0.f, 0.f};
    int indices[2];
    float weights[2];
    router_select(logits, bias, 5, 2, 2.5f, true, indices, weights);
    expect(indices[0] == 1 && indices[1] == 3, "router keeps the lower index on a tie");
    expect(near(weights[0], 1.25f) && near(weights[1], 1.25f), "router renormalizes");

    float forget = 0.f;
    const float proj[] = {0.f};
    const float dt[] = {0.f};
    const float a_log[] = {0.f};
    kda_forget_gate(proj, dt, a_log, -5.f, 1, 1, &forget);
    expect(near(forget, -2.5f), "forget gate lower bound");

    float mem[3] = {0.f, 0.f, 0.f};
    const float taps[] = {0.f, 0.f, 0.f, 2.f};
    const float input[] = {3.f};
    float conv = 0.f;
    causal_conv_silu(input, taps, mem, 1, 4, &conv);
    expect(near(conv, 5.985164f), "first causal tap is silu");
    expect(near(mem[2], 3.f), "conv state stores the current input");

    auto silu_of = [](float value) {
        const float exponent = std::exp(value >= 0.f ? -value : value);
        const float sig = value >= 0.f ? 1.f / (1.f + exponent) : exponent / (1.f + exponent);
        return value * sig;
    };
    const float alias_taps[] = {0.f, 0.f, 1.f, 1.f};
    const float alias_inputs[] = {2.f, 0.f, -1.f, 3.f, 0.f};
    float alias_hist[3] = {};
    float separate_hist[3] = {};
    for (int step = 0; step < 5; ++step) {
        float aliased = alias_inputs[step];
        float separate = 0.f;
        causal_conv_silu(&aliased, alias_taps, alias_hist, 1, 4, &aliased);
        causal_conv_silu(&alias_inputs[step], alias_taps, separate_hist, 1, 4, &separate);
        if (step == 0) {
            expect(near(aliased, 1.761594f), "in-place step 0");
            expect(near(alias_hist[2], 2.f), "history tail is the raw input");
            expect(!near(alias_hist[2], aliased), "history stored silu");
        }
        if (step == 1) expect(near(aliased, 1.761594f), "in-place step 1");
        expect(near(aliased, separate), "in-place output diverged");
        for (int tap = 0; tap < 3; ++tap) expect(near(alias_hist[tap], separate_hist[tap]), "in-place history diverged");
    }

    const float weighted_taps[] = {0.f, 0.f, 1.f, 2.f};
    const float weighted_inputs[] = {3.f, 1.f, -2.f, 4.f};
    float weighted_alias_hist[3] = {};
    float weighted_hist[3] = {};
    for (int step = 0; step < 4; ++step) {
        float aliased = weighted_inputs[step];
        float separate = 0.f;
        causal_conv_silu(&aliased, weighted_taps, weighted_alias_hist, 1, 4, &aliased);
        causal_conv_silu(&weighted_inputs[step], weighted_taps, weighted_hist, 1, 4, &separate);
        if (step == 0) {
            expect(near(aliased, 5.985164f) && near(separate, 5.985164f), "weighted first output");
            expect(near(weighted_alias_hist[2], 3.f) && near(weighted_hist[2], 3.f), "weighted history tail");
        }
        if (step == 1) expect(near(separate, 4.966536f), "weighted second output");
        expect(near(aliased, separate), "weighted in-place output diverged");
        for (int tap = 0; tap < 3; ++tap) {
            expect(near(weighted_alias_hist[tap], weighted_hist[tap]), "weighted history diverged");
        }
    }

    float kernel1_mem_a[] = {1.25f, -3.5f};
    float kernel1_mem_b[] = {1.25f, -3.5f};
    const float kernel1_saved[] = {1.25f, -3.5f};
    const float kernel1_weight[] = {2.f, -1.f};
    const float kernel1_inputs[][2] = {{-1.5f, 0.5f}, {0.25f, -2.f}, {-3.f, 1.5f}};
    for (int step = 0; step < 3; ++step) {
        float aliased[2] = {kernel1_inputs[step][0], kernel1_inputs[step][1]};
        float raw[2] = {kernel1_inputs[step][0], kernel1_inputs[step][1]};
        float separate[2] = {};
        causal_conv_silu(aliased, kernel1_weight, kernel1_mem_a, 2, 1, aliased);
        causal_conv_silu(raw, kernel1_weight, kernel1_mem_b, 2, 1, separate);
        if (step == 0) {
            expect(near(separate[0], silu_of(2.f * -1.5f)) && near(separate[1], silu_of(-1.f * 0.5f)), "kernel 1 silu");
        }
        expect(near(aliased[0], separate[0]) && near(aliased[1], separate[1]), "kernel 1 in-place");
        expect(near(kernel1_mem_a[0], kernel1_saved[0]) && near(kernel1_mem_a[1], kernel1_saved[1]), "kernel 1 history a");
        expect(near(kernel1_mem_b[0], kernel1_saved[0]) && near(kernel1_mem_b[1], kernel1_saved[1]), "kernel 1 history b");
    }

    const float kernel2_weight[] = {0.5f, 1.f, -1.f, 2.f};
    const float kernel2_inputs[][2] = {{1.f, -2.f}, {-0.5f, 3.f}, {2.f, 0.5f}};
    float kernel2_mem_a[] = {0.25f, -0.5f};
    float kernel2_mem_b[] = {0.25f, -0.5f};
    for (int step = 0; step < 3; ++step) {
        float aliased[2] = {kernel2_inputs[step][0], kernel2_inputs[step][1]};
        float raw[2] = {kernel2_inputs[step][0], kernel2_inputs[step][1]};
        float separate[2] = {};
        causal_conv_silu(aliased, kernel2_weight, kernel2_mem_a, 2, 2, aliased);
        causal_conv_silu(raw, kernel2_weight, kernel2_mem_b, 2, 2, separate);
        if (step == 0) {
            expect(near(separate[0], silu_of(1.125f)) && near(separate[1], silu_of(-3.5f)), "kernel 2 first mix");
        }
        expect(near(aliased[0], separate[0]) && near(aliased[1], separate[1]), "kernel 2 in-place");
        expect(near(kernel2_mem_a[0], raw[0]) && near(kernel2_mem_a[1], raw[1]), "kernel 2 history is raw");
        expect(near(kernel2_mem_b[0], raw[0]) && near(kernel2_mem_b[1], raw[1]), "kernel 2 separate history");
    }

    float untouched = 4.f;
    float untouched_out = 8.f;
    float untouched_mem = 3.f;
    bool kernel_threw = false;
    try {
        causal_conv_silu(&untouched, &untouched, &untouched_mem, 1, 0, &untouched_out);
    } catch (const std::runtime_error&) {
        kernel_threw = true;
    }
    expect(kernel_threw && untouched == 4.f && untouched_out == 8.f && untouched_mem == 3.f, "kernel < 1");

    float overlap_samples[] = {1.f, -2.f, 3.f};
    float overlap_saved[] = {1.f, -2.f, 3.f};
    float overlap_mem[] = {4.f, 5.f, 6.f, 7.f, 8.f, 9.f};
    float overlap_mem_saved[] = {4.f, 5.f, 6.f, 7.f, 8.f, 9.f};
    const float overlap_weight[] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    bool overlap_threw = false;
    try {
        causal_conv_silu(overlap_samples, overlap_weight, overlap_mem, 2, 4, overlap_samples + 1);
    } catch (const std::runtime_error& error) {
        overlap_threw = std::string(error.what()) == "causal conv partial overlap";
    }
    expect(overlap_threw, "partial overlap should throw");
    for (int index = 0; index < 3; ++index) expect(overlap_samples[index] == overlap_saved[index], "overlap wrote x");
    for (int index = 0; index < 6; ++index) expect(overlap_mem[index] == overlap_mem_saved[index], "overlap wrote mem");

    float adjacent[] = {1.f, -1.f, 0.f, 0.f};
    float adjacent_mem[] = {0.5f, -0.25f};
    const float adjacent_weight[] = {0.f, 1.f, 0.f, 1.f};
    causal_conv_silu(adjacent, adjacent_weight, adjacent_mem, 2, 2, adjacent + 2);
    expect(near(adjacent_mem[0], 1.f) && near(adjacent_mem[1], -1.f), "adjacent history");
    expect(near(adjacent[2], silu_of(1.f)) && near(adjacent[3], silu_of(-1.f)), "adjacent output");
    expect(near(adjacent[0], 1.f) && near(adjacent[1], -1.f), "adjacent left input");

    const float snap_weight[] = {1.f, -0.5f, 0.25f, 0.5f, 1.f, -1.f};
    const float snap_initial[] = {0.5f, -1.f, 2.f, 0.25f};
    const float snap_inputs[][2] = {{1.f, -2.f}, {0.5f, 0.25f}, {-1.5f, 3.f}};
    float snap_mem[4] = {0.5f, -1.f, 2.f, 0.25f};
    float snap_outs[3][2] = {};
    float snap_hists[3][4] = {};
    for (int step = 0; step < 3; ++step) {
        causal_conv_silu(snap_inputs[step], snap_weight, snap_mem, 2, 3, snap_outs[step]);
        for (int index = 0; index < 4; ++index) snap_hists[step][index] = snap_mem[index];
    }
    for (int index = 0; index < 4; ++index) snap_mem[index] = snap_initial[index];
    for (int step = 0; step < 3; ++step) {
        float replay[2] = {};
        causal_conv_silu(snap_inputs[step], snap_weight, snap_mem, 2, 3, replay);
        expect(near(replay[0], snap_outs[step][0]) && near(replay[1], snap_outs[step][1]), "snapshot output");
        for (int index = 0; index < 4; ++index) expect(near(snap_mem[index], snap_hists[step][index]), "snapshot history");
    }

    constexpr int kda_channels = 64;
    constexpr int kda_kernel = 4;
    constexpr int kda_history = kda_kernel - 1;
    std::vector<float> kda_weight(static_cast<std::size_t>(kda_channels * kda_kernel));
    std::vector<float> kda_mem_in(static_cast<std::size_t>(kda_channels * kda_history));
    std::vector<float> kda_mem_out(kda_mem_in.size());
    std::vector<float> kda_input(static_cast<std::size_t>(kda_channels));
    std::vector<float> kda_alias(static_cast<std::size_t>(kda_channels));
    std::vector<float> kda_output(static_cast<std::size_t>(kda_channels));
    for (int channel = 0; channel < kda_channels; ++channel) {
        for (int tap = 0; tap < kda_kernel; ++tap) {
            kda_weight[static_cast<std::size_t>(channel * kda_kernel + tap)] = static_cast<float>((channel + tap) % 5) * 0.25f - 0.5f;
        }
        for (int tap = 0; tap < kda_history; ++tap) {
            kda_mem_in[static_cast<std::size_t>(channel * kda_history + tap)] = static_cast<float>((channel + 3 - tap) % 3) * 0.5f - 0.25f;
        }
        kda_input[static_cast<std::size_t>(channel)] = static_cast<float>(channel % 7) * 0.3f - 1.f;
    }
    kda_mem_out = kda_mem_in;
    kda_alias = kda_input;
    for (int step = 0; step < 2; ++step) {
        causal_conv_silu(kda_input.data(), kda_weight.data(), kda_mem_out.data(), kda_channels, kda_kernel, kda_output.data());
        causal_conv_silu(kda_alias.data(), kda_weight.data(), kda_mem_in.data(), kda_channels, kda_kernel, kda_alias.data());
        for (int channel = 0; channel < kda_channels; ++channel) {
            expect(near(kda_alias[static_cast<std::size_t>(channel)], kda_output[static_cast<std::size_t>(channel)]),
                   "kda-sized in-place output");
            kda_input[static_cast<std::size_t>(channel)] = kda_output[static_cast<std::size_t>(channel)] * 0.5f - 0.1f;
            kda_alias[static_cast<std::size_t>(channel)] = kda_input[static_cast<std::size_t>(channel)];
        }
        for (std::size_t index = 0; index < kda_mem_in.size(); ++index) {
            expect(near(kda_mem_in[index], kda_mem_out[index]), "kda-sized in-place history");
        }
    }

    float state[4] = {};
    const float q[] = {3.f, 4.f};
    const float k[] = {0.f, 2.f};
    const float v[] = {1.f, 0.f};
    const float g[] = {0.f, 0.f};
    const float beta[] = {1.f};
    float out[2];
    kda_recurrent_heads(state, q, k, v, g, beta, 1, 2, out);
    expect(near(out[0], 0.565685343f) && near(out[1], 0.f), "kda recurrence");

    const float streams[] = {0.5f, 1.5f};
    float fn[16] = {};
    fn[0] = 1.f;
    fn[3] = 1.f;
    float base[8] = {};
    const float scale[] = {1.f, 1.f, 1.f};
    float post[2];
    float comb[4];
    float collapsed[1];
    mhc_project(streams, 2, 1, fn, base, scale, 1e-5f, 1e-6f, 2, post, comb, collapsed);
    expect(near(collapsed[0], 1.494128191f), "mhc collapse");
    expect(near(post[0], 1.f) && near(post[1], 1.f), "mhc post");
    expect(near(comb[0], 0.4999995f) && near(comb[3], 0.4999995f), "mhc sinkhorn");

    const float residual[] = {0.5f, 1.5f};
    const float branch[] = {2.f};
    const float mix_post[] = {1.f, 1.f};
    const float mix_comb[] = {0.5f, 0.5f, 0.5f, 0.5f};
    float mixed[2];
    mhc_combine(residual, branch, mix_post, mix_comb, 2, 1, mixed);
    expect(near(mixed[0], 3.f) && near(mixed[1], 3.f), "mhc combine");

    std::string text;
    std::vector<std::int32_t> tokens;
    expect(fixed_prompt_tokens("fixed-text-v1", text, tokens) && text == "Hi" && tokens.size() == 1U && tokens[0] == 13041,
           "fixed prompt");
    expect(!fixed_prompt_tokens("other", text, tokens), "unknown prompt");

    std::filesystem::path store_dir;
    std::filesystem::path tp_dir;
    auto cleanup_temps = [&]() {
        std::error_code error;
        if (!store_dir.empty()) std::filesystem::remove_all(store_dir, error);
        if (!tp_dir.empty()) std::filesystem::remove_all(tp_dir, error);
    };
    auto require = [&](bool condition, const char* message) {
        if (!condition) {
            cleanup_temps();
            std::cerr << message << '\n';
            std::exit(1);
        }
    };
    auto count_open_fds = []() {
        DIR* directory = ::opendir("/proc/self/fd");
        if (directory == nullptr) {
            std::cerr << "fd scan failed\n";
            std::exit(1);
        }
        const int skip = ::dirfd(directory);
        int count = 0;
        while (const dirent* entry = ::readdir(directory)) {
            char* end = nullptr;
            const long parsed = std::strtol(entry->d_name, &end, 10);
            if (end == entry->d_name || *end != '\0') continue;
            if (static_cast<int>(parsed) == skip) continue;
            ++count;
        }
        ::closedir(directory);
        return count;
    };
    auto maps_contain = [](const std::string& needle) {
        std::ifstream input("/proc/self/maps");
        std::string line;
        while (std::getline(input, line)) {
            if (line.find(needle) != std::string::npos) return true;
        }
        return false;
    };

    store_dir = std::filesystem::temp_directory_path() / ("ninfer-exl3-store-" + std::to_string(::getpid()));
    std::filesystem::remove_all(store_dir);
    require(std::filesystem::create_directory(store_dir), "store temp dir");
    {
        std::ofstream out(store_dir / "model.safetensors", std::ios::binary);
        const char zeros[8] = {};
        out.write(zeros, sizeof(zeros));
        require(static_cast<bool>(out), "failed to write zero safetensors header");
    }
    require(std::filesystem::file_size(store_dir / "model.safetensors") == 8U, "safetensors size");
    const int fds_before_store = count_open_fds();
    for (int attempt = 0; attempt < 32; ++attempt) {
        bool threw = false;
        try {
            const auto generated = generate_text(store_dir, "fixed-text-v1", "greedy", 1);
            (void)generated;
        } catch (const std::exception& error) {
            threw = std::string(error.what()).find("bad safetensors header") != std::string::npos;
            if (!threw) std::cerr << error.what() << '\n';
        }
        require(threw, "store constructor did not reject the zero header");
    }
    const int fds_after_store = count_open_fds();
    if (fds_after_store != fds_before_store) {
        std::cerr << "store fd count " << fds_before_store << " -> " << fds_after_store << '\n';
        require(false, "store fd leak");
    }
    require(!maps_contain(store_dir.string()), "store mapping leaked");
    cleanup_temps();
    require(!std::filesystem::exists(store_dir), "store temp dir remains");

    tp_dir = std::filesystem::temp_directory_path() / ("ninfer-exl3-tp2-" + std::to_string(::getpid()));
    std::filesystem::remove_all(tp_dir);
    require(std::filesystem::create_directory(tp_dir), "tp2 temp dir");
    const int fds_before_tp = count_open_fds();
    bool tp_threw = false;
    GenerateResult tp_result;
    try {
        tp_result = generate_text(tp_dir, "fixed-text-v1", "tp2", 1);
    } catch (const std::exception&) {
        tp_threw = true;
    }
    require(tp_threw || !tp_result.ok, "tp2 empty checkpoint succeeded");
    const int fds_after_tp = count_open_fds();
    if (fds_after_tp != fds_before_tp) {
        std::cerr << "tp2 fd count " << fds_before_tp << " -> " << fds_after_tp << '\n';
        require(false, "tp2 fd leak");
    }
    int child_status = 0;
    const pid_t leftover = ::waitpid(-1, &child_status, WNOHANG);
    require(leftover == -1, "tp2 left a child process");
    cleanup_temps();
    require(!std::filesystem::exists(tp_dir), "tp2 temp dir remains");
    return 0;
}
