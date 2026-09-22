#include "ninfer_glm53/exl3_decode.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

bool near(float actual, float expected) { return std::fabs(actual - expected) < 1e-5f; }

}  // namespace

int main() {
    using namespace ninfer::glm53;
    std::uint16_t perm[256];
    exl3_tensor_core_perm(perm);
    expect(perm[0] == 0 && perm[1] == 16 && perm[2] == 128 && perm[3] == 144, "tensor-core perm head");
    expect(perm[4] == 8 && perm[5] == 24 && perm[6] == 136 && perm[7] == 152, "tensor-core perm tail");

    std::uint16_t packed[kExl3PackedU16];
    std::uint16_t states[256];
    for (auto& word : packed) word = 0xffffu;
    unpack_trellis_k4(packed, states);
    for (const auto state : states) expect(state == 0xffffu, "all-ones tile is state 65535");
    expect(near(mcg_symbol(states[0]), mcg_symbol(0xffffu)), "all-ones mcg uses the 16-bit state");
    expect(std::fabs(mcg_symbol(0xffffu) - mcg_symbol(0xfu)) > 0.1f, "nibble 15 is not state 65535");

    for (auto& word : packed) word = 0;
    unpack_trellis_k4(packed, states);
    for (const auto state : states) expect(state == 0, "all-zero tile is state 0");

    // Non-periodic payload. Expected windows come from the published bit schedule, not from pack_trellis_k4.
    for (int i = 0; i < kExl3PackedU16; ++i) packed[i] = static_cast<std::uint16_t>(0x1111u * static_cast<unsigned>(i + 1));
    unpack_trellis_k4(packed, states);
    expect(states[0] == 0x32f2u && states[1] == 0x2f22u, "cross-word window 0");
    expect(states[16] == 0x3336u && states[17] == 0x3366u, "cross-word window 8");
    expect(states[254] == 0x0332u && states[255] == 0x332fu, "wrapped tail window");

    for (auto& word : packed) word = 0;
    float suh[16];
    float svh[16];
    for (int i = 0; i < 16; ++i) {
        suh[i] = 1.f;
        svh[i] = 1.f;
    }
    suh[0] = 2.f;
    svh[0] = 0.5f;
    svh[1] = 3.f;
    float tile[256];
    exl3_decode_tile(packed, suh, svh, tile);
    float x[16] = {1.f};
    float y[16] = {};
    exl3_gemv_tile(x, tile, y);
    const float symbol0 = mcg_symbol(0);
    expect(near(y[0], symbol0 * 2.f * 0.5f), "exl3 gemv column 0");
    expect(near(y[1], symbol0 * 2.f * 3.f), "exl3 gemv column 1");
    return 0;
}
