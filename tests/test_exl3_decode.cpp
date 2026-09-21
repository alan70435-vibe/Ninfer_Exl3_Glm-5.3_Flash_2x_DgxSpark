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

    std::uint16_t symbols[256];
    for (int i = 0; i < 256; ++i) symbols[i] = static_cast<std::uint16_t>((i * 3) & 15);
    std::uint16_t packed[kExl3PackedU16];
    std::uint16_t unpacked[256];
    pack_trellis_k4(symbols, packed);
    unpack_trellis_k4(packed, unpacked);
    for (int i = 0; i < 256; ++i) {
        if ((unpacked[i] & 15) != symbols[i]) {
            std::cerr << "trellis round trip failed at " << i << '\n';
            return 1;
        }
    }

    for (int i = 0; i < 256; ++i) symbols[i] = 0;
    int slot00 = -1;
    int slot01 = -1;
    for (int p = 0; p < 256; ++p) {
        if (perm[p] == 0) slot00 = p;
        if (perm[p] == 1) slot01 = p;
    }
    expect(slot00 == 0 && slot01 >= 0, "row-major slots");
    symbols[slot00] = 2;
    symbols[slot01] = 4;
    pack_trellis_k4(symbols, packed);
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
    // CUDA __hadd oracle for MCG symbols 2 and 4.
    expect(near(y[0], -0.759277344f * 2.f * 0.5f), "exl3 gemv column 0");
    expect(near(y[1], 0.665039062f * 2.f * 3.f), "exl3 gemv column 1");
    return 0;
}
