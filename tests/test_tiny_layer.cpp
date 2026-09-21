#include "ninfer_glm53/exl3_decode.hpp"
#include "ninfer_glm53/tiny_layer.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace {

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

}  // namespace

int main() {
    using namespace ninfer::glm53;
    constexpr int width = 16;
    std::uint16_t symbols[256] = {};
    symbols[0] = 2;
    std::uint16_t packed[kExl3PackedU16];
    pack_trellis_k4(symbols, packed);
    float suh[16];
    float svh[16];
    for (int i = 0; i < width; ++i) {
        suh[i] = 1.f;
        svh[i] = 1.f;
    }
    float tile[256];
    exl3_decode_tile(packed, suh, svh, tile);

    float x[16];
    float norm_w[16];
    for (int i = 0; i < width; ++i) {
        x[i] = 0.25f * static_cast<float>(i + 1);
        norm_w[i] = 1.f;
    }
    float y[16];
    dense_residual_proj(x, norm_w, tile, 1e-5f, width, y);

    float normed[16];
    rmsnorm(x, norm_w, 1e-5f, width, normed);
    for (int row = 0; row < width; ++row) {
        float sum = x[row];
        for (int col = 0; col < width; ++col) sum += tile[row * width + col] * normed[col];
        if (std::fabs(y[row] - sum) > 1e-5f) {
            std::cerr << "layer mismatch at " << row << '\n';
            return 1;
        }
    }
    expect(y[0] != x[0], "projection changes the residual");
    return 0;
}
