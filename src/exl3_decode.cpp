#include "ninfer_glm53/exl3_decode.hpp"

#include <cstring>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ninfer::glm53 {
namespace {

float fp16_to_f32(std::uint16_t bits) {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
    const std::uint32_t exp = (bits >> 10) & 0x1fu;
    const std::uint32_t mant = bits & 0x3ffu;
    std::uint32_t out = 0;
    if (exp == 0u) {
        if (mant == 0u) {
            out = sign;
        } else {
            std::uint32_t m = mant;
            int e = -14;
            while ((m & 0x400u) == 0u) {
                m <<= 1;
                --e;
            }
            m &= 0x3ffu;
            out = sign | (static_cast<std::uint32_t>(e + 127) << 23) | (m << 13);
        }
    } else if (exp == 31u) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | (static_cast<std::uint32_t>(static_cast<int>(exp) - 15 + 127) << 23) | (mant << 13);
    }
    float value = 0.f;
    std::memcpy(&value, &out, sizeof(value));
    return value;
}

std::uint16_t f32_to_fp16(float value) {
    std::uint32_t x = 0;
    std::memcpy(&x, &value, sizeof(x));
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    const int exponent = static_cast<int>((x >> 23) & 0xffu) - 127 + 15;
    std::uint32_t mantissa = x & 0x7fffffu;
    if ((x & 0x7fffffffu) > 0x7f800000u) return static_cast<std::uint16_t>(sign | 0x7e00u);
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<std::uint16_t>(sign);
        mantissa |= 0x800000u;
        const std::uint32_t shift = static_cast<std::uint32_t>(14 - exponent);
        std::uint32_t half_mant = mantissa >> shift;
        const std::uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const std::uint32_t halfway = 1u << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (half_mant & 1u) != 0u)) ++half_mant;
        return static_cast<std::uint16_t>(sign | half_mant);
    }
    if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    std::uint32_t half_mant = mantissa >> 13;
    const std::uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half_mant & 1u) != 0u)) {
        ++half_mant;
        if (half_mant == 0x400u) {
            if (exponent + 1 >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
            return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent + 1) << 10));
        }
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10) | half_mant);
}

std::uint32_t swap16_lanes(std::uint32_t value) { return (value << 16) | (value >> 16); }

}  // namespace

float mcg_symbol(std::uint32_t symbol) {
    const std::uint32_t mixed = symbol * kExl3McgMultiplier;
    // lop3.b32 dest, dest, 0x8fff8fff, 0x3b603b60, 0x6a == c ^ (a & b)
    const std::uint32_t bits = 0x3b603b60u ^ (mixed & 0x8fff8fffu);
    const float lo = fp16_to_f32(static_cast<std::uint16_t>(bits & 0xffffu));
    const float hi = fp16_to_f32(static_cast<std::uint16_t>(bits >> 16));
    return fp16_to_f32(f32_to_fp16(lo + hi));
}

void exl3_tensor_core_perm(std::uint16_t perm[256]) {
    for (int t = 0; t < 32; ++t) {
        const int r0 = (t % 4) * 2;
        const int c0 = t / 4;
        perm[t * 8 + 0] = static_cast<std::uint16_t>(r0 * 16 + c0);
        perm[t * 8 + 1] = static_cast<std::uint16_t>((r0 + 1) * 16 + c0);
        perm[t * 8 + 2] = static_cast<std::uint16_t>((r0 + 8) * 16 + c0);
        perm[t * 8 + 3] = static_cast<std::uint16_t>((r0 + 9) * 16 + c0);
        perm[t * 8 + 4] = static_cast<std::uint16_t>(r0 * 16 + (c0 + 8));
        perm[t * 8 + 5] = static_cast<std::uint16_t>((r0 + 1) * 16 + (c0 + 8));
        perm[t * 8 + 6] = static_cast<std::uint16_t>((r0 + 8) * 16 + (c0 + 8));
        perm[t * 8 + 7] = static_cast<std::uint16_t>((r0 + 9) * 16 + (c0 + 8));
    }
}

void pack_trellis_k4(const std::uint16_t symbols[256], std::uint16_t packed[kExl3PackedU16]) {
    constexpr int k_bits = kExl3KBits;
    std::uint16_t staged[kExl3PackedU16] = {};
    for (int span = 0; span < 16; ++span) {
        int i = 16 * span;
        int j = k_bits * span;
        int k = 32;
        std::uint32_t buf = 0;
        for (int n = 0; n < 16; ++n) {
            const std::uint32_t v = symbols[i] & ((1u << k_bits) - 1u);
            k -= k_bits;
            buf |= v << k;
            if (k <= 16) {
                staged[j] = static_cast<std::uint16_t>(buf >> 16);
                buf <<= 16;
                k += 16;
                ++j;
            }
            ++i;
        }
    }
    for (int pair = 0; pair < kExl3PackedU16 / 2; ++pair) {
        const std::uint32_t joined = static_cast<std::uint32_t>(staged[pair * 2]) |
                                     (static_cast<std::uint32_t>(staged[pair * 2 + 1]) << 16);
        const std::uint32_t swapped = swap16_lanes(joined);
        packed[pair * 2] = static_cast<std::uint16_t>(swapped & 0xffffu);
        packed[pair * 2 + 1] = static_cast<std::uint16_t>(swapped >> 16);
    }
}

void unpack_trellis_k4(const std::uint16_t packed[kExl3PackedU16], std::uint16_t symbols[256]) {
    std::uint16_t staged[kExl3PackedU16];
    std::uint16_t transitions[256];
    for (int pair = 0; pair < kExl3PackedU16 / 2; ++pair) {
        const std::uint32_t joined = static_cast<std::uint32_t>(packed[pair * 2]) |
                                     (static_cast<std::uint32_t>(packed[pair * 2 + 1]) << 16);
        const std::uint32_t swapped = swap16_lanes(joined);
        staged[pair * 2] = static_cast<std::uint16_t>(swapped & 0xffffu);
        staged[pair * 2 + 1] = static_cast<std::uint16_t>(swapped >> 16);
    }
    for (int span = 0; span < 16; ++span) {
        std::uint64_t chunk = 0;
        for (int word = 0; word < 4; ++word) chunk = (chunk << 16) | staged[span * 4 + word];
        for (int n = 0; n < 16; ++n) {
            transitions[span * 16 + n] = static_cast<std::uint16_t>((chunk >> (60 - 4 * n)) & 0xfu);
        }
    }
    // Every state ends at this transition and includes the preceding 12 bits.
    // Wrap within this tile, never into a neighboring tile or 16-weight span.
    for (int i = 0; i < 256; ++i) {
        std::uint16_t state = 0;
        for (int back = 3; back >= 0; --back)
            state = static_cast<std::uint16_t>((state << 4) | transitions[(i + 256 - back) % 256]);
        symbols[i] = state;
    }
}

void exl3_decode_inner_tile(const std::uint16_t packed[kExl3PackedU16], float tile_kn[256]) {
    std::uint16_t states[256], perm[256];
    unpack_trellis_k4(packed, states);
    exl3_tensor_core_perm(perm);
    for (int pos = 0; pos < 256; ++pos) tile_kn[perm[pos]] = mcg_symbol(states[pos]);
}

void exl3_decode_tile(const std::uint16_t packed[kExl3PackedU16], const float suh[kExl3Tile],
                      const float svh[kExl3Tile], float tile_kn[256]) {
    exl3_decode_inner_tile(packed, tile_kn);
    for (int k = 0; k < kExl3Tile; ++k)
        for (int n = 0; n < kExl3Tile; ++n) tile_kn[k * kExl3Tile + n] *= suh[k] * svh[n];
}

void exl3_gemv_tile(const float x[kExl3Tile], const float tile_kn[kExl3Tile * kExl3Tile],
                    float y[kExl3Tile]) {
    for (int n = 0; n < kExl3Tile; ++n) {
        float sum = 0.f;
        for (int k = 0; k < kExl3Tile; ++k) sum += x[k] * tile_kn[k * kExl3Tile + n];
        y[n] = sum;
    }
}

namespace {
void hadamard128(std::span<double> values) {
    constexpr std::size_t block = kExl3HadamardBlock;
    const double normalization = 1.0 / std::sqrt(static_cast<double>(block));
    for (std::size_t base = 0; base < values.size(); base += block) {
        for (std::size_t stride = 1; stride < block; stride *= 2)
            for (std::size_t j = 0; j < block; j += 2 * stride)
                for (std::size_t i = 0; i < stride; ++i) {
                    const auto a = values[base + j + i], b = values[base + j + i + stride];
                    values[base + j + i] = a + b;
                    values[base + j + i + stride] = a - b;
                }
        for (std::size_t i = 0; i < block; ++i) values[base + i] *= normalization;
    }
}
void finite_half(std::span<const std::uint16_t> values) {
    for (const auto value : values)
        if ((value & 0x7c00u) == 0x7c00u) throw std::invalid_argument("nonfinite EXL3 F16 scale/bias");
}
}  // namespace

std::vector<double> exl3_linear_reference(const Exl3LinearView& m, std::span<const float> input,
                                         std::size_t rows, std::size_t max_elements) {
    const auto k = m.in_features, n = m.out_features;
    if (!k || !n || k % 128 || n % 128 || k > max_elements / n)
        throw std::invalid_argument("EXL3 dimensions must be 128-aligned and within the element budget");
    if (!rows || rows > 32 || input.size() / rows != k || input.size() % rows)
        throw std::invalid_argument("EXL3 input requires 1..32 exact rows");
    if (m.trellis.size() != k * n / 4 || m.suh.size() != k || m.svh.size() != n ||
        (!m.bias.empty() && m.bias.size() != n))
        throw std::invalid_argument("EXL3 storage extent mismatch");
    finite_half(m.suh); finite_half(m.svh); finite_half(m.bias);
    for (float value : input)
        if (!std::isfinite(value)) throw std::invalid_argument("nonfinite EXL3 input");
    std::vector<double> result(rows * n, 0.0), transformed(k);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t i = 0; i < k; ++i)
            transformed[i] = static_cast<double>(input[row * k + i]) * fp16_to_f32(m.suh[i]);
        hadamard128(transformed);
        auto output = std::span(result).subspan(row * n, n);
        for (std::size_t kb = 0; kb < k / 16; ++kb) {
            for (std::size_t nb = 0; nb < n / 16; ++nb) {
                float tile[256];
                exl3_decode_inner_tile(m.trellis.data() + (kb * (n / 16) + nb) * 64, tile);
                for (std::size_t i = 0; i < 16; ++i)
                    for (std::size_t j = 0; j < 16; ++j)
                        output[nb * 16 + j] += transformed[kb * 16 + i] * tile[i * 16 + j];
            }
        }
        hadamard128(output);
        for (std::size_t j = 0; j < n; ++j) {
            output[j] = output[j] * fp16_to_f32(m.svh[j]) + (m.bias.empty() ? 0.0 : fp16_to_f32(m.bias[j]));
            if (!std::isfinite(output[j])) throw std::overflow_error("EXL3 reference output overflow");
        }
    }
    return result;
}

}  // namespace ninfer::glm53
