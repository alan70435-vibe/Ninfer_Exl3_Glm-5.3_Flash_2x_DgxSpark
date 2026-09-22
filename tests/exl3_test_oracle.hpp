#pragma once
// Independent scalar oracle: raw-bit windows, analytic binary16 rounding and
// dense Sylvester H128 multiplication, not the implementation's packer/FHT.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

inline double oracle_half(std::uint16_t bits) {
    const auto e = (bits >> 10) & 31, f = bits & 1023;
    const auto a = e ? std::ldexp(double(1024 + f), int(e) - 25) : std::ldexp(double(f), -24);
    return bits & 0x8000 ? -a : a;
}
inline double oracle_mcg(std::uint16_t state) {
    const std::uint32_t mixed = (std::uint32_t(state) * 0xcbac1fedu & 0x8fff8fffu) ^ 0x3b603b60u;
    const double sum = oracle_half(static_cast<std::uint16_t>(mixed)) + oracle_half(static_cast<std::uint16_t>(mixed >> 16));
    if (sum == 0) return 0;
    const double unit = std::ldexp(1.0, std::max(-24, std::ilogb(std::fabs(sum)) - 10));
    const double units = std::fabs(sum) / unit, low = std::floor(units), rest = units - low;
    const bool up = rest > 0.5 || (rest == 0.5 && std::fmod(low, 2) != 0);
    return std::copysign((low + (up ? 1.0 : 0.0)) * unit, sum);
}
inline std::uint16_t oracle_state(const std::uint16_t* packed, unsigned position) {
    std::uint16_t result = 0;
    for (unsigned bit = 0; bit < 16; ++bit) {
        const auto offset = (4 * (position + 1) + 1024 - 16 + bit) % 1024;
        const auto word = std::uint32_t(packed[2 * (offset / 32)]) |
                          (std::uint32_t(packed[2 * (offset / 32) + 1]) << 16);
        result = static_cast<std::uint16_t>((std::uint32_t(result) << 1) | ((word >> (31 - offset % 32)) & 1));
    }
    return result;
}
inline unsigned oracle_order(unsigned row, unsigned col) {
    return (row & 1) | ((row & 8) >> 2) | ((col & 8) >> 1) | ((row & 6) << 2) | ((col & 7) << 5);
}
inline std::vector<double> oracle_h128(std::span<const double> x) {
    std::vector<double> y(x.size());
    for (std::size_t base = 0; base < x.size(); base += 128)
        for (unsigned i = 0; i < 128; ++i)
            for (unsigned j = 0; j < 128; ++j)
                y[base+i] += ((std::popcount(i & j) & 1) ? -x[base+j] : x[base+j]) / std::sqrt(128.0);
    return y;
}
