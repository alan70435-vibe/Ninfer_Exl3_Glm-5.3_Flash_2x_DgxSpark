#pragma once
#include <cstdint>

namespace ninfer::glm53 {
// The only definition shared by checkpoint binding and host decoding.
inline constexpr std::uint32_t kExl3McgMultiplier = 0xCBAC1FEDu;
inline constexpr int kExl3Tile = 16;
inline constexpr int kExl3KBits = 4;
inline constexpr int kExl3PackedU16 = 64;
inline constexpr int kExl3HadamardBlock = 128;
}  // namespace ninfer::glm53
