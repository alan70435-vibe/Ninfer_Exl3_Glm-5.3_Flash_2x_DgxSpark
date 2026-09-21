#pragma once

#include <cstdint>

namespace ninfer::glm53 {

inline constexpr std::uint32_t kExl3McgMultiplier = 0xCBAC1FEDu;
inline constexpr int kExl3Tile = 16;
inline constexpr int kExl3KBits = 4;
inline constexpr int kExl3PackedU16 = 64;

// MCG codebook entry. The sum is rounded back to fp16, matching ExLlamaV3 __hadd.
[[nodiscard]] float mcg_symbol(std::uint32_t symbol);

// Tensor-core order: perm[kernel_pos] is the row-major index k*16+n.
void exl3_tensor_core_perm(std::uint16_t perm[256]);

void pack_trellis_k4(const std::uint16_t symbols[256], std::uint16_t packed[kExl3PackedU16]);
void unpack_trellis_k4(const std::uint16_t packed[kExl3PackedU16], std::uint16_t symbols[256]);

// One 16x16 expert tile. tile_kn[k * 16 + n] = codebook(symbol) * suh[k] * svh[n].
void exl3_decode_tile(const std::uint16_t packed[kExl3PackedU16], const float suh[kExl3Tile],
                      const float svh[kExl3Tile], float tile_kn[kExl3Tile * kExl3Tile]);

// y[n] = sum_k x[k] * tile_kn[k * 16 + n]
void exl3_gemv_tile(const float x[kExl3Tile], const float tile_kn[kExl3Tile * kExl3Tile],
                    float y[kExl3Tile]);

}  // namespace ninfer::glm53
