#pragma once

#include "ninfer_glm53/exl3_abi.hpp"
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::glm53 {

// MCG consumes a 16-bit trellis STATE, not a 4-bit transition. The sum rounds
// back to fp16, matching the upstream __hadd. Callers pass states in 0..65535.
[[nodiscard]] float mcg_symbol(std::uint32_t state);
void exl3_tensor_core_perm(std::uint16_t perm[256]);

// Pack the low four transition bits of each entry. Arbitrary 16-bit inputs do
// not round-trip: unpack reconstructs overlapping, tail-biting 16-bit windows.
void pack_trellis_k4(const std::uint16_t transitions[256], std::uint16_t packed[kExl3PackedU16]);
void unpack_trellis_k4(const std::uint16_t packed[kExl3PackedU16], std::uint16_t states[256]);

// Inner-basis Q only, in row-major [K,N] order. Not original-basis weights.
void exl3_decode_inner_tile(const std::uint16_t packed[kExl3PackedU16], float tile_kn[256]);

// Legacy scaled INNER-basis diagnostic, not a checkpoint linear. Scales do not
// commute with H128. Do not compose this helper into an original-basis GEMM.
[[deprecated("scaled inner-basis diagnostic; use exl3_linear_reference")]]
void exl3_decode_tile(const std::uint16_t packed[kExl3PackedU16], const float suh[kExl3Tile],
                      const float svh[kExl3Tile], float tile_kn[kExl3Tile * kExl3Tile]);
void exl3_gemv_tile(const float x[kExl3Tile], const float tile_kn[256], float y[kExl3Tile]);

struct Exl3LinearView {
    std::size_t in_features{};
    std::size_t out_features{};
    // Host-endian uint16 words read from little-endian checkpoint storage.
    // trellis: [K/16,N/16,64]; scales and optional bias contain F16 bit patterns.
    std::span<const std::uint16_t> trellis;
    std::span<const std::uint16_t> suh;
    std::span<const std::uint16_t> svh;
    std::span<const std::uint16_t> bias{};
};

// CPU correctness reference: x D_suh H128 Q H128 D_svh + bias.
// Requires K,N multiples of 128 and 1..32 rows. Invalid extents/nonfinite inputs
// throw before returning any output. Decodes one tile at a time, not a full
// dense weight replica. No CUDA, TP2, checkpoint authentication or speed claim.
[[nodiscard]] std::vector<double> exl3_linear_reference(
    const Exl3LinearView& matrix, std::span<const float> input, std::size_t rows = 1,
    std::size_t max_elements = 16U * 1024U * 1024U);

}  // namespace ninfer::glm53
