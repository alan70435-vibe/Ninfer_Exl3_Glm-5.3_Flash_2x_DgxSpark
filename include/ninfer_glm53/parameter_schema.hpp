#pragma once

#include "ninfer_glm53/model_spec.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::glm53 {

// Safetensors dtype names used by the EXL3/TR3 and DFlash2 checkpoints.
enum class DType : std::uint8_t {
    kBf16,
    kF16,
    kF32,
    kI16,
    kI32,
};

enum class StorageClass : std::uint8_t {
    kNativeBf16,
    kNativeF32,
    kExl3ScaleF16,
    kExl3TrellisI16,
    kExl3McgI32,
};

// ExLlamaV3 MCG multiplier stored in every routed-expert `.mcg` tensor.
// Measured on the TR3 4 bpw checkpoint and recorded in exl3-mcg-storage-abi.json.
inline constexpr std::uint32_t kExl3McgMultiplier = 0xCBAC1FEDu;

struct VisionSpec {
    std::uint32_t depth{};
    std::uint32_t hidden_size{};
    std::uint32_t num_heads{};
    std::uint32_t intermediate_size{};
    std::uint32_t out_hidden_size{};
    std::uint32_t in_channels{};
    std::uint32_t patch_size{};
    std::uint32_t temporal_patch_size{};
    std::uint32_t spatial_merge_size{};
    std::uint32_t projection_intermediate_size{};
};

struct DFlash2Spec {
    std::uint32_t layers{};
    std::uint32_t hidden_size{};
    std::uint32_t intermediate_size{};
    std::uint32_t vocab_size{};
    std::uint32_t num_heads{};
    std::uint32_t num_kv_heads{};
    std::uint32_t head_dim{};
    std::uint32_t selector_rank{};
    std::uint32_t conv_kernel_size{};
    std::uint32_t conv_group_size{};
    std::uint32_t block_size{};
    std::uint32_t num_target_layers{};
    std::array<std::uint32_t, 5> target_layer_ids{};
};

// One stored tensor. logical_id is the runtime name. source_name is the
// checkpoint key. Shapes are the safetensors storage shapes, not a decoded
// GEMM view: EXL3 trellis packing is still opaque at this layer.
struct ExpectedTensor {
    std::string logical_id;
    std::string source_name;
    DType dtype{DType::kBf16};
    StorageClass storage{StorageClass::kNativeBf16};
    std::vector<std::int64_t> shape;
    bool mcg_payload{false};
};

[[nodiscard]] const VisionSpec& glm53_flash_vision_spec();
[[nodiscard]] const DFlash2Spec& glm53_flash_dflash2_spec();
[[nodiscard]] std::vector<std::string> validate_vision_spec(const VisionSpec& spec);
[[nodiscard]] std::vector<std::string> validate_dflash2_spec(const DFlash2Spec& spec);

[[nodiscard]] std::vector<ExpectedTensor> expected_glm53_flash_tensors(
    const ModelSpec& model, const VisionSpec& vision);
[[nodiscard]] std::vector<ExpectedTensor> expected_dflash2_tensors(const DFlash2Spec& spec);
[[nodiscard]] std::vector<ExpectedTensor> expected_glm53_flash_tensors();
[[nodiscard]] std::vector<ExpectedTensor> expected_dflash2_tensors();

[[nodiscard]] std::vector<std::string> validate_parameter_catalog(const std::vector<ExpectedTensor>& catalog);
[[nodiscard]] std::string logical_catalog_sha256(const std::vector<ExpectedTensor>& catalog);

[[nodiscard]] std::string_view to_string(DType dtype) noexcept;
[[nodiscard]] std::string_view to_string(StorageClass storage) noexcept;
[[nodiscard]] std::string shape_to_string(const std::vector<std::int64_t>& shape);

}  // namespace ninfer::glm53
