#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_ROCM_HISTORY_REGION_PLAN_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_ROCM_HISTORY_REGION_PLAN_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/history_capacity.hpp"

namespace gufo::models::qwen38_flash_next::rocm::history {

/// One position-scaled history-family buffer of a session. The K/V regions
/// hold the packed q8_0 stores in quantized mode and the f16 caches
/// otherwise; Shape::kv_row_bytes already accounts for the mode, so the
/// plan is mode-agnostic.
struct RegionSpec {
  enum class Family : std::uint8_t {
    kTrunkK,
    kTrunkV,
    kTrunkBlockKeys,
    kMtpK,
    kMtpV,
    kMtpBlockKeys,
  };
  Family family;
  /// Attention-layer ordinal (0..Shape::attention_layers-1). MTP regions
  /// report 0; the executor maps ordinals onto its real layer indices.
  std::size_t layer;
  /// Device bytes of the region at the planned position count.
  std::size_t bytes;
};

/// Enumerates the position-scaled regions one session needs, in the layout
/// order the executor fills them: layer-major [K, V, block keys] over the
/// attention layers, then the MTP draft's [K, V, block keys] when the shape
/// is speculative. Linear layers carry no position-scaled families and are
/// excluded by construction (Shape::attention_layers counts only them).
///
/// Invariant: the sum of region bytes equals `history::BytesFor(shape,
/// positions)` for every shape and position count — the executor's byte
/// accounting and this plan cannot drift apart.
[[nodiscard]] inline std::vector<RegionSpec> RegionPlanFor(
    const Shape& shape, std::uint32_t positions) {
  std::vector<RegionSpec> plan;
  if (positions == 0)
    return plan;
  const std::size_t kv_bytes = std::size_t{positions} * shape.kv_row_bytes;
  const std::size_t block_bytes =
      (std::size_t{positions} / shape.compress_ratio + 1) *
      shape.indexer_head_dim * 2;
  plan.reserve(std::size_t{shape.attention_layers} * 3 + (shape.mtp ? 3 : 0));
  for (std::size_t layer = 0; layer < shape.attention_layers; ++layer) {
    plan.push_back({RegionSpec::Family::kTrunkK, layer, kv_bytes});
    plan.push_back({RegionSpec::Family::kTrunkV, layer, kv_bytes});
    plan.push_back({RegionSpec::Family::kTrunkBlockKeys, layer, block_bytes});
  }
  if (shape.mtp) {
    plan.push_back({RegionSpec::Family::kMtpK, 0, kv_bytes});
    plan.push_back({RegionSpec::Family::kMtpV, 0, kv_bytes});
    plan.push_back({RegionSpec::Family::kMtpBlockKeys, 0, block_bytes});
  }
  return plan;
}

/// Sum of region bytes; equals BytesFor(shape, positions) for the plan's
/// inputs.
[[nodiscard]] inline std::size_t TotalRegionBytes(
    const std::vector<RegionSpec>& plan) noexcept {
  std::size_t total = 0;
  for (const RegionSpec& spec : plan) {
    total += spec.bytes;
  }
  return total;
}

}  // namespace gufo::models::qwen38_flash_next::rocm::history

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_ROCM_HISTORY_REGION_PLAN_HPP_
