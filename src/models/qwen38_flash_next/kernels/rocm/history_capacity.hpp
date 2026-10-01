#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_ROCM_HISTORY_CAPACITY_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_ROCM_HISTORY_CAPACITY_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace gufo::models::qwen38_flash_next::rocm::history {

/// The geometry the position-scaled history families depend on. Plain data
/// so the sizing policy stays CPU-testable without a device.
struct Shape {
  std::size_t attention_layers;
  /// Bytes of one K (or V) cache row per position: kv_width × 2 in f16
  /// mode, kv_quant::RowBytes(kv_width) when the caches hold packed q8_0.
  std::size_t kv_row_bytes;
  std::size_t indexer_head_dim;
  std::uint32_t compress_ratio;
  bool mtp;
};

/// Device bytes of the position-scaled history families (per attention
/// layer K/V and pooled block keys, plus the MTP draft's own K/V and block
/// keys for speculative sessions) at `positions` rows. The indexer raw
/// ring is bounded by top-k and the batch, not the context, so it stays
/// fixed and is excluded. Linear in `positions` up to the single +1 block
/// row per family.
[[nodiscard]] inline std::size_t BytesFor(const Shape& shape,
                                          std::uint32_t positions) noexcept {
  if (positions == 0)
    return 0;
  const std::size_t kv = std::size_t{positions} * shape.kv_row_bytes;
  const std::size_t block =
      (std::size_t{positions} / shape.compress_ratio + 1) *
      shape.indexer_head_dim * 2;
  const std::size_t trunk = shape.attention_layers * (2 * kv + block);
  const std::size_t draft = shape.mtp ? 2 * kv + block : 0;
  return trunk + draft;
}

/// Elastic growth schedule: double from the current capacity, but never
/// below what the caller needs, and never past the model context.
[[nodiscard]] inline std::uint32_t NextCapacity(std::uint32_t current,
                                                std::uint32_t needed,
                                                std::uint32_t max) noexcept {
  const std::uint32_t doubled =
      current <= max / 2 ? current * 2 : max;
  return std::min(max, std::max(doubled, needed));
}

}  // namespace gufo::models::qwen38_flash_next::rocm::history

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_ROCM_HISTORY_CAPACITY_HPP_
