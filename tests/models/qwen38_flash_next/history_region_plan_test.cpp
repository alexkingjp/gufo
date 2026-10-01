#include "src/models/qwen38_flash_next/kernels/rocm/history_region_plan.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace history = gufo::models::qwen38_flash_next::rocm::history;

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

history::Shape Shape(std::size_t layers, std::size_t kv_row_bytes,
                     std::size_t head_dim, std::uint32_t ratio, bool mtp) {
  return history::Shape{
      .attention_layers = layers,
      .kv_row_bytes = kv_row_bytes,
      .indexer_head_dim = head_dim,
      .compress_ratio = ratio,
      .mtp = mtp,
  };
}

/// The plan's total must equal the executor's byte accounting for every
/// shape and position count — the invariant the stage-2 integration relies
/// on (regions replace exactly the buffers BytesFor counted).
void CheckPlanMatchesByteAccounting() {
  const std::size_t layer_counts[] = {1, 4, 8, 48};
  const std::size_t kv_rows[] = {256, 1024, 2048, 4096};
  const std::size_t head_dims[] = {64, 128, 256};
  const std::uint32_t ratios[] = {8, 16, 32, 64};
  const std::uint32_t position_counts[] = {1,    2,     15,     16,    17,
                                           2048, 32768, 131072, 262144};
  for (const std::size_t layers : layer_counts) {
    for (const std::size_t kv_row : kv_rows) {
      for (const std::size_t head_dim : head_dims) {
        for (const std::uint32_t ratio : ratios) {
          for (const bool mtp : {false, true}) {
            const auto shape = Shape(layers, kv_row, head_dim, ratio, mtp);
            for (const std::uint32_t positions : position_counts) {
              const auto plan = history::RegionPlanFor(shape, positions);
              Require(history::TotalRegionBytes(plan) ==
                          history::BytesFor(shape, positions),
                      "plan total must equal BytesFor");
              const std::size_t expected_regions = layers * 3 + (mtp ? 3 : 0);
              Require(plan.size() == expected_regions,
                      "region count is layer-major K/V/block plus MTP");
            }
          }
        }
      }
    }
  }
}

void CheckPerRegionArithmetic() {
  const auto shape = Shape(4, 2048, 128, 16, true);
  const std::uint32_t positions = 32768;
  const auto plan = history::RegionPlanFor(shape, positions);
  const std::size_t kv_bytes = std::size_t{positions} * 2048;
  const std::size_t block_bytes = (std::size_t{positions} / 16 + 1) * 128 * 2;
  std::size_t trunk_k = 0, trunk_v = 0, trunk_block = 0;
  std::size_t mtp_k = 0, mtp_v = 0, mtp_block = 0;
  for (const auto& spec : plan) {
    using F = history::RegionSpec::Family;
    switch (spec.family) {
      case F::kTrunkK:
        Require(spec.layer < 4, "trunk layer ordinal in range");
        trunk_k += spec.bytes;
        break;
      case F::kTrunkV:
        trunk_v += spec.bytes;
        break;
      case F::kTrunkBlockKeys:
        trunk_block += spec.bytes;
        break;
      case F::kMtpK:
        mtp_k += spec.bytes;
        break;
      case F::kMtpV:
        mtp_v += spec.bytes;
        break;
      case F::kMtpBlockKeys:
        mtp_block += spec.bytes;
        break;
    }
  }
  Require(trunk_k == 4 * kv_bytes, "trunk K total");
  Require(trunk_v == 4 * kv_bytes, "trunk V total");
  Require(trunk_block == 4 * block_bytes, "trunk block keys total");
  Require(mtp_k == kv_bytes && mtp_v == kv_bytes, "draft K/V total");
  Require(mtp_block == block_bytes, "draft block keys total");
  Require(plan.front().family == history::RegionSpec::Family::kTrunkK &&
              plan.front().layer == 0,
          "layout starts at trunk layer 0 K");
  Require(plan.back().family == history::RegionSpec::Family::kMtpBlockKeys,
          "layout ends at the draft block keys");
}

void CheckEmptyPlan() {
  const auto shape = Shape(4, 2048, 128, 16, true);
  Require(history::RegionPlanFor(shape, 0).empty(),
          "zero positions need no regions");
  Require(history::TotalRegionBytes({}) == 0, "empty plan totals zero");
}

int main() {
  CheckPlanMatchesByteAccounting();
  CheckPerRegionArithmetic();
  CheckEmptyPlan();
  std::cout << "history_region_plan tests passed\n";
  return 0;
}
