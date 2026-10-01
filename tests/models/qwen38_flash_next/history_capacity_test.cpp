#include "src/models/qwen38_flash_next/kernels/rocm/history_capacity.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace history = gufo::models::qwen38_flash_next::rocm::history;

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

history::Shape TestShape(bool mtp) {
  return history::Shape{
      .attention_layers = 8,
      .kv_row_bytes = 2048,  // one 1024-element f16 row
      .indexer_head_dim = 128,
      .compress_ratio = 16,
      .mtp = mtp,
  };
}

void CheckByteAccounting() {
  for (const bool mtp : {false, true}) {
    const auto shape = TestShape(mtp);
    // Zero positions cost nothing.
    Require(history::BytesFor(shape, 0) == 0, "empty history must be free");
    const auto one = history::BytesFor(shape, 1);
    // Families: every attention layer holds K+V rows (kv_row * 2 bytes
    // each) plus one pooled-key row; the draft mirrors that with a single
    // K/V pair and one more pooled-key family.
    const std::size_t kv_rows = shape.attention_layers * 2 + (mtp ? 2 : 0);
    const std::size_t block_families = shape.attention_layers + (mtp ? 1 : 0);
    const std::size_t expected_one =
        kv_rows * shape.kv_row_bytes +
        block_families * (shape.indexer_head_dim * 2);
    Require(one == expected_one, "single-position bytes are wrong");
    // Growth is linear in positions; the fixed +1 rows cancel in the
    // differences.
    const auto two = history::BytesFor(shape, 2);
    const auto big_a = history::BytesFor(shape, 100000);
    const auto big_b = history::BytesFor(shape, 200000);
    const auto big_c = history::BytesFor(shape, 300000);
    // Linear across large ranges: 100000 positions cross exactly
    // 100000/compress_ratio pooled-key rows per family.
    const std::size_t per_position =
        kv_rows * shape.kv_row_bytes +
        block_families * shape.indexer_head_dim * 2 / shape.compress_ratio;
    Require(big_b - big_a == 100000 * per_position,
            "history bytes must scale linearly with positions");
    Require(big_b - big_a == big_c - big_b,
            "history bytes must scale linearly with positions");
    // 1 -> 2 crosses no pooled-key boundary (2/16 + 1 == 1 row).
    Require(two - one == kv_rows * shape.kv_row_bytes,
            "the first position adds only K/V rows");
    // The MTP draft mirrors the trunk K/V plus one block family.
    Require(history::BytesFor(TestShape(true), 1000) >
                history::BytesFor(TestShape(false), 1000),
            "speculative sessions must reserve more history");
  }
}

void CheckGrowthSchedule() {
  const std::uint32_t max = 262144;
  // Doubling from the initial capacity.
  Require(history::NextCapacity(32768, 32769, max) == 65536,
          "small overflow must double");
  // A large jump lands exactly on the need, not past it.
  Require(history::NextCapacity(32768, 100000, max) == 100000,
          "big need must be honored exactly");
  // Never below the current capacity.
  Require(history::NextCapacity(65536, 1000, max) == 131072,
          "doubling must hold even for small needs");
  // Never past the model context.
  Require(history::NextCapacity(200000, 250000, max) == max,
          "growth must saturate at the model context");
  Require(history::NextCapacity(max, max, max) == max,
          "saturated capacity stays saturated");
  // Overflow safety: doubling past max must clamp, not wrap.
  Require(history::NextCapacity(200000, 1, max) == max,
          "doubling near max must clamp");
}

void CheckGrowthSequenceCost() {
  // Walking a 150K-token conversation from a 32K start: the schedule must
  // terminate at a capacity >= the need, and the total copied bytes must
  // stay within ~2x the final size (geometric growth amortization).
  const auto shape = TestShape(true);
  std::uint32_t capacity = 32768;
  const std::uint32_t need = 150000;
  std::size_t copied = 0;
  std::size_t steps = 0;
  while (capacity < need) {
    const auto next = history::NextCapacity(capacity, need, 262144);
    Require(next > capacity, "growth must make progress");
    copied += history::BytesFor(shape, capacity);
    capacity = next;
    ++steps;
  }
  Require(capacity >= need, "growth sequence must reach the need");
  Require(steps <= 20, "growth must not loop excessively");
  const auto final_bytes = history::BytesFor(shape, capacity);
  Require(copied <= 2 * final_bytes,
          "geometric growth must amortize copies within ~2x the final size");
}

int main() {
  try {
    CheckByteAccounting();
    CheckGrowthSchedule();
    CheckGrowthSequenceCost();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return 1;
  }
  std::cout << "history_capacity tests passed\n";
  return 0;
}
