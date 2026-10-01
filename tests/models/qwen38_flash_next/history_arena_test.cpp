#include "src/models/qwen38_flash_next/kernels/rocm/history_arena.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace history = gufo::models::qwen38_flash_next::rocm::history;

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

/// Counts every device call so the tests can prove the arena never
/// allocates or frees outside construction and destruction.
class CountingDevice final : public history::ArenaDevice {
public:
  // Honors the ArenaDevice contract the production hipMalloc wrapper
  // provides: block bases are at least the arena's region alignment.
  [[nodiscard]] void* Allocate(std::size_t bytes) override {
    allocations_ += 1;
    void* raw = std::malloc(bytes + kAlign);
    if (raw == nullptr)
      return nullptr;
    void* aligned = raw;
    std::size_t space = bytes + kAlign;
    if (!std::align(kAlign, bytes, aligned, space)) {
      std::free(raw);
      return nullptr;
    }
    originals_[aligned] = raw;
    return aligned;
  }

  void Free(void* ptr) noexcept override {
    frees_ += 1;
    const auto it = originals_.find(ptr);
    if (it != originals_.end()) {
      std::free(it->second);
      originals_.erase(it);
    }
  }

  [[nodiscard]] std::size_t allocations() const noexcept {
    return allocations_;
  }
  [[nodiscard]] std::size_t frees() const noexcept { return frees_; }
  [[nodiscard]] bool leaked() const noexcept { return !originals_.empty(); }

private:
  static constexpr std::size_t kAlign = 256;
  std::size_t allocations_{0};
  std::size_t frees_{0};
  std::map<void*, void*> originals_;
};

history::Shape TestShape(bool mtp) {
  return history::Shape{
      .attention_layers = 8,
      .kv_row_bytes = 2048,
      .indexer_head_dim = 128,
      .compress_ratio = 16,
      .mtp = mtp,
  };
}

void CheckConstructionAndZeroRuntimeGrowth() {
  CountingDevice device;
  {
    history::HistoryArena arena({.budget_bytes = 8 * 1024 * 1024,
                                 .block_bytes = 1024 * 1024,
                                 .region_alignment = 256},
                                device);
    const auto stats = arena.stats();
    Require(stats.budget_bytes == 8 * 1024 * 1024, "effective budget");
    Require(stats.blocks == 8, "block count");
    Require(stats.device_allocations == 8, "one allocation per block");
    Require(stats.free_bytes == stats.budget_bytes, "pool starts free");
    Require(stats.live_regions == 0 && stats.live_bytes == 0,
            "pool starts live-empty");
    Require(stats.largest_free_bytes == 1024 * 1024, "largest span is a block");
    Require(device.allocations() == 8, "device saw exactly the pool");

    {
      auto a = arena.Reserve(4096);
      auto b = arena.Reserve(4096);
      auto c = arena.Reserve(1024 * 1024);  // exactly one block
      Require(a && b && c, "reserves succeed inside budget");
      const auto mid = arena.stats();
      Require(mid.live_regions == 3, "live regions tracked");
      Require(mid.live_bytes == 4096 + 4096 + 1024 * 1024,
              "live bytes tracked");
      Require(device.allocations() == 8, "no runtime device allocation");
    }
    Require(device.allocations() == 8, "release did not allocate");
    Require(device.frees() == 0, "release did not free device memory");
    const auto after = arena.stats();
    Require(after.live_regions == 0, "regions returned");
    Require(after.free_bytes == after.budget_bytes, "pool fully reusable");
    Require(after.largest_free_bytes == 1024 * 1024, "blocks coalesced whole");
  }
  Require(!device.leaked(), "no device leak");
}

void CheckExhaustionFailsClosed() {
  CountingDevice device;
  {
    history::HistoryArena arena({.budget_bytes = 2 * 1024 * 1024,
                                 .block_bytes = 1024 * 1024,
                                 .region_alignment = 256},
                                device);
    auto a = arena.Reserve(1024 * 1024);
    auto b = arena.Reserve(1024 * 1024);
    Require(a && b, "two block-sized regions fill the pool");
    bool exhausted = false;
    try {
      std::ignore = arena.Reserve(1);
    } catch (const history::ArenaExhausted&) {
      exhausted = true;
    }
    Require(exhausted, "a full pool exhausts even for one byte");

    b.reset();
    auto again = arena.TryReserve(1024 * 1024);
    Require(again != nullptr,
            "released block serves the next block-sized region");
    bool over_budget = false;
    try {
      std::ignore = arena.Reserve(1024 * 1024 + 1);
    } catch (const history::ArenaExhausted&) {
      over_budget = true;
    }
    Require(over_budget, "over-budget request fails closed");
    const auto stats = arena.stats();
    Require(stats.dropped_bytes == 0, "no dropped bytes on the normal path");
  }
  Require(!device.leaked(), "no device leak");
}

void CheckRegionReuseAndCoalescing() {
  CountingDevice device;
  {
    history::HistoryArena arena({.budget_bytes = 1024 * 1024,
                                 .block_bytes = 1024 * 1024,
                                 .region_alignment = 256},
                                device);
    auto r1 = arena.Reserve(204800);
    auto r2 = arena.Reserve(204800);
    auto r3 = arena.Reserve(204800);
    auto r4 = arena.Reserve(204800);
    auto r5 = arena.Reserve(204800);
    Require(r1 && r2 && r3 && r4 && r5, "five packed regions");
    const std::uintptr_t a_addr =
        reinterpret_cast<std::uintptr_t>(r1->device_ptr());
    const std::uintptr_t b_addr =
        reinterpret_cast<std::uintptr_t>(r2->device_ptr());
    Require(a_addr % 256 == 0 && b_addr % 256 == 0, "alignment honored");
    Require(b_addr - a_addr == 204800, "regions packed in order");
    const auto mid = arena.stats();
    Require(mid.free_bytes == 1024 * 1024 - 5 * 204800,
            "carved bytes leave the pool");

    r2.reset();
    r4.reset();  // two non-adjacent holes
    bool fragmented_refusal = arena.TryReserve(2 * 204800 + 256) == nullptr;
    Require(fragmented_refusal,
            "fragmented pool refuses a region larger than any hole");
    auto hole = arena.TryReserve(204800);
    Require(hole != nullptr, "first hole reused");
    auto hole2 = arena.TryReserve(204800);
    Require(hole2 != nullptr, "second hole reused");
    bool tail_refusal = arena.TryReserve(204800) == nullptr;
    Require(tail_refusal, "only the small tail remains");

    hole.reset();
    hole2.reset();
    r1.reset();
    r3.reset();
    r5.reset();
    const auto after = arena.stats();
    Require(after.free_bytes == 1024 * 1024, "coalescing restored the block");
    Require(after.largest_free_bytes == 1024 * 1024, "whole block again");

    auto reused = arena.Reserve(1024 * 1024);
    Require(reused != nullptr, "whole block serves a full-block region again");
  }
  Require(!device.leaked(), "no device leak");
}

void CheckInvalidOptionsAndSizes() {
  CountingDevice device;
  bool invalid = false;
  try {
    std::ignore = history::HistoryArena(
        {.budget_bytes = 1024, .block_bytes = 0, .region_alignment = 256},
        device);
  } catch (const std::invalid_argument&) {
    invalid = true;
  }
  Require(invalid, "zero block size rejected");
  invalid = false;
  try {
    std::ignore = history::HistoryArena(
        {.budget_bytes = 1024, .block_bytes = 4096, .region_alignment = 256},
        device);
  } catch (const std::invalid_argument&) {
    invalid = true;
  }
  Require(invalid, "budget below one block rejected");
  invalid = false;
  try {
    std::ignore = history::HistoryArena({.budget_bytes = 1024 * 1024,
                                         .block_bytes = 1024 * 1024,
                                         .region_alignment = 100},
                                        device);
  } catch (const std::invalid_argument&) {
    invalid = true;
  }
  Require(invalid, "non-power-of-two alignment rejected");

  {
    history::HistoryArena arena({.budget_bytes = 1024 * 1024,
                                 .block_bytes = 1024 * 1024,
                                 .region_alignment = 256},
                                device);
    Require(!arena.TryReserve(0), "zero-byte reserve is null, not a region");
    bool thrown = false;
    try {
      std::ignore = arena.Reserve(0);
    } catch (const std::invalid_argument&) {
      thrown = true;
    }
    Require(thrown, "zero-byte Reserve throws invalid_argument");
  }
  Require(!device.leaked(), "no device leak");
}

void CheckPartialConstructionFailsClosed() {
  class RefusingDevice final : public history::ArenaDevice {
  public:
    [[nodiscard]] void* Allocate(std::size_t bytes) override {
      attempts_ += 1;
      if (attempts_ > 2)
        return nullptr;  // device runs out on the third block
      return std::malloc(bytes);
    }
    void Free(void* ptr) noexcept override { std::free(ptr); }
    [[nodiscard]] std::size_t attempts() const noexcept { return attempts_; }
    [[nodiscard]] bool leaked() const noexcept { return !live_.empty(); }

  private:
    std::size_t attempts_{0};
    std::set<void*> live_;
  };
  RefusingDevice device;
  bool failed = false;
  try {
    std::ignore = history::HistoryArena({.budget_bytes = 4 * 1024 * 1024,
                                         .block_bytes = 1024 * 1024,
                                         .region_alignment = 256},
                                        device);
  } catch (const std::runtime_error&) {
    failed = true;
  }
  Require(failed, "partial pool construction throws");
  Require(device.attempts() == 3, "construction stopped at the refusal");
  Require(!device.leaked(), "partial pool was released");
}

void CheckAccountingAndLifetime() {
  CountingDevice device;
  auto arena = std::make_shared<history::HistoryArena>(
      history::ArenaOptions{.budget_bytes = 4 * 1024 * 1024,
                            .block_bytes = 1024 * 1024,
                            .region_alignment = 256},
      device);

  const auto shape = TestShape(true);
  const std::uint32_t positions = 96;  // session = 3,555,072 bytes < 4 MiB
  const std::size_t session =
      history::HistoryArena::SessionBytes(shape, positions);
  Require(session == history::BytesFor(shape, positions),
          "SessionBytes delegates to BytesFor");
  Require(session == 36864 * positions + 2304 * (positions / 16 + 1),
          "session bytes match the family arithmetic");
  Require(
      history::HistoryArena::MaxPositions(shape, session, 4096) == positions,
      "MaxPositions inverts exactly at the boundary");
  Require(
      history::HistoryArena::MaxPositions(shape, session - 1, 4096) < positions,
      "MaxPositions is strict below the boundary");
  Require(history::HistoryArena::MaxPositions(shape, session, 0) == 0,
          "MaxPositions respects the context cap");

  // Place the session as per-family regions and check the accounting.
  std::vector<std::shared_ptr<history::HistoryArena::Region>> regions;
  std::size_t placed = 0;
  while (placed < session) {
    const std::size_t want =
        std::min<std::size_t>(session - placed, 1024 * 1024);
    auto region = arena->Reserve(want);
    Require(region->bytes() == want, "aligned wants round to themselves");
    placed += region->bytes();
    regions.push_back(std::move(region));
  }
  const auto full = arena->stats();
  Require(full.live_bytes == session, "session occupies exactly its bytes");
  Require(full.free_bytes == full.budget_bytes - session,
          "remainder stays free");

  // Regions outliving the arena handle keep device memory valid; the device
  // frees only when everything is gone.
  regions.clear();
  arena.reset();
  Require(device.frees() == 4, "blocks freed once when everything is gone");
  Require(device.allocations() == 4, "no hidden allocations");
  Require(!device.leaked(), "no device leak");
}

int main() {
  for (int step = 0; step < 6; ++step) {
    try {
      switch (step) {
        case 0:
          CheckConstructionAndZeroRuntimeGrowth();
          break;
        case 1:
          CheckExhaustionFailsClosed();
          break;
        case 2:
          CheckRegionReuseAndCoalescing();
          break;
        case 3:
          CheckInvalidOptionsAndSizes();
          break;
        case 4:
          CheckPartialConstructionFailsClosed();
          break;
        case 5:
          CheckAccountingAndLifetime();
          break;
      }
    } catch (const std::exception& e) {
      std::cout << "FAILED at step " << step << ": " << e.what() << '\n';
      return 1;
    }
    std::cout << "step " << step << " ok\n";
  }
  std::cout << "history_arena tests passed\n";
  return 0;
}
