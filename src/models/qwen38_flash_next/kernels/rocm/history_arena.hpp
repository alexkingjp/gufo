#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_ROCM_HISTORY_ARENA_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_ROCM_HISTORY_ARENA_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/history_capacity.hpp"

namespace gufo::models::qwen38_flash_next::rocm::history {

/// Device memory source for the shared history arena. Production wraps
/// hipMalloc/hipFree; CPU tests substitute a host fake. Allocate returns
/// nullptr on failure so every path can fail closed.
class ArenaDevice {
public:
  virtual ~ArenaDevice() = default;
  [[nodiscard]] virtual void* Allocate(std::size_t bytes) = 0;
  virtual void Free(void* ptr) noexcept = 0;
};

struct ArenaOptions {
  /// Total pool. Rounded down to a whole number of blocks at construction;
  /// the effective budget is reported in stats.
  std::size_t budget_bytes{0};
  /// Per-block granularity. Regions never span blocks, so this bounds the
  /// largest single region (one history-family buffer) and keeps the free
  /// lists small.
  std::size_t block_bytes{0};
  std::size_t region_alignment{256};
};

/// Thrown by Reserve when no free span fits. Callers must treat this as a
/// request-level failure, never process level.
class ArenaExhausted : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

/// Bounded, preallocated, refcounted region pool for session history
/// families (P1 shared device pool, stage 1).
///
/// Every block is allocated once at construction; serving never calls the
/// device allocator again (zero runtime growth — the elastic-growth fault
/// trigger). Sessions and retained cache entries hold Region handles via
/// shared_ptr; releasing the last handle returns the span to its block's
/// free list (coalesced) without any device call. Exhaustion fails closed
/// at reserve time instead of growing.
///
/// All methods are thread-safe. Regions share ownership of the arena
/// implementation, so device memory stays valid while any region lives and
/// is freed exactly once when the arena and all regions are gone. The
/// ArenaDevice must outlive that point.
class HistoryArena {
private:
  struct Impl;

public:
  class Region {
  public:
    ~Region() {
      if (impl_)
        impl_->Release(block_, offset_, bytes_);
    }
    Region(const Region&) = delete;
    Region& operator=(const Region&) = delete;

    [[nodiscard]] void* device_ptr() const noexcept { return device_ptr_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

  private:
    friend class HistoryArena;
    friend struct Impl;
    Region(std::shared_ptr<Impl> impl, std::size_t block, std::size_t offset,
           std::size_t bytes, void* device_ptr)
        : impl_(std::move(impl)),
          block_(block),
          offset_(offset),
          bytes_(bytes),
          device_ptr_(device_ptr) {}
    std::shared_ptr<Impl> impl_;
    std::size_t block_;
    std::size_t offset_;
    std::size_t bytes_;
    void* device_ptr_;
  };

  /// Allocates the whole pool up front. Throws std::invalid_argument for
  /// nonsensical options and std::runtime_error when the device cannot
  /// provide the blocks (no partial pool is retained).
  explicit HistoryArena(ArenaOptions options, ArenaDevice& device)
      : impl_(std::make_shared<Impl>(options, device)) {}

  /// Throws ArenaExhausted when no span fits; invalid sizes throw
  /// std::invalid_argument.
  [[nodiscard]] std::shared_ptr<Region> Reserve(std::size_t bytes) {
    if (bytes == 0)
      throw std::invalid_argument("history arena regions need bytes > 0");
    auto region = impl_->TryReserve(bytes);
    if (!region)
      throw ArenaExhausted("history arena has no free span for " +
                           std::to_string(bytes) + " bytes");
    return region;
  }

  /// Null when the request is invalid or no free span fits. Never allocates
  /// device memory.
  [[nodiscard]] std::shared_ptr<Region> TryReserve(std::size_t bytes) {
    return impl_->TryReserve(bytes);
  }

  struct Stats {
    std::size_t budget_bytes;
    std::size_t block_bytes;
    std::size_t blocks;
    std::size_t live_regions;
    std::size_t live_bytes;
    std::size_t free_bytes;
    std::size_t largest_free_bytes;
    std::size_t device_allocations;
    /// Bytes from regions whose release could not update the free list
    /// (allocation failure inside a destructor). Zero in every normal path;
    /// nonzero means the reusable pool silently shrank.
    std::size_t dropped_bytes;
  };
  [[nodiscard]] Stats stats() const { return impl_->MakeStats(); }

  /// Device bytes of one session's position-scaled history families
  /// (`history::BytesFor`). Exposed on the arena so sizing decisions and
  /// their tests share one accounting path.
  [[nodiscard]] static std::size_t SessionBytes(
      const Shape& shape, std::uint32_t positions) noexcept {
    return BytesFor(shape, positions);
  }

  /// Largest position count whose session fits `byte_budget`, capped at
  /// `max_positions`. BytesFor is strictly monotone in positions for
  /// positions >= 1, so a binary search is exact.
  [[nodiscard]] static std::uint32_t MaxPositions(
      const Shape& shape, std::size_t byte_budget,
      std::uint32_t max_positions) noexcept {
    std::uint32_t low = 0;
    std::uint32_t high = max_positions;
    while (low < high) {
      const std::uint64_t mid =
          (std::uint64_t{low} + std::uint64_t{high} + 1) / 2;
      if (SessionBytes(shape, static_cast<std::uint32_t>(mid)) <= byte_budget)
        low = static_cast<std::uint32_t>(mid);
      else
        high = static_cast<std::uint32_t>(mid - 1);
    }
    return low;
  }

private:
  struct FreeSpan {
    std::size_t offset;
    std::size_t bytes;
  };

  struct Block {
    void* device_ptr{nullptr};
    std::vector<FreeSpan> free_spans;  // sorted by offset, non-adjacent
  };

  struct Impl : std::enable_shared_from_this<Impl> {
    Impl(ArenaOptions options, ArenaDevice& device_ref)
        : options_(options), device_(device_ref) {
      if (options_.block_bytes == 0 || options_.budget_bytes == 0)
        throw std::invalid_argument(
            "history arena budget and block size must be positive");
      if (options_.region_alignment == 0 ||
          options_.region_alignment & (options_.region_alignment - 1))
        throw std::invalid_argument(
            "history arena region alignment must be a power of two");
      blocks_ = options_.budget_bytes / options_.block_bytes;
      if (blocks_ == 0)
        throw std::invalid_argument(
            "history arena budget is smaller than one block");
      std::vector<void*> allocated;
      allocated.reserve(blocks_);
      for (std::size_t i = 0; i < blocks_; ++i) {
        void* ptr = device_.Allocate(options_.block_bytes);
        if (ptr == nullptr) {
          for (void* p : allocated)
            device_.Free(p);
          throw std::runtime_error("history arena could not allocate its pool");
        }
        allocated.push_back(ptr);
      }
      block_table_.resize(blocks_);
      for (std::size_t i = 0; i < blocks_; ++i) {
        block_table_[i].device_ptr = allocated[i];
        block_table_[i].free_spans.push_back({0, options_.block_bytes});
      }
      effective_budget_ = blocks_ * options_.block_bytes;
      free_bytes_ = effective_budget_;
    }

    ~Impl() {
      for (const Block& block : block_table_)
        device_.Free(block.device_ptr);
    }

    [[nodiscard]] std::shared_ptr<Region> TryReserve(std::size_t bytes) {
      if (bytes == 0)
        return {};
      std::lock_guard lock(mutex_);
      const std::size_t aligned = AlignUp(bytes);
      if (aligned < bytes)
        return {};
      for (std::size_t b = 0; b < block_table_.size(); ++b) {
        Block& block = block_table_[b];
        for (std::size_t s = 0; s < block.free_spans.size(); ++s) {
          const FreeSpan span = block.free_spans[s];
          const std::size_t span_end = span.offset + span.bytes;
          std::size_t start = span.offset;
          if (const std::size_t rem = start % options_.region_alignment;
              rem != 0)
            start += options_.region_alignment - rem;
          if (start < span.offset || start >= span_end)
            continue;
          const std::size_t end = start + aligned;
          if (end < start || end > span_end)
            continue;
          // Carve [start, end) out of the span; keep the free list sorted
          // and non-adjacent.
          block.free_spans.erase(block.free_spans.begin() +
                                 static_cast<std::ptrdiff_t>(s));
          if (span.offset < start)
            InsertSpan(block, {span.offset, start - span.offset});
          if (end < span_end)
            InsertSpan(block, {end, span_end - end});
          live_regions_ += 1;
          live_bytes_ += aligned;
          free_bytes_ -= aligned;
          void* ptr = static_cast<std::byte*>(block.device_ptr) + start;
          return std::shared_ptr<Region>(
              new Region(shared_from_this(), b, start, aligned, ptr));
        }
      }
      return {};
    }

    void InsertSpan(Block& block, FreeSpan span) {
      const auto at = std::lower_bound(
          block.free_spans.begin(), block.free_spans.end(), span.offset,
          [](const FreeSpan& s, std::size_t offset) {
            return s.offset < offset;
          });
      block.free_spans.insert(at, span);
    }

    void Release(std::size_t block_index, std::size_t offset,
                 std::size_t bytes) noexcept {
      try {
        std::lock_guard lock(mutex_);
        Block& block = block_table_[block_index];
        InsertSpan(block, {offset, bytes});
        // Coalesce adjacent spans; sortedness makes neighbors local.
        std::vector<FreeSpan>& spans = block.free_spans;
        std::size_t out = 0;
        for (std::size_t i = 0; i < spans.size(); ++i) {
          if (out > 0 &&
              spans[out - 1].offset + spans[out - 1].bytes == spans[i].offset)
            spans[out - 1].bytes += spans[i].bytes;
          else
            spans[out++] = spans[i];
        }
        spans.resize(out);
        live_regions_ -= 1;
        live_bytes_ -= bytes;
        free_bytes_ += bytes;
      } catch (...) {
        // A region destructor must not throw. Dropping the span shrinks the
        // reusable pool instead; stats expose the loss.
        std::lock_guard lock(mutex_);
        dropped_bytes_ += bytes;
      }
    }

    [[nodiscard]] Stats MakeStats() const {
      std::lock_guard lock(mutex_);
      std::size_t largest = 0;
      for (const Block& block : block_table_)
        for (const FreeSpan& span : block.free_spans)
          largest = std::max(largest, span.bytes);
      return {effective_budget_,
              options_.block_bytes,
              block_table_.size(),
              live_regions_,
              live_bytes_,
              free_bytes_,
              largest,
              blocks_,
              dropped_bytes_};
    }

    [[nodiscard]] std::size_t AlignUp(std::size_t value) const noexcept {
      const std::size_t align = options_.region_alignment;
      return (value / align) * align == value ? value
                                              : (value / align + 1) * align;
    }

    ArenaOptions options_;
    ArenaDevice& device_;
    std::size_t blocks_{0};
    std::size_t effective_budget_{0};
    std::vector<Block> block_table_;
    mutable std::mutex mutex_;
    std::size_t live_regions_{0};
    std::size_t live_bytes_{0};
    std::size_t free_bytes_{0};
    std::size_t dropped_bytes_{0};
  };

  std::shared_ptr<Impl> impl_;
};

}  // namespace gufo::models::qwen38_flash_next::rocm::history

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_ROCM_HISTORY_ARENA_HPP_
