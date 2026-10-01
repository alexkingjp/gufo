#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_LOOKUP_CACHE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_LOOKUP_CACHE_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace gufo::models::qwen38_flash_next::lookup {

/// Follower counts for one context n-gram, kept sorted by token. Follower
/// sets are tiny in practice (most contexts have a single follower and
/// virtually all have fewer than 100), so arrays scanned with
/// count-tracking maxima beat trees and chained maps on both cache misses
/// and allocations.
struct FollowerEntry {
  std::uint32_t token;
  std::uint32_t count;
};

class FollowerTable {
 public:
  void Bump(std::uint32_t token);
  /// Highest-count follower, or nullptr when nothing was counted yet.
  [[nodiscard]] const FollowerEntry* Best() const noexcept;
  [[nodiscard]] std::uint32_t total() const noexcept { return total_; }

 private:
  std::vector<FollowerEntry> entries_;  // sorted by token
  std::uint32_t total_{0};
};

/// Prompt-lookup cache over one session's tokens (the "context cache" of
/// prompt-lookup decoding). An open-addressed flat map keyed by a 64-bit mix
/// of the context tokens; each value counts follower tokens observed after
/// that context. Catch-up insertion is O(new tokens) and a query is a
/// handful of probes, so drafting stays negligible next to one GPU
/// verification cycle. Hash collisions can only mispropose a token the
/// trunk then rejects — acceptance cost, never correctness.
class ContextNgramCache {
 public:
  /// Longest context (in tokens) retained before the follower. An n-gram of
  /// size n is a context of n-1 tokens plus its follower.
  static constexpr std::size_t kMaxContextTokens = 3;

  /// Acceptance thresholds indexed by context length 0..3: minimum follower
  /// count and minimum count/total probability. The defaults mirror the
  /// reference prompt-lookup research (n-gram sizes 1..4).
  struct Thresholds {
    std::uint32_t min_count[kMaxContextTokens + 1]{2, 2, 1, 1};
    float min_probability[kMaxContextTokens + 1]{0.66f, 0.5f, 0.5f, 0.5f};
  };

  void Reset();
  /// Counts followers for tokens[built_..); call after the token vector
  /// grows. Repeated calls with unchanged prefixes are no-ops.
  void CatchUp(std::span<const std::int32_t> tokens);
  /// The most confident follower of the longest suffix of `context` (at most
  /// kMaxContextTokens tokens) that passes the thresholds. Longest match
  /// wins and stops the search: shorter contexts are strictly less
  /// reliable, so their hits cannot override a confident long one.
  [[nodiscard]] std::optional<std::uint32_t> ProposeOne(
      std::span<const std::int32_t> context) const;

  [[nodiscard]] std::size_t built() const noexcept { return built_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

 private:
  static std::uint64_t Mix(std::uint64_t value) noexcept;
  [[nodiscard]] std::uint64_t ContextKey(
      std::span<const std::int32_t> context) const noexcept;
  [[nodiscard]] const FollowerTable* Find(std::uint64_t key) const noexcept;
  FollowerTable& Insert(std::uint64_t key);
  void Grow();

  // Open addressing with linear probing over a power-of-two capacity.
  // keys_[i] == 0 marks an empty slot; hashes are remapped away from 0.
  // tables_ is a dense pool: slots_ store pool indices that survive rehash,
  // and table_keys_ carries each pool entry's key so Grow can re-probe.
  std::vector<std::uint64_t> keys_;
  std::vector<std::uint32_t> slots_;
  std::vector<std::uint64_t> table_keys_;
  std::vector<FollowerTable> tables_;
  std::size_t size_{0};
  std::size_t built_{0};
  Thresholds thresholds_;
};

}  // namespace gufo::models::qwen38_flash_next::lookup

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_LOOKUP_CACHE_HPP_
