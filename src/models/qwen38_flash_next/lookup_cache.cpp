#include "src/models/qwen38_flash_next/lookup_cache.hpp"

#include <algorithm>

namespace gufo::models::qwen38_flash_next::lookup {
namespace {
constexpr std::uint64_t kHashSeed = 0x9e3779b97f4a7c15ull;
}  // namespace

void FollowerTable::Bump(std::uint32_t token) {
  ++total_;
  const auto position = std::lower_bound(
      entries_.begin(), entries_.end(), token,
      [](const FollowerEntry& entry, std::uint32_t value) {
        return entry.token < value;
      });
  if (position != entries_.end() && position->token == token) {
    ++position->count;
    return;
  }
  entries_.insert(position, FollowerEntry{token, 1});
}

const FollowerEntry* FollowerTable::Best() const noexcept {
  if (entries_.empty()) return nullptr;
  const FollowerEntry* best = &entries_.front();
  for (const auto& entry : entries_) {
    if (entry.count > best->count) best = &entry;
  }
  return best;
}

std::uint64_t ContextNgramCache::Mix(std::uint64_t value) noexcept {
  value ^= value >> 33;
  value *= 0xff51afd7ed558ccdull;
  value ^= value >> 33;
  value *= 0xc4ceb9fe1a85ec53ull;
  value ^= value >> 33;
  return value;
}

std::uint64_t ContextNgramCache::ContextKey(
    std::span<const std::int32_t> context) const noexcept {
  std::uint64_t hash = kHashSeed;
  for (const auto token : context) {
    hash = Mix(hash ^ (static_cast<std::uint64_t>(
                           static_cast<std::uint32_t>(token))
                       << 1));
  }
  return hash == 0 ? 1 : hash;  // 0 marks empty slots
}

const FollowerTable* ContextNgramCache::Find(
    std::uint64_t key) const noexcept {
  if (keys_.empty()) return nullptr;
  const std::size_t mask = keys_.size() - 1;
  std::size_t slot = static_cast<std::size_t>(key) & mask;
  for (std::size_t probes = 0; probes < keys_.size(); ++probes) {
    const auto probed = keys_[slot];
    if (probed == 0) return nullptr;
    if (probed == key) return &tables_[slots_[slot]];
    slot = (slot + 1) & mask;
  }
  return nullptr;
}

FollowerTable& ContextNgramCache::Insert(std::uint64_t key) {
  if (keys_.empty() || size_ * 10 >= keys_.size() * 7) Grow();
  const std::size_t mask = keys_.size() - 1;
  std::size_t slot = static_cast<std::size_t>(key) & mask;
  for (;;) {
    const auto probed = keys_[slot];
    if (probed == 0) {
      keys_[slot] = key;
      slots_[slot] = static_cast<std::uint32_t>(tables_.size());
      table_keys_.push_back(key);
      tables_.emplace_back();
      ++size_;
      return tables_.back();
    }
    if (probed == key) return tables_[slots_[slot]];
    slot = (slot + 1) & mask;
  }
}

void ContextNgramCache::Grow() {
  const std::size_t capacity = keys_.empty() ? 1024 : keys_.size() * 2;
  keys_.assign(capacity, 0);
  slots_.assign(capacity, 0);
  size_ = 0;
  const auto count = static_cast<std::uint32_t>(tables_.size());
  for (std::uint32_t index = 0; index < count; ++index) {
    const auto key = table_keys_[index];
    std::size_t slot = static_cast<std::size_t>(key) & (capacity - 1);
    while (keys_[slot] != 0) slot = (slot + 1) & (capacity - 1);
    keys_[slot] = key;
    slots_[slot] = index;
    ++size_;
  }
}

void ContextNgramCache::Reset() {
  keys_.clear();
  slots_.clear();
  table_keys_.clear();
  tables_.clear();
  size_ = 0;
  built_ = 0;
}

void ContextNgramCache::CatchUp(std::span<const std::int32_t> tokens) {
  for (std::size_t i = built_; i < tokens.size(); ++i) {
    const auto token = static_cast<std::uint32_t>(tokens[i]);
    const std::size_t longest = std::min(kMaxContextTokens, i);
    for (std::size_t context_length = 0; context_length <= longest;
         ++context_length) {
      Insert(ContextKey(tokens.subspan(i - context_length, context_length)))
          .Bump(token);
    }
  }
  built_ = tokens.size();
}

std::optional<std::uint32_t> ContextNgramCache::ProposeOne(
    std::span<const std::int32_t> context) const {
  for (std::size_t context_length =
           std::min(kMaxContextTokens, context.size());
       ; --context_length) {
    if (const auto* table = Find(ContextKey(context.last(context_length)))) {
      if (const auto* best = table->Best();
          best->count >= thresholds_.min_count[context_length] &&
          static_cast<float>(best->count) >=
              thresholds_.min_probability[context_length] *
                  static_cast<float>(table->total())) {
        return best->token;
      }
    }
    if (context_length == 0) return std::nullopt;
  }
}

}  // namespace gufo::models::qwen38_flash_next::lookup
