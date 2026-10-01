#include "src/cli/serve/continuation_cache.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace gufo::server {

namespace {

constexpr auto kCancellationPollInterval = std::chrono::milliseconds{10};

bool IsPrefix(std::span<const ContinuationToken> prefix,
              std::span<const ContinuationToken> tokens) {
  return prefix.size() <= tokens.size() &&
         std::equal(prefix.begin(), prefix.end(), tokens.begin());
}

void EmitSnapshotEvents(const ContinuationCache::SnapshotEventSink& sink,
                        std::span<const SnapshotEvent> events) noexcept {
  if (!sink) {
    return;
  }
  for (const auto& event : events) {
    try {
      sink(event);
    } catch (...) {
      // Cache observability must never affect request execution.
      continue;
    }
  }
}

}  // namespace

struct ContinuationCache::Entry {
  explicit Entry(std::unique_ptr<ContinuationState> model_state)
      : state(std::move(model_state)) {}

  std::unique_ptr<ContinuationState> state;
  std::vector<ContinuationToken> tokens;
  std::vector<std::uint8_t> input_identity;
  std::vector<ContinuationToken> live_tokens;
  std::vector<std::uint8_t> live_identity;
  std::uint64_t state_last_used{0};
  bool available{true};
  bool valid{false};
  bool dirty{false};
};

struct ContinuationCache::Impl {
  struct SnapshotEntry {
    std::shared_ptr<const ContinuationSnapshot> snapshot;
    std::vector<ContinuationToken> tokens;
    std::vector<std::uint8_t> input_identity;
    std::size_t snapshot_bytes{0};
    std::size_t restoring{0};
    // A lease must not protect an unrelated snapshot that later reuses a slot.
    std::uint64_t id{0};
    std::uint64_t last_used{0};
    std::size_t reuse_count{0};
    std::size_t protected_leases{0};
  };

  using OwnerMap =
      std::map<std::shared_ptr<const void>, std::size_t, std::owner_less<>>;
  struct LearnedPrefix {
    std::vector<ContinuationToken> tokens;
    std::vector<std::uint8_t> identity;
    std::uint64_t last_seen;
    std::size_t observations;
  };
  OwnerMap owners;
  std::vector<LearnedPrefix> learned;
  std::uint64_t learn_clock{0};
  std::size_t destroying{0};

  void ReapOwners(std::unique_lock<std::mutex>& lock) {
    if (destroying != 0)
      return;
    OwnerMap retiring;
    std::size_t bytes = 0;
    for (auto it = owners.begin(); it != owners.end();) {
      if (it->first.use_count() == 1) {
        bytes += it->second;
        retiring.insert(owners.extract(it++));
      } else {
        ++it;
      }
    }
    if (retiring.empty())
      return;
    // The table controls the last strong reference. Weak expiration happens
    // before a deleter completes and is not proof that its bytes are free.
    ++destroying;
    lock.unlock();
    retiring.clear();
    lock.lock();
    --destroying;
    retained_snapshot_bytes -= bytes;
  }

  std::vector<std::unique_ptr<Entry>> entries;
  std::vector<SnapshotEntry> snapshots;
  SnapshotSupport snapshot_support;
  mutable std::mutex mutex;
  std::condition_variable condition;
  std::uint64_t clock{0};
  std::size_t snapshot_capacity_bytes{0};
  std::size_t retained_snapshot_bytes{0};
  std::size_t reserved_snapshot_bytes{0};

  [[nodiscard]] bool snapshot_mode() const noexcept {
    return static_cast<bool>(snapshot_support.restore);
  }
};

ContinuationCache::Lease::Lease(ContinuationCache* cache, std::size_t index,
                                bool cache_hit, std::size_t cached_tokens,
                                std::uint64_t source_id,
                                std::size_t restored_snapshot_bytes,
                                double restore_ms) noexcept
    : cache_(cache),
      index_(index),
      cache_hit_(cache_hit),
      cached_tokens_(cached_tokens),
      source_id_(source_id),
      restored_snapshot_bytes_(restored_snapshot_bytes),
      restore_ms_(restore_ms) {}

ContinuationCache::Lease::~Lease() {
  Invalidate();
}

ContinuationCache::Lease::Lease(Lease&& other) noexcept
    : cache_(std::exchange(other.cache_, nullptr)),
      index_(std::exchange(other.index_, 0)),
      cache_hit_(std::exchange(other.cache_hit_, false)),
      cached_tokens_(std::exchange(other.cached_tokens_, 0)),
      source_id_(std::exchange(other.source_id_, 0)),
      restored_snapshot_bytes_(
          std::exchange(other.restored_snapshot_bytes_, 0)),
      restore_ms_(std::exchange(other.restore_ms_, 0.0)),
      restored_from_disk_(std::exchange(other.restored_from_disk_, false)),
      reserved_snapshot_bytes_(
          std::exchange(other.reserved_snapshot_bytes_, 0)),
      preserve_source_(std::exchange(other.preserve_source_, false)),
      input_identity_(std::move(other.input_identity_)),
      lookup_(std::exchange(other.lookup_, {})) {}

ContinuationCache::Lease& ContinuationCache::Lease::operator=(
    Lease&& other) noexcept {
  if (this != &other) {
    Invalidate();
    cache_ = std::exchange(other.cache_, nullptr);
    index_ = std::exchange(other.index_, 0);
    cache_hit_ = std::exchange(other.cache_hit_, false);
    cached_tokens_ = std::exchange(other.cached_tokens_, 0);
    source_id_ = std::exchange(other.source_id_, 0);
    restored_snapshot_bytes_ = std::exchange(other.restored_snapshot_bytes_, 0);
    restore_ms_ = std::exchange(other.restore_ms_, 0.0);
    restored_from_disk_ = std::exchange(other.restored_from_disk_, false);
    reserved_snapshot_bytes_ = std::exchange(other.reserved_snapshot_bytes_, 0);
    preserve_source_ = std::exchange(other.preserve_source_, false);
    input_identity_ = std::move(other.input_identity_);
    lookup_ = std::exchange(other.lookup_, {});
  }
  return *this;
}

ContinuationState& ContinuationCache::Lease::state() const {
  if (cache_ == nullptr) {
    throw std::logic_error("continuation cache lease is empty");
  }
  return cache_->StateAt(index_);
}

void ContinuationCache::Lease::AdoptRestoredPrefix(std::size_t cached_tokens,
                                                   std::size_t restored_bytes,
                                                   double restore_ms) {
  if (cache_ == nullptr) {
    throw std::logic_error("continuation cache lease is empty");
  }
  if (cache_hit_ || cached_tokens == 0 || restored_bytes == 0 ||
      restore_ms < 0.0) {
    throw std::invalid_argument(
        "invalid lower-tier continuation restore metrics");
  }
  cache_hit_ = true;
  cached_tokens_ = cached_tokens;
  restored_snapshot_bytes_ = restored_bytes;
  restore_ms_ = restore_ms;
  restored_from_disk_ = true;
  lookup_ = {};
}

bool ContinuationCache::Lease::TryReserveSnapshot(std::size_t snapshot_bytes,
                                                  std::size_t token_count,
                                                  bool preserve_source,
                                                  bool allow_eviction) {
  if (cache_ == nullptr) {
    throw std::logic_error("continuation cache lease is empty");
  }
  if (reserved_snapshot_bytes_ != 0) {
    throw std::logic_error(
        "continuation cache lease already has a snapshot reservation");
  }
  if (preserve_source && !preserve_source_) {
    cache_->ProtectSource(source_id_, true);
    preserve_source_ = true;
  }
  if (!cache_->ReserveSnapshot(source_id_, snapshot_bytes, token_count,
                               preserve_source || preserve_source_,
                               allow_eviction)) {
    return false;
  }
  reserved_snapshot_bytes_ = snapshot_bytes;
  preserve_source_ = preserve_source_ || preserve_source;
  return true;
}

void ContinuationCache::Lease::SkipSnapshot(SnapshotEventReason reason,
                                            std::size_t snapshot_bytes,
                                            std::size_t token_count) noexcept {
  if (cache_ == nullptr) {
    return;
  }
  cache_->SkipSnapshot(reserved_snapshot_bytes_, reason, snapshot_bytes,
                       token_count);
  reserved_snapshot_bytes_ = 0;
}

std::size_t ContinuationCache::Lease::Commit(
    std::vector<ContinuationToken> tokens,
    std::shared_ptr<const ContinuationSnapshot> snapshot,
    std::vector<ContinuationToken> live_tokens) {
  if (cache_ == nullptr) {
    throw std::logic_error("continuation cache lease is empty");
  }
  const std::size_t retained = cache_->Commit(
      index_, source_id_, reserved_snapshot_bytes_, std::move(tokens),
      std::move(snapshot), std::move(input_identity_), std::move(live_tokens),
      true, preserve_source_);
  if (preserve_source_)
    cache_->ProtectSource(source_id_, false);
  cache_ = nullptr;
  reserved_snapshot_bytes_ = 0;
  return retained;
}

std::size_t ContinuationCache::Lease::PublishSnapshot(
    std::vector<ContinuationToken> tokens,
    std::shared_ptr<const ContinuationSnapshot> snapshot) {
  if (cache_ == nullptr)
    throw std::logic_error("continuation cache lease is empty");
  std::uint64_t published_id = 0;
  const auto retained =
      cache_->Commit(index_, source_id_, reserved_snapshot_bytes_,
                     std::move(tokens), std::move(snapshot), input_identity_,
                     {}, false, preserve_source_, &published_id);
  if (published_id != 0) {
    if (preserve_source_)
      cache_->ProtectSource(source_id_, false);
    source_id_ = published_id;
    preserve_source_ = true;
  }
  reserved_snapshot_bytes_ = 0;
  return retained;
}

void ContinuationCache::Lease::Invalidate() noexcept {
  if (cache_ != nullptr) {
    cache_->Invalidate(index_, reserved_snapshot_bytes_);
    if (preserve_source_)
      cache_->ProtectSource(source_id_, false);
    cache_ = nullptr;
    reserved_snapshot_bytes_ = 0;
  }
}

ContinuationCache::ContinuationCache(std::size_t capacity,
                                     const StateFactory& factory)
    : ContinuationCache(capacity, factory, SnapshotSupport{}) {}

ContinuationCache::ContinuationCache(std::size_t capacity,
                                     const StateFactory& factory,
                                     SnapshotSupport snapshot_support)
    : impl_(std::make_unique<Impl>()) {
  if (capacity == 0) {
    throw std::invalid_argument(
        "continuation cache capacity must be at least one");
  }
  if (!factory) {
    throw std::invalid_argument(
        "continuation cache state factory must be callable");
  }
  impl_->snapshot_support = std::move(snapshot_support);
  if (impl_->snapshot_mode()) {
    if (impl_->snapshot_support.entry_capacity == 0) {
      throw std::invalid_argument(
          "continuation snapshot entry capacity must be at least one");
    }
    impl_->snapshots.resize(impl_->snapshot_support.entry_capacity);
  }

  impl_->entries.reserve(capacity);
  for (std::size_t index = 0; index < capacity; ++index) {
    auto state = factory();
    if (state == nullptr) {
      throw std::runtime_error(
          "continuation cache state factory returned null");
    }
    impl_->entries.push_back(std::make_unique<Entry>(std::move(state)));
  }
  if (impl_->snapshot_mode() && impl_->snapshot_support.capacity_bytes) {
    try {
      impl_->snapshot_capacity_bytes = impl_->snapshot_support.capacity_bytes();
    } catch (...) {
      impl_->snapshot_capacity_bytes = 0;
    }
  }
}

ContinuationCache::~ContinuationCache() = default;

bool ContinuationCache::Lease::HasSnapshotFor(
    std::span<const ContinuationToken> tokens) const {
  if (!cache_)
    return false;
  const std::lock_guard lock(cache_->impl_->mutex);
  for (auto& source : cache_->impl_->snapshots) {
    if (source.snapshot && source.input_identity == input_identity_ &&
        std::ranges::equal(source.tokens, tokens)) {
      source.last_used = ++cache_->impl_->clock;
      return true;
    }
  }
  return false;
}

ContinuationCache::Lease ContinuationCache::Acquire(
    std::span<const ContinuationToken> prompt,
    const CancellationCheck& is_cancelled,
    std::span<const std::uint8_t> input_identity,
    const std::function<void(ContinuationState&)>& prepare_state,
    bool reuse_prompt, std::size_t min_cached_tokens) {
  while (true) {
    std::unique_lock<std::mutex> lock(impl_->mutex);

    const std::size_t no_entry = impl_->entries.size();
    const std::size_t no_snapshot = impl_->snapshots.size();
    std::size_t source = no_snapshot;
    std::size_t live_source = no_entry;
    std::size_t cached_tokens = 0;
    if (reuse_prompt && impl_->snapshot_mode()) {
      for (std::size_t index = 0; index < impl_->snapshots.size(); ++index) {
        const auto& entry = impl_->snapshots[index];
        if (entry.snapshot && entry.tokens.size() >= min_cached_tokens &&
            std::ranges::equal(entry.input_identity, input_identity) &&
            IsPrefix(entry.tokens, prompt) &&
            (source == no_snapshot || entry.tokens.size() > cached_tokens)) {
          source = index;
          cached_tokens = entry.tokens.size();
        }
      }
    }
    // A live final frontier can extend the immutable prompt snapshot. Prefer
    // it on equal prefix lengths too: no restoration or D2H copy is needed.
    for (std::size_t index = 0; reuse_prompt && index < impl_->entries.size();
         ++index) {
      const auto& entry = *impl_->entries[index];
      const auto& tokens =
          impl_->snapshot_mode() ? entry.live_tokens : entry.tokens;
      const auto& identity =
          impl_->snapshot_mode() ? entry.live_identity : entry.input_identity;
      if (entry.available && !tokens.empty() &&
          (impl_->snapshot_mode() || entry.valid) &&
          tokens.size() >= cached_tokens &&
          tokens.size() >= min_cached_tokens &&
          std::ranges::equal(identity, input_identity) &&
          IsPrefix(tokens, prompt)) {
        live_source = index;
        cached_tokens = tokens.size();
      }
    }

    const bool live_hit = live_source != no_entry;
    const bool cache_hit = live_hit || source != no_snapshot;
    ContinuationLookup lookup;
    if (!cache_hit) {
      lookup.miss_reason = reuse_prompt ? "no_checkpoint" : "disabled";
      const auto inspect = [&](const auto& tokens, const auto& identity) {
        if (tokens.empty())
          return;
        if (!std::ranges::equal(identity, input_identity)) {
          if (lookup.miss_reason == "no_checkpoint")
            lookup.miss_reason = "input_changed";
          return;
        }
        const auto common =
            static_cast<std::size_t>(std::mismatch(tokens.begin(), tokens.end(),
                                                   prompt.begin(), prompt.end())
                                         .first -
                                     tokens.begin());
        if (lookup.checkpoint_tokens == 0 ||
            common > lookup.common_prefix_tokens) {
          lookup = {"prefix_changed", common, tokens.size()};
        }
      };
      if (reuse_prompt) {
        for (const auto& entry : impl_->snapshots) {
          if (entry.snapshot)
            inspect(entry.tokens, entry.input_identity);
        }
        for (const auto& entry : impl_->entries) {
          if (entry->valid)
            inspect(entry->tokens, entry->input_identity);
          inspect(entry->live_tokens, entry->live_identity);
        }
      }
    }
    std::size_t selected = live_source;
    if (!live_hit) {
      selected = no_entry;
      std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
      for (std::size_t index = 0; index < impl_->entries.size(); ++index) {
        const auto& entry = *impl_->entries[index];
        if (entry.available && entry.state_last_used < oldest) {
          selected = index;
          oldest = entry.state_last_used;
        }
      }
    }

    if (selected != no_entry) {
      std::vector<std::uint8_t> lease_identity(input_identity.begin(),
                                               input_identity.end());
      auto& entry = *impl_->entries[selected];
      entry.available = false;
      const bool needs_invalidation = entry.dirty && !cache_hit;
      entry.dirty = true;
      entry.state_last_used = ++impl_->clock;
      entry.live_tokens.clear();
      entry.live_identity.clear();

      std::shared_ptr<const ContinuationSnapshot> snapshot;
      std::uint64_t source_id = 0;
      if (impl_->snapshot_mode()) {
        if (source != no_snapshot) {
          auto& source_entry = impl_->snapshots[source];
          source_id = source_entry.id;
          source_entry.last_used = ++impl_->clock;
          ++source_entry.reuse_count;
          if (!live_hit) {
            snapshot = source_entry.snapshot;
            ++source_entry.restoring;
          }
        }
      } else {
        entry.valid = false;
        entry.tokens.clear();
      }
      lock.unlock();

      std::size_t restored_snapshot_bytes = 0;
      double restore_ms = 0.0;
      try {
        if (needs_invalidation)
          entry.state->Invalidate();
        if (prepare_state)
          prepare_state(*entry.state);
        if (cache_hit && impl_->snapshot_mode() && !live_hit) {
          if (snapshot == nullptr) {
            throw std::runtime_error(
                "continuation cache snapshot entry is empty");
          }
          restored_snapshot_bytes = snapshot->PayloadBytes();
          const auto restore_start = std::chrono::steady_clock::now();
          impl_->snapshot_support.restore(*entry.state, *snapshot);
          restore_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - restore_start)
                           .count();
        }
      } catch (...) {
        entry.state->Invalidate();
        {
          const std::lock_guard<std::mutex> failure_lock(impl_->mutex);
          if (snapshot) {
            snapshot.reset();
            --impl_->snapshots[source].restoring;
          }
          entry.available = true;
          entry.dirty = false;
          entry.state_last_used = ++impl_->clock;
        }
        impl_->condition.notify_one();
        throw;
      }
      if (snapshot) {
        const std::lock_guard<std::mutex> restore_lock(impl_->mutex);
        snapshot.reset();
        --impl_->snapshots[source].restoring;
      }
      Lease lease(this, selected, cache_hit, cached_tokens, source_id,
                  restored_snapshot_bytes, restore_ms);
      lease.input_identity_ = std::move(lease_identity);
      lease.lookup_ = lookup;
      return lease;
    }

    lock.unlock();
    if (is_cancelled && is_cancelled()) {
      return {};
    }
    lock.lock();
    impl_->condition.wait_for(lock, kCancellationPollInterval);
  }
}

std::vector<std::size_t> ContinuationCache::LearnPrefixBoundaries(
    std::span<const ContinuationToken> prompt,
    std::span<const std::uint8_t> input_identity, std::size_t min_tokens) {
  constexpr std::size_t kMetadataBytes = 2 * 1024 * 1024;
  constexpr std::size_t kCandidateCapacity = 8;
  constexpr std::uint64_t kMaxAgeRequests = 128;
  const std::lock_guard lock(impl_->mutex);
  const auto now = ++impl_->learn_clock;
  std::erase_if(impl_->learned, [&](const auto& entry) {
    return now - entry.last_seen > kMaxAgeRequests;
  });
  std::size_t common = 0;
  const auto inspect = [&](const auto& tokens, const auto& identity) {
    if (!std::ranges::equal(identity, input_identity))
      return;
    const auto count =
        static_cast<std::size_t>(std::mismatch(tokens.begin(), tokens.end(),
                                               prompt.begin(), prompt.end())
                                     .first -
                                 tokens.begin());
    if (count < tokens.size() && count < prompt.size())
      common = std::max(common, count);
  };
  for (const auto& entry : impl_->snapshots) {
    if (entry.snapshot)
      inspect(entry.tokens, entry.input_identity);
  }
  for (const auto& entry : impl_->entries) {
    if (entry->valid)
      inspect(entry->tokens, entry->input_identity);
    inspect(entry->live_tokens, entry->live_identity);
  }
  if (common >= min_tokens && common != 0 &&
      input_identity.size() <= kMetadataBytes &&
      common <= (kMetadataBytes - input_identity.size()) /
                    sizeof(ContinuationToken)) {
    const auto prefix = prompt.first(common);
    auto found = std::ranges::find_if(impl_->learned, [&](const auto& entry) {
      return std::ranges::equal(entry.identity, input_identity) &&
             std::ranges::equal(entry.tokens, prefix);
    });
    if (found != impl_->learned.end()) {
      ++found->observations;
      found->last_seen = now;
    } else {
      const auto required =
          common * sizeof(ContinuationToken) + input_identity.size();
      const auto fits = [&] {
        std::size_t used = 0;
        for (const auto& entry : impl_->learned)
          used += entry.tokens.size() * sizeof(ContinuationToken) +
                  entry.identity.size();
        return impl_->learned.size() < kCandidateCapacity &&
               required <= kMetadataBytes - used;
      };
      while (!fits()) {
        const auto oldest = std::ranges::min_element(
            impl_->learned, {}, &Impl::LearnedPrefix::last_seen);
        impl_->learned.erase(oldest);
      }
      impl_->learned.push_back({{prefix.begin(), prefix.end()},
                                {input_identity.begin(), input_identity.end()},
                                now,
                                1});
    }
  }
  std::vector<std::size_t> candidates;
  for (auto& entry : impl_->learned) {
    if (entry.tokens.size() >= min_tokens &&
        entry.tokens.size() < prompt.size() &&
        std::ranges::equal(entry.identity, input_identity) &&
        IsPrefix(entry.tokens, prompt)) {
      candidates.push_back(entry.tokens.size());
    }
  }
  std::ranges::sort(candidates, std::greater<>{});
  return candidates;
}

std::size_t ContinuationCache::capacity() const noexcept {
  return impl_->entries.size();
}

std::size_t ContinuationCache::snapshot_entry_capacity() const noexcept {
  return impl_->snapshots.size();
}

std::size_t ContinuationCache::snapshot_capacity_bytes() const noexcept {
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->snapshot_capacity_bytes;
}

std::size_t ContinuationCache::retained_snapshot_bytes() const noexcept {
  std::unique_lock lock(impl_->mutex);
  impl_->ReapOwners(lock);
  return impl_->retained_snapshot_bytes;
}

std::size_t ContinuationCache::reserved_snapshot_bytes() const noexcept {
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->reserved_snapshot_bytes;
}

std::vector<SlotLineage> ContinuationCache::FreeSlotLineages() const {
  std::vector<SlotLineage> lineages;
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  lineages.reserve(impl_->entries.size());
  for (std::size_t index = 0; index < impl_->entries.size(); ++index) {
    const auto& entry = *impl_->entries[index];
    if (!entry.available)
      continue;
    const auto& tokens = impl_->snapshot_mode() ? entry.live_tokens
                                                : entry.tokens;
    const auto& identity = impl_->snapshot_mode() ? entry.live_identity
                                                  : entry.input_identity;
    if (tokens.empty() || identity.empty())
      continue;
    if (impl_->snapshot_mode() || entry.valid)
      lineages.push_back(
          {index, tokens.size(), {identity.begin(), identity.end()}});
  }
  return lineages;
}

void ContinuationCache::ProtectSource(std::uint64_t source_id,
                                      bool protect) noexcept {
  if (source_id == 0)
    return;
  const std::lock_guard lock(impl_->mutex);
  for (auto& entry : impl_->snapshots) {
    if (entry.snapshot && entry.id == source_id) {
      if (protect)
        ++entry.protected_leases;
      else if (entry.protected_leases != 0)
        --entry.protected_leases;
      break;
    }
  }
}

ContinuationState& ContinuationCache::StateAt(std::size_t index) {
  return *impl_->entries.at(index)->state;
}

bool ContinuationCache::ReserveSnapshot(std::uint64_t source_id,
                                        std::size_t snapshot_bytes,
                                        std::size_t token_count,
                                        bool preserve_source,
                                        bool allow_eviction) {
  std::vector<SnapshotEvent> events;
  bool admitted = false;
  std::unique_lock lock(impl_->mutex);
  impl_->ReapOwners(lock);
  const auto make_event = [&](SnapshotEventAction action,
                              SnapshotEventReason reason, std::size_t bytes,
                              std::size_t tokens) {
    return SnapshotEvent{action,
                         reason,
                         bytes,
                         tokens,
                         impl_->retained_snapshot_bytes,
                         impl_->reserved_snapshot_bytes,
                         impl_->snapshot_capacity_bytes};
  };
  const auto fits = [&] {
    const auto used =
        impl_->retained_snapshot_bytes + impl_->reserved_snapshot_bytes;
    if (snapshot_bytes == 0 || used > impl_->snapshot_capacity_bytes)
      return false;
    const auto free = impl_->snapshot_capacity_bytes - used;
    // An optional early tap leaves room for at least one ordinary snapshot.
    return snapshot_bytes <= (allow_eviction ? free : free / 2);
  };
  const bool possible =
      snapshot_bytes != 0 &&
      impl_->reserved_snapshot_bytes <= impl_->snapshot_capacity_bytes &&
      snapshot_bytes <=
          impl_->snapshot_capacity_bytes - impl_->reserved_snapshot_bytes;
  while (possible && !fits() && allow_eviction) {
    std::size_t target = impl_->snapshots.size();
    for (const bool allow_source : {false, true}) {
      if (allow_source && preserve_source)
        break;
      for (std::size_t i = 0; i < impl_->snapshots.size(); ++i) {
        const auto& entry = impl_->snapshots[i];
        if (!entry.snapshot || entry.restoring != 0 ||
            entry.protected_leases != 0 ||
            (!allow_source && entry.id == source_id))
          continue;
        if (target == impl_->snapshots.size() ||
            std::tie(entry.reuse_count, entry.last_used) <
                std::tie(impl_->snapshots[target].reuse_count,
                         impl_->snapshots[target].last_used))
          target = i;
      }
      if (target != impl_->snapshots.size())
        break;
    }
    if (target == impl_->snapshots.size())
      break;
    auto& entry = impl_->snapshots[target];
    events.push_back(make_event(SnapshotEventAction::kRemoved,
                                SnapshotEventReason::kByteCapacity,
                                entry.snapshot_bytes, entry.tokens.size()));
    auto retired = std::move(entry.snapshot);
    entry = {};
    // Physical owners keep external/restore/destructor readers charged.
    ++impl_->destroying;
    lock.unlock();
    retired.reset();
    lock.lock();
    --impl_->destroying;
    impl_->ReapOwners(lock);
  }
  if (fits()) {
    impl_->reserved_snapshot_bytes += snapshot_bytes;
    admitted = true;
  } else {
    events.push_back(make_event(SnapshotEventAction::kSkipped,
                                SnapshotEventReason::kByteCapacity,
                                snapshot_bytes, token_count));
  }
  for (auto& event : events) {
    event.retained_snapshot_bytes = impl_->retained_snapshot_bytes;
    event.reserved_snapshot_bytes = impl_->reserved_snapshot_bytes;
  }
  lock.unlock();
  EmitSnapshotEvents(impl_->snapshot_support.on_event, events);
  return admitted;
}

void ContinuationCache::SkipSnapshot(std::size_t reservation_bytes,
                                     SnapshotEventReason reason,
                                     std::size_t snapshot_bytes,
                                     std::size_t token_count) noexcept {
  SnapshotEvent event;
  {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    if (reservation_bytes <= impl_->reserved_snapshot_bytes) {
      impl_->reserved_snapshot_bytes -= reservation_bytes;
    } else {
      impl_->reserved_snapshot_bytes = 0;
    }
    event = {
        .action = SnapshotEventAction::kSkipped,
        .reason = reason,
        .snapshot_bytes = snapshot_bytes,
        .token_count = token_count,
        .retained_snapshot_bytes = impl_->retained_snapshot_bytes,
        .reserved_snapshot_bytes = impl_->reserved_snapshot_bytes,
        .capacity_bytes = impl_->snapshot_capacity_bytes,
    };
  }
  EmitSnapshotEvents(impl_->snapshot_support.on_event,
                     std::span<const SnapshotEvent>(&event, 1));
}

std::size_t ContinuationCache::Commit(
    std::size_t index, std::uint64_t source_id, std::size_t reservation_bytes,
    std::vector<ContinuationToken> tokens,
    std::shared_ptr<const ContinuationSnapshot> snapshot,
    std::vector<std::uint8_t> input_identity,
    std::vector<ContinuationToken> live_tokens, bool release_state,
    bool preserve_source, std::uint64_t* published_id) {
  const auto token_count = tokens.size();
  const auto snapshot_bytes = snapshot ? snapshot->PayloadBytes() : 0;
  auto reason = SnapshotEventReason::kCaptureFailure;
  Impl::OwnerMap owners;
  bool retain = impl_->snapshot_mode() && snapshot && !tokens.empty() &&
                reservation_bytes != 0 && snapshot_bytes != 0;
  if (snapshot) {
    const auto physical = snapshot->StorageOwners();
    if (physical.empty()) {
      owners.emplace(snapshot, snapshot_bytes);
      retain = retain && snapshot_bytes == reservation_bytes;
    } else {
      for (const auto& owner : physical) {
        const auto found = owners.find(owner.owner);
        if (!owner.owner || owner.bytes == 0 ||
            (found != owners.end() && found->second != owner.bytes)) {
          retain = false;
          break;
        }
        if (found == owners.end())
          owners.emplace(owner.owner, owner.bytes);
      }
    }
    if (!retain)
      reason = SnapshotEventReason::kReservationMismatch;
  }
  std::shared_ptr<const ContinuationSnapshot> retired;
  std::vector<SnapshotEvent> events;
  events.reserve(1);
  std::size_t retained_bytes = 0;
  std::size_t release_bytes = 0;
  std::size_t published_target = impl_->snapshots.size();
  {
    std::unique_lock lock(impl_->mutex);
    impl_->ReapOwners(lock);
    release_bytes = std::min(reservation_bytes, impl_->reserved_snapshot_bytes);
    if (release_bytes != reservation_bytes) {
      retain = false;
      reason = SnapshotEventReason::kReservationMismatch;
    }
    const auto event = [&](SnapshotEventAction action, SnapshotEventReason why,
                           std::size_t bytes, std::size_t count) {
      return SnapshotEvent{action,
                           why,
                           bytes,
                           count,
                           impl_->retained_snapshot_bytes,
                           impl_->reserved_snapshot_bytes,
                           impl_->snapshot_capacity_bytes};
    };
    auto& state = *impl_->entries.at(index);
    if (impl_->snapshot_mode()) {
      if (release_state) {
        state.live_tokens = std::move(live_tokens);
        state.live_identity = input_identity;
      }
      const auto none = impl_->snapshots.size();
      std::size_t target = none;
      bool exact = false;
      if (retain) {
        for (std::size_t i = 0; i < none; ++i) {
          const auto& entry = impl_->snapshots[i];
          if (entry.snapshot && entry.tokens == tokens &&
              entry.input_identity == input_identity) {
            exact = true;
            if (entry.restoring == 0 &&
                (entry.protected_leases == 0 ||
                 (entry.id == source_id && preserve_source &&
                  entry.protected_leases == 1)))
              target = i;
            break;
          }
        }
        if (target == none && !exact) {
          for (std::size_t i = 0; i < none; ++i) {
            if (!impl_->snapshots[i].snapshot) {
              target = i;
              break;
            }
          }
        }
        if (target == none && !exact) {
          for (const bool allow_source : {false, true}) {
            if (allow_source && preserve_source)
              break;
            for (std::size_t i = 0; i < none; ++i) {
              const auto& entry = impl_->snapshots[i];
              if (entry.restoring != 0 || entry.protected_leases != 0 ||
                  (!allow_source && entry.id == source_id))
                continue;
              if (target == none ||
                  std::tie(entry.reuse_count, entry.last_used) <
                      std::tie(impl_->snapshots[target].reuse_count,
                               impl_->snapshots[target].last_used))
                target = i;
            }
            if (target != none)
              break;
          }
        }
        std::size_t added = 0;
        for (const auto& [owner, bytes] : owners) {
          const auto found = impl_->owners.find(owner);
          if (found != impl_->owners.end()) {
            if (bytes != found->second)
              retain = false;
          } else if (bytes > reservation_bytes - added) {
            retain = false;
          } else {
            added += bytes;
          }
        }
        const auto used = impl_->retained_snapshot_bytes +
                          impl_->reserved_snapshot_bytes - release_bytes;
        if (target == none) {
          retain = false;
          reason = SnapshotEventReason::kEntryCapacity;
        } else if (!retain || used > impl_->snapshot_capacity_bytes ||
                   added > impl_->snapshot_capacity_bytes - used) {
          retain = false;
          reason = SnapshotEventReason::kReservationMismatch;
        }
        if (retain) {
          auto& entry = impl_->snapshots[target];
          if (entry.snapshot) {
            events.push_back(event(SnapshotEventAction::kRemoved,
                                   exact
                                       ? SnapshotEventReason::kExactReplacement
                                       : SnapshotEventReason::kEntryCapacity,
                                   entry.snapshot_bytes, entry.tokens.size()));
            retired = std::move(entry.snapshot);
          }
          entry = {};
          entry.snapshot = std::move(snapshot);
          entry.tokens = std::move(tokens);
          entry.input_identity = std::move(input_identity);
          entry.snapshot_bytes = snapshot_bytes;
          entry.id = entry.last_used = ++impl_->clock;
          if (retired) {
            ++entry.restoring;
            published_target = target;
          }
          if (published_id) {
            *published_id = entry.id;
            ++entry.protected_leases;
          }
          // Transfer preallocated ownership nodes without allocating after
          // publication has mutated a retained slot.
          impl_->owners.merge(owners);
          impl_->retained_snapshot_bytes += added;
          impl_->reserved_snapshot_bytes -= release_bytes;
          release_bytes = 0;
          retained_bytes = snapshot_bytes;
        }
      }
      if (!retain && (snapshot || reservation_bytes != 0)) {
        // External callers can still hold rejected candidates. Keep their
        // physical owners resident until our table can destroy them safely.
        for (auto it = owners.begin(); it != owners.end();) {
          const auto& [owner, bytes] = *it;
          if (!owner || bytes == 0 ||
              (!impl_->owners.contains(owner) &&
               bytes > std::numeric_limits<std::size_t>::max() -
                           impl_->retained_snapshot_bytes)) {
            it = owners.erase(it);
            continue;
          }
          if (!impl_->owners.contains(owner))
            impl_->retained_snapshot_bytes += bytes;
          ++it;
        }
        impl_->owners.merge(owners);
        impl_->reserved_snapshot_bytes -= release_bytes;
        release_bytes = 0;
        events.push_back(event(SnapshotEventAction::kSkipped, reason,
                               snapshot_bytes, token_count));
      }
    } else {
      state.tokens = std::move(tokens);
      state.input_identity = std::move(input_identity);
      state.valid = !state.tokens.empty();
    }
    if (release_state) {
      state.dirty = true;
      state.available = true;
      state.state_last_used = ++impl_->clock;
    }
    ++impl_->destroying;
  }
  // Neither rejected candidates nor retired owners become free during their
  // destructors. Destruction can reenter admission on another execution slot.
  owners.clear();
  snapshot.reset();
  retired.reset();
  {
    std::unique_lock lock(impl_->mutex);
    --impl_->destroying;
    impl_->reserved_snapshot_bytes -= release_bytes;
    impl_->ReapOwners(lock);
    if (published_target != impl_->snapshots.size())
      --impl_->snapshots[published_target].restoring;
    for (auto& event : events) {
      event.retained_snapshot_bytes = impl_->retained_snapshot_bytes;
      event.reserved_snapshot_bytes = impl_->reserved_snapshot_bytes;
    }
  }
  EmitSnapshotEvents(impl_->snapshot_support.on_event, events);
  impl_->condition.notify_all();
  return retained_bytes;
}

void ContinuationCache::Invalidate(std::size_t index,
                                   std::size_t reservation_bytes) noexcept {
  auto& entry = *impl_->entries[index];
  entry.state->Invalidate();
  {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    if (reservation_bytes <= impl_->reserved_snapshot_bytes) {
      impl_->reserved_snapshot_bytes -= reservation_bytes;
    } else {
      impl_->reserved_snapshot_bytes = 0;
    }
    if (!impl_->snapshot_mode()) {
      entry.tokens.clear();
      entry.valid = false;
    }
    entry.live_tokens.clear();
    entry.live_identity.clear();
    entry.dirty = false;
    entry.available = true;
    entry.state_last_used = ++impl_->clock;
  }
  impl_->condition.notify_one();
}

}  // namespace gufo::server
