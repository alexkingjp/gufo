#include "src/cli/serve/continuation_cache.hpp"

#include <array>
#include <atomic>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

struct FakeState final : gufo::server::ContinuationState {
  FakeState(std::size_t state_id, std::vector<std::size_t>* invalidations)
      : id(state_id), invalidation_counts(invalidations) {}

  void Invalidate() noexcept override { ++invalidation_counts->at(id); }

  std::size_t id;
  std::size_t value{0};
  std::vector<std::size_t>* invalidation_counts;
};

struct FakeSnapshot final : gufo::server::ContinuationSnapshot {
  explicit FakeSnapshot(std::size_t value,
                        std::size_t payload_bytes = sizeof(std::size_t))
      : value(value), payload_bytes(payload_bytes) {}

  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    return payload_bytes;
  }

  std::size_t value;
  std::size_t payload_bytes;
};

struct TrackedSnapshot final : gufo::server::ContinuationSnapshot {
  TrackedSnapshot(std::atomic<std::size_t>& live_bytes, std::size_t bytes,
                  std::function<void()> on_destroy = {})
      : live_bytes(live_bytes),
        bytes(bytes),
        on_destroy(std::move(on_destroy)) {
    live_bytes += bytes;
  }

  ~TrackedSnapshot() override {
    if (on_destroy)
      on_destroy();
    live_bytes -= bytes;
  }

  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    return bytes;
  }

  std::atomic<std::size_t>& live_bytes;
  std::size_t bytes;
  std::function<void()> on_destroy;
};

struct SnapshotCacheFixture {
  SnapshotCacheFixture(
      std::size_t states, std::size_t bytes,
      std::size_t entries =
          gufo::server::ContinuationCache::SnapshotSupport{}.entry_capacity)
      : invalidations(states),
        cache(
            states,
            [&] {
              return std::make_unique<FakeState>(states_created++,
                                                 &invalidations);
            },
            {
                .restore =
                    [](auto& state, const auto& snapshot) {
                      dynamic_cast<FakeState&>(state).value =
                          dynamic_cast<const FakeSnapshot&>(snapshot).value;
                    },
                .capacity_bytes = [bytes] { return bytes; },
                .on_event = [&](const auto& event) { events.push_back(event); },
                .entry_capacity = entries,
            }) {}

  void Save(std::vector<gufo::server::ContinuationToken> tokens,
            std::size_t value,
            std::vector<gufo::server::ContinuationToken> live_tokens = {}) {
    auto lease = cache.Acquire(tokens);
    dynamic_cast<FakeState&>(lease.state()).value = value;
    Expect(lease.TryReserveSnapshot(8, tokens.size()),
           "fixture snapshot reserves before allocation");
    Expect(lease.Commit(std::move(tokens),
                        std::make_unique<FakeSnapshot>(value, 8),
                        std::move(live_tokens)) == 8,
           "fixture snapshot is retained");
  }

  std::vector<std::size_t> invalidations;
  std::size_t states_created{0};
  std::vector<gufo::server::SnapshotEvent> events;
  gufo::server::ContinuationCache cache;
};

void TestColdMissThenExactExtensionHit() {
  std::vector<std::size_t> invalidations(1);
  std::size_t next_id = 0;
  gufo::server::ContinuationCache cache(1, [&] {
    return std::make_unique<FakeState>(next_id++, &invalidations);
  });

  const std::vector<gufo::server::ContinuationToken> first_prompt{1, 2, 3};
  {
    auto lease = cache.Acquire(first_prompt);
    Expect(static_cast<bool>(lease), "cold request acquires the slot");
    Expect(!lease.cache_hit(), "first request is a cache miss");
    Expect(lease.lookup().miss_reason == "no_checkpoint",
           "cold miss is distinguished from changed input");
    Expect(lease.cached_tokens() == 0, "cold request reuses no tokens");
    Expect(dynamic_cast<FakeState&>(lease.state()).id == 0,
           "lease exposes the opaque model state");
    lease.Commit({1, 2, 3, 4});
  }

  const std::vector<gufo::server::ContinuationToken> extension{1, 2, 3,
                                                               4, 5, 6};
  {
    auto lease = cache.Acquire(extension);
    Expect(lease.cache_hit(), "exact extension reuses the slot");
    Expect(lease.lookup().miss_reason.empty(), "hits carry no miss reason");
    Expect(lease.cached_tokens() == 4,
           "hit reports the complete retained prefix");
    Expect(dynamic_cast<FakeState&>(lease.state()).id == 0,
           "hit returns the same opaque state");
    Expect(invalidations[0] == 0, "hit does not invalidate retained state");
    lease.Commit(extension);
  }
}

void TestDivergenceInvalidatesOldState() {
  std::vector<std::size_t> invalidations(1);
  gufo::server::ContinuationCache cache(
      1, [&] { return std::make_unique<FakeState>(0, &invalidations); });

  {
    auto lease =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2});
    lease.Commit({1, 2, 3});
  }
  {
    auto lease =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 9});
    Expect(!lease.cache_hit(), "divergent request is a miss");
    Expect(lease.lookup().miss_reason == "prefix_changed" &&
               lease.lookup().common_prefix_tokens == 1 &&
               lease.lookup().checkpoint_tokens == 3,
           "miss identifies the first changed token without exposing it");
    Expect(invalidations[0] == 1,
           "divergence invalidates the previous model state");
    lease.Commit({1, 9});
  }
}

void TestUncommittedLeaseIsInvalidated() {
  std::vector<std::size_t> invalidations(1);
  gufo::server::ContinuationCache cache(
      1, [&] { return std::make_unique<FakeState>(0, &invalidations); });

  {
    auto lease = cache.Acquire(std::vector<gufo::server::ContinuationToken>{7});
    Expect(static_cast<bool>(lease), "request acquires the slot");
  }
  Expect(invalidations[0] == 1,
         "abandoned request invalidates partially computed state");

  auto retry =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{7, 8});
  Expect(!retry.cache_hit(), "abandoned state is never reused");
  retry.Commit({7, 8});
}

void TestLongestAvailablePrefixWins() {
  std::vector<std::size_t> invalidations(2);
  std::size_t next_id = 0;
  gufo::server::ContinuationCache cache(2, [&] {
    return std::make_unique<FakeState>(next_id++, &invalidations);
  });

  {
    auto first = cache.Acquire(std::vector<gufo::server::ContinuationToken>{1});
    first.Commit({1, 2});
  }
  {
    auto second =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{9});
    second.Commit({9, 8, 7});
  }

  auto lease =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{9, 8, 7, 6});
  Expect(lease.cache_hit(), "one available entry matches");
  Expect(lease.cached_tokens() == 3, "longest exact prefix is selected");
  Expect(dynamic_cast<FakeState&>(lease.state()).id == 1,
         "matching state is selected rather than an arbitrary slot");
  lease.Commit({9, 8, 7, 6});
}

void TestWaitingAcquireCanBeCancelled() {
  std::vector<std::size_t> invalidations(1);
  gufo::server::ContinuationCache cache(
      1, [&] { return std::make_unique<FakeState>(0, &invalidations); });
  auto held =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3});
  std::atomic<bool> cancelled{true};

  auto cancelled_lease =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3, 4},
                    [&] { return cancelled.load(); });
  Expect(!cancelled_lease, "cancelled waiter does not acquire a state");
  held.Commit({1, 2, 3});
}

void TestSnapshotCanBranchIntoTwoIndependentStateSlots() {
  std::vector<std::size_t> invalidations(2);
  std::size_t next_id = 0;
  gufo::server::ContinuationCache cache(
      2, [&] { return std::make_unique<FakeState>(next_id++, &invalidations); },
      {
          .restore =
              [](gufo::server::ContinuationState& state,
                 const gufo::server::ContinuationSnapshot& snapshot) {
                auto& fake = dynamic_cast<FakeState&>(state);
                const auto& saved = dynamic_cast<const FakeSnapshot&>(snapshot);
                fake.value = saved.value;
              },
          .capacity_bytes = [] { return 1024; },
          .on_event = {},
      });

  {
    auto root =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3});
    dynamic_cast<FakeState&>(root.state()).value = 7;
    Expect(root.TryReserveSnapshot(sizeof(std::size_t), 3),
           "root snapshot reserves aggregate capacity before allocation");
    root.Commit({1, 2, 3}, std::make_unique<FakeSnapshot>(7));
  }

  auto first =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3, 4});
  auto second =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3, 5});
  Expect(first.cache_hit() && second.cache_hit(),
         "one snapshot can satisfy two simultaneous leases");
  auto& first_state = dynamic_cast<FakeState&>(first.state());
  auto& second_state = dynamic_cast<FakeState&>(second.state());
  Expect(first_state.id != second_state.id,
         "snapshot branches use different mutable state slots");
  Expect(first_state.value == 7 && second_state.value == 7,
         "both mutable states restore the root payload");

  first_state.value = 8;
  second_state.value = 9;
  Expect(first.TryReserveSnapshot(sizeof(std::size_t), 4),
         "first branch reserves snapshot capacity");
  Expect(second.TryReserveSnapshot(sizeof(std::size_t), 4),
         "second branch reserves snapshot capacity");
  first.Commit({1, 2, 3, 4}, std::make_unique<FakeSnapshot>(8));
  second.Commit({1, 2, 3, 5}, std::make_unique<FakeSnapshot>(9));

  auto root_again =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3, 6});
  Expect(root_again.cache_hit() && root_again.cached_tokens() == 3,
         "branch commits preserve the shared root snapshot");
  Expect(dynamic_cast<FakeState&>(root_again.state()).value == 7,
         "branch mutation never changes the immutable root payload");
  root_again.Invalidate();
}

void TestByteCapacityEvictsBeforeSnapshotAllocation() {
  using gufo::server::SnapshotEventAction;
  using gufo::server::SnapshotEventReason;

  std::vector<std::size_t> invalidations(2);
  std::size_t next_id = 0;
  std::vector<gufo::server::SnapshotEvent> events;
  gufo::server::ContinuationCache cache(
      2, [&] { return std::make_unique<FakeState>(next_id++, &invalidations); },
      {
          .restore =
              [](gufo::server::ContinuationState& state,
                 const gufo::server::ContinuationSnapshot& snapshot) {
                dynamic_cast<FakeState&>(state).value =
                    dynamic_cast<const FakeSnapshot&>(snapshot).value;
              },
          .capacity_bytes = [] { return 12; },
          .on_event =
              [&](const gufo::server::SnapshotEvent& event) {
                events.push_back(event);
              },
      });

  {
    auto root =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3});
    Expect(root.TryReserveSnapshot(8, 3),
           "initial snapshot fits the byte budget");
    Expect(root.Commit({1, 2, 3}, std::make_unique<FakeSnapshot>(7, 8)) == 8,
           "initial snapshot is retained");
  }
  Expect(cache.retained_snapshot_bytes() == 8,
         "retained bytes account the initial snapshot");

  auto replacement =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{9, 8, 7});
  Expect(replacement.TryReserveSnapshot(12, 3),
         "reservation evicts stale bytes before snapshot allocation");
  Expect(cache.retained_snapshot_bytes() == 0 &&
             cache.reserved_snapshot_bytes() == 12,
         "eviction transfers budget from retained to reserved bytes");
  Expect(events.size() == 1 &&
             events.front().action == SnapshotEventAction::kRemoved &&
             events.front().reason == SnapshotEventReason::kByteCapacity &&
             events.front().snapshot_bytes == 8 &&
             events.front().token_count == 3,
         "byte-pressure removal reports sanitized reason and dimensions");
  Expect(replacement.Commit({9, 8, 7}, std::make_unique<FakeSnapshot>(9, 12)) ==
             12,
         "reserved replacement is retained");
  Expect(cache.retained_snapshot_bytes() == 12 &&
             cache.reserved_snapshot_bytes() == 0,
         "commit converts the reservation into exact retained bytes");
}

void TestConcurrentReservationsCannotOvercommitBudget() {
  using gufo::server::SnapshotEventAction;
  using gufo::server::SnapshotEventReason;

  std::vector<std::size_t> invalidations(2);
  std::size_t next_id = 0;
  std::vector<gufo::server::SnapshotEvent> events;
  gufo::server::ContinuationCache cache(
      2, [&] { return std::make_unique<FakeState>(next_id++, &invalidations); },
      {
          .restore = [](gufo::server::ContinuationState&,
                        const gufo::server::ContinuationSnapshot&) {},
          .capacity_bytes = [] { return 12; },
          .on_event =
              [&](const gufo::server::SnapshotEvent& event) {
                events.push_back(event);
              },
      });

  auto first = cache.Acquire(std::vector<gufo::server::ContinuationToken>{1});
  auto second = cache.Acquire(std::vector<gufo::server::ContinuationToken>{2});
  Expect(first.TryReserveSnapshot(8, 1),
         "first in-flight snapshot reserves bytes");
  Expect(!second.TryReserveSnapshot(8, 1),
         "second reservation is rejected instead of overcommitting");
  Expect(cache.retained_snapshot_bytes() == 0 &&
             cache.reserved_snapshot_bytes() == 8,
         "only the admitted in-flight reservation is accounted");
  Expect(events.size() == 1 &&
             events.front().action == SnapshotEventAction::kSkipped &&
             events.front().reason == SnapshotEventReason::kByteCapacity &&
             events.front().snapshot_bytes == 8 &&
             events.front().token_count == 1,
         "capacity refusal emits a sanitized skip event");

  second.Commit({2});
  first.Commit({1}, std::make_unique<FakeSnapshot>(1, 8));
  Expect(cache.retained_snapshot_bytes() == 8 &&
             cache.reserved_snapshot_bytes() == 0,
         "completed requests leave no leaked reservation");
}

void TestImpossibleReservationPreservesRetainedEntries() {
  using gufo::server::SnapshotEventAction;
  using gufo::server::SnapshotEventReason;

  std::vector<std::size_t> invalidations(1);
  std::vector<gufo::server::SnapshotEvent> events;
  gufo::server::ContinuationCache cache(
      1, [&] { return std::make_unique<FakeState>(0, &invalidations); },
      {
          .restore =
              [](gufo::server::ContinuationState& state,
                 const gufo::server::ContinuationSnapshot& snapshot) {
                dynamic_cast<FakeState&>(state).value =
                    dynamic_cast<const FakeSnapshot&>(snapshot).value;
              },
          .capacity_bytes = [] { return 8; },
          .on_event =
              [&](const gufo::server::SnapshotEvent& event) {
                events.push_back(event);
              },
      });

  {
    auto root =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2});
    Expect(root.TryReserveSnapshot(8, 2), "root fills the byte budget");
    root.Commit({1, 2}, std::make_unique<FakeSnapshot>(7, 8));
  }
  {
    auto oversized =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{9});
    Expect(!oversized.TryReserveSnapshot(9, 1),
           "snapshot larger than the total budget is rejected");
    oversized.Commit({9});
  }

  Expect(cache.retained_snapshot_bytes() == 8,
         "impossible admission does not evict a useful retained entry");
  Expect(events.size() == 1 &&
             events.front().action == SnapshotEventAction::kSkipped &&
             events.front().reason == SnapshotEventReason::kByteCapacity &&
             events.front().snapshot_bytes == 9,
         "oversized refusal emits one skip and no removal");
  auto extension =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3});
  Expect(extension.cache_hit() && extension.cached_tokens() == 2,
         "retained root remains reusable after oversized refusal");
  extension.Invalidate();
}

void TestAbandonedReservationIsReleased() {
  std::vector<std::size_t> invalidations(1);
  gufo::server::ContinuationCache cache(
      1, [&] { return std::make_unique<FakeState>(0, &invalidations); },
      {
          .restore = [](gufo::server::ContinuationState&,
                        const gufo::server::ContinuationSnapshot&) {},
          .capacity_bytes = [] { return 8; },
          .on_event = {},
      });

  {
    auto abandoned =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{1});
    Expect(abandoned.TryReserveSnapshot(8, 1),
           "abandoned request owns the whole reservation");
  }
  Expect(cache.reserved_snapshot_bytes() == 0,
         "lease destruction releases its in-flight reservation");

  auto retry = cache.Acquire(std::vector<gufo::server::ContinuationToken>{2});
  Expect(retry.TryReserveSnapshot(8, 1),
         "released bytes are immediately available to another request");
  retry.Commit({2}, std::make_unique<FakeSnapshot>(2, 8));
}

void TestReservationMismatchSkipsRetentionWithoutFailingCommit() {
  using gufo::server::SnapshotEventAction;
  using gufo::server::SnapshotEventReason;

  std::vector<std::size_t> invalidations(1);
  std::vector<gufo::server::SnapshotEvent> events;
  gufo::server::ContinuationCache cache(
      1, [&] { return std::make_unique<FakeState>(0, &invalidations); },
      {
          .restore = [](gufo::server::ContinuationState&,
                        const gufo::server::ContinuationSnapshot&) {},
          .capacity_bytes = [] { return 16; },
          .on_event =
              [&](const gufo::server::SnapshotEvent& event) {
                events.push_back(event);
              },
      });

  auto lease =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{4, 5});
  Expect(lease.TryReserveSnapshot(4, 2),
         "snapshot estimate reserves before allocation");
  Expect(lease.Commit({4, 5}, std::make_unique<FakeSnapshot>(1, 8)) == 0,
         "an underestimated snapshot is skipped without failing commit");
  Expect(cache.retained_snapshot_bytes() == 0 &&
             cache.reserved_snapshot_bytes() == 0,
         "mismatch releases the complete reservation");
  Expect(
      events.size() == 1 &&
          events.front().action == SnapshotEventAction::kSkipped &&
          events.front().reason == SnapshotEventReason::kReservationMismatch &&
          events.front().snapshot_bytes == 8 && events.front().token_count == 2,
      "reservation mismatch is observable without prompt content");

  auto retry =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{4, 5, 6});
  Expect(!retry.cache_hit(), "skipped snapshot is a deterministic cache miss");
  retry.Invalidate();
}

void TestEntryReplacementLogsRemovedSnapshot() {
  using gufo::server::SnapshotEventAction;
  using gufo::server::SnapshotEventReason;

  std::vector<std::size_t> invalidations(1);
  std::vector<gufo::server::SnapshotEvent> events;
  gufo::server::ContinuationCache cache(
      1, [&] { return std::make_unique<FakeState>(0, &invalidations); },
      {
          .restore = [](gufo::server::ContinuationState&,
                        const gufo::server::ContinuationSnapshot&) {},
          .capacity_bytes = [] { return 32; },
          .on_event =
              [&](const gufo::server::SnapshotEvent& event) {
                events.push_back(event);
              },
          .entry_capacity = 1,
      });

  {
    auto first =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2});
    Expect(first.TryReserveSnapshot(8, 2), "first entry reserves bytes");
    first.Commit({1, 2}, std::make_unique<FakeSnapshot>(1, 8));
  }
  {
    auto second =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{9, 8, 7});
    Expect(second.TryReserveSnapshot(8, 3), "replacement reserves bytes");
    second.Commit({9, 8, 7}, std::make_unique<FakeSnapshot>(2, 8));
  }

  Expect(events.size() == 1 &&
             events.front().action == SnapshotEventAction::kRemoved &&
             events.front().reason == SnapshotEventReason::kEntryCapacity &&
             events.front().snapshot_bytes == 8 &&
             events.front().token_count == 2,
         "entry replacement logs the removed snapshot dimensions");
  Expect(cache.retained_snapshot_bytes() == 8,
         "entry replacement keeps exact aggregate accounting");
}

void TestOneStateRetainsThreeInterleavedConversations() {
  SnapshotCacheFixture fixture(1, 128);
  auto& cache = fixture.cache;
  Expect(cache.capacity() == 1 && cache.snapshot_entry_capacity() == 24 &&
             fixture.states_created == 1,
         "default snapshot capacity is independent of model state allocation");
  for (const auto chat : {1U, 2U, 3U})
    fixture.Save({chat, 10}, chat, {chat, 10, 11});
  Expect(cache.retained_snapshot_bytes() == 24,
         "three immutable prompts survive reuse of one mutable state");

  for (const auto chat : {1U, 2U, 3U}) {
    auto followup = cache.Acquire(
        std::vector<gufo::server::ContinuationToken>{chat, 10, 11, 12});
    Expect(followup.cache_hit() && followup.cached_tokens() == 2 &&
               followup.restored_snapshot_bytes() == 8,
           "interleaved follow-up restores its own prompt checkpoint");
    Expect(dynamic_cast<FakeState&>(followup.state()).value == chat,
           "interleaved conversations restore distinct payloads");
    Expect(followup.TryReserveSnapshot(8, 4),
           "follow-up snapshot uses remaining retention budget");
    followup.Commit({chat, 10, 11, 12},
                    std::make_unique<FakeSnapshot>(chat + 10, 8),
                    {chat, 10, 11, 12, 13});
  }
  auto immediate = cache.Acquire(
      std::vector<gufo::server::ContinuationToken>{3, 10, 11, 12, 13, 14});
  Expect(immediate.cached_tokens() == 5 &&
             immediate.restored_snapshot_bytes() == 0,
         "independent retention preserves copy-free immediate live reuse");
  Expect(fixture.states_created == 1 && fixture.events.empty(),
         "retaining chats creates no extra model states or evictions");
}

void TestIndependentEntryAndByteBoundsUseLru() {
  using gufo::server::SnapshotEventAction;
  using gufo::server::SnapshotEventReason;
  for (const bool entry_limited : {true, false}) {
    SnapshotCacheFixture fixture(1, entry_limited ? 64 : 24,
                                 entry_limited ? 3 : 24);
    auto& cache = fixture.cache;
    fixture.Save({1}, 1);
    fixture.Save({2}, 2);
    fixture.Save({3}, 3);
    {
      auto recent =
          cache.Acquire(std::vector<gufo::server::ContinuationToken>{1});
      Expect(recent.cache_hit(), "reading an older snapshot touches its LRU");
    }
    fixture.Save({4}, 4);
    Expect(cache.retained_snapshot_bytes() == 24 &&
               cache.reserved_snapshot_bytes() == 0,
           "independent limits retain exactly three accounted payloads");
    Expect(fixture.events.size() == 1 &&
               fixture.events[0].action == SnapshotEventAction::kRemoved &&
               fixture.events[0].reason ==
                   (entry_limited ? SnapshotEventReason::kEntryCapacity
                                  : SnapshotEventReason::kByteCapacity),
           "the binding capacity reports its eviction reason");
    {
      auto evicted =
          cache.Acquire(std::vector<gufo::server::ContinuationToken>{2});
      Expect(!evicted.cache_hit(), "least-recently-used snapshot is evicted");
    }
    for (const auto chat : {1U, 3U, 4U}) {
      auto retained =
          cache.Acquire(std::vector<gufo::server::ContinuationToken>{chat});
      Expect(retained.cache_hit() &&
                 dynamic_cast<FakeState&>(retained.state()).value == chat,
             "recent snapshots remain restorable under either capacity");
    }
  }
}

void TestLiveHitTouchesFallbackAndFailedCapturePreservesIt() {
  SnapshotCacheFixture fixture(1, 64, 2);
  auto& cache = fixture.cache;
  fixture.Save({1}, 1, {1, 10});
  fixture.Save({2}, 2);
  {
    auto first = cache.Acquire(std::vector<gufo::server::ContinuationToken>{1});
    first.Commit({1}, nullptr, {1, 10});
  }
  // Touch the other snapshot without disturbing the live frontier.
  {
    auto live =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 10});
    Expect(live.HasSnapshotFor(std::vector<gufo::server::ContinuationToken>{2}),
           "exact snapshot availability touches its LRU");
    live.Commit({1}, nullptr, {1, 10});
  }
  {
    auto live =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 10, 11});
    Expect(live.cache_hit() && live.restored_snapshot_bytes() == 0,
           "live extension avoids restoration");
    Expect(live.TryReserveSnapshot(8, 3), "capture reserves capacity");
    live.SkipSnapshot(gufo::server::SnapshotEventReason::kCaptureFailure, 8, 3);
    live.Invalidate();
  }
  Expect(cache.reserved_snapshot_bytes() == 0,
         "capture failure releases all reserved bytes");
  fixture.Save({3}, 3);
  auto fallback =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 9});
  Expect(fallback.cache_hit() && fallback.cached_tokens() == 1,
         "a live hit refreshes its immutable fallback despite capture failure");
}

void TestExactReplacementDoesNotConsumeAnotherSnapshotSlot() {
  SnapshotCacheFixture fixture(1, 64, 2);
  fixture.Save({1}, 1);
  fixture.Save({2}, 2);
  fixture.Save({1}, 10);
  Expect(fixture.cache.retained_snapshot_bytes() == 16 &&
             fixture.events.size() == 1 &&
             fixture.events[0].reason ==
                 gufo::server::SnapshotEventReason::kExactReplacement,
         "exact replacement preserves the other slot and aggregate bytes");
  auto updated =
      fixture.cache.Acquire(std::vector<gufo::server::ContinuationToken>{1});
  Expect(dynamic_cast<FakeState&>(updated.state()).value == 10,
         "exact replacement publishes the new opaque payload");
}

void TestSnapshotSlotsCannotBecomeMutableStates() {
  SnapshotCacheFixture fixture(1, 64, 3);
  fixture.Save({1}, 1);
  fixture.Save({2}, 2);
  auto held =
      fixture.cache.Acquire(std::vector<gufo::server::ContinuationToken>{1});
  auto cancelled = fixture.cache.Acquire(
      std::vector<gufo::server::ContinuationToken>{2}, [] { return true; });
  Expect(!cancelled && fixture.states_created == 1,
         "an available snapshot cannot bypass the mutable concurrency bound");
}

void TestRestoringSnapshotStaysAccountedUntilReaderFinishes() {
  using gufo::server::SnapshotEventReason;
  for (const bool fail_restore : {false, true}) {
    std::vector<std::size_t> invalidations(2);
    std::size_t states_created = 0;
    std::vector<gufo::server::SnapshotEvent> events;
    std::promise<void> restore_started;
    std::promise<void> finish_restore;
    auto finish = finish_restore.get_future();
    gufo::server::ContinuationCache cache(
        2,
        [&] {
          return std::make_unique<FakeState>(states_created++, &invalidations);
        },
        {
            .restore =
                [&](auto& state, const auto& snapshot) {
                  restore_started.set_value();
                  finish.wait();
                  if (fail_restore)
                    throw std::runtime_error("injected restore failure");
                  dynamic_cast<FakeState&>(state).value =
                      dynamic_cast<const FakeSnapshot&>(snapshot).value;
                },
            .capacity_bytes = [] { return 16; },
            .on_event = [&](const auto& event) { events.push_back(event); },
            .entry_capacity = 1,
        });
    auto saved = std::make_shared<FakeSnapshot>(7, 8);
    const std::weak_ptr<const FakeSnapshot> original = saved;
    {
      auto root =
          cache.Acquire(std::vector<gufo::server::ContinuationToken>{1});
      Expect(root.TryReserveSnapshot(8, 1), "root snapshot reserves capacity");
      root.Commit({1}, std::move(saved));
    }
    auto restoring = std::async(std::launch::async, [&] {
      try {
        auto lease =
            cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2});
        Expect(dynamic_cast<FakeState&>(lease.state()).value == 7,
               "restore reader sees the original immutable payload");
      } catch (const std::runtime_error&) {
        return true;
      }
      return false;
    });
    restore_started.get_future().wait();
    {
      auto writer =
          cache.Acquire(std::vector<gufo::server::ContinuationToken>{9});
      Expect(!writer.TryReserveSnapshot(16, 1) &&
                 cache.retained_snapshot_bytes() == 8 && !original.expired(),
             "byte pressure cannot unaccount or destroy an active restore");
      Expect(writer.TryReserveSnapshot(8, 1),
             "remaining bytes can still reserve a candidate payload");
      Expect(writer.Commit({1}, std::make_unique<FakeSnapshot>(9, 8)) == 0,
             "exact replacement cannot overwrite a snapshot being restored");
    }
    Expect(cache.retained_snapshot_bytes() == 8 &&
               cache.reserved_snapshot_bytes() == 0 && events.size() == 2 &&
               events.back().reason == SnapshotEventReason::kEntryCapacity,
           "busy-slot refusal releases reservations without evicting readers");
    finish_restore.set_value();
    Expect(restoring.get() == fail_restore,
           "restore failures propagate after releasing the reader pin");
    auto replacement =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{9});
    Expect(replacement.TryReserveSnapshot(16, 1) && original.expired(),
           "completed or failed restore immediately permits budget eviction");
    replacement.Commit({9}, std::make_unique<FakeSnapshot>(9, 16));
    Expect(cache.retained_snapshot_bytes() == 16 &&
               cache.reserved_snapshot_bytes() == 0,
           "post-restore replacement accounts only its own payload");
  }
}

void TestZeroByteBudgetRetainsOnlyTheLiveFrontier() {
  SnapshotCacheFixture fixture(1, 0);
  auto& cache = fixture.cache;
  Expect(cache.snapshot_entry_capacity() == 24 &&
             cache.snapshot_capacity_bytes() == 0,
         "entry capacity cannot override an exhausted byte budget");
  {
    auto first = cache.Acquire(std::vector<gufo::server::ContinuationToken>{1});
    Expect(!first.TryReserveSnapshot(8, 1),
           "zero budget refuses allocation before capture");
    first.Commit({1}, nullptr, {1, 2});
  }
  {
    auto live =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3});
    Expect(live.cache_hit() && live.cached_tokens() == 2 &&
               live.restored_snapshot_bytes() == 0,
           "live continuation remains usable with zero snapshot budget");
    live.Commit({1, 2, 3}, nullptr, {1, 2, 3});
  }
  {
    auto other = cache.Acquire(std::vector<gufo::server::ContinuationToken>{9});
    other.Commit({9}, nullptr, {9});
  }
  auto lost =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3});
  Expect(!lost.cache_hit() && cache.retained_snapshot_bytes() == 0 &&
             cache.reserved_snapshot_bytes() == 0,
         "interleaved retention is not promised when no snapshot fits");
}

void TestLeaseSourceIdentitySurvivesSnapshotSlotReuse() {
  SnapshotCacheFixture fixture(2, 64, 2);
  auto& cache = fixture.cache;
  fixture.Save({1}, 1);
  fixture.Save({2}, 2);
  auto old_source =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 3});
  {
    auto writer =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{9});
    Expect(writer.TryReserveSnapshot(8, 1), "exact replacement reserves bytes");
    writer.PublishSnapshot({1}, std::make_unique<FakeSnapshot>(10, 8));
    Expect(
        writer.HasSnapshotFor(std::vector<gufo::server::ContinuationToken>{2}),
        "other snapshot becomes newer than the reused source slot");
    writer.Invalidate();
  }
  Expect(old_source.TryReserveSnapshot(8, 2), "branch snapshot reserves bytes");
  old_source.Commit({1, 3}, std::make_unique<FakeSnapshot>(13, 8));
  {
    auto other = cache.Acquire(std::vector<gufo::server::ContinuationToken>{2});
    Expect(
        other.cache_hit(),
        "a stale lease source cannot shield the replacement in its old slot");
  }
  auto replaced =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{1});
  Expect(
      !replaced.cache_hit(),
      "LRU evicts the replacement rather than treating it as the old source");
}

void TestIndependentSnapshotsKeepIdentityAndLongestPrefixIsolation() {
  SnapshotCacheFixture fixture(1, 64, 4);
  auto& cache = fixture.cache;
  const std::vector<std::uint8_t> first_identity{1}, second_identity{2};
  const auto save = [&](std::vector<gufo::server::ContinuationToken> tokens,
                        const std::vector<std::uint8_t>& identity,
                        std::size_t value) {
    auto lease = cache.Acquire(tokens, {}, identity);
    Expect(lease.TryReserveSnapshot(8, tokens.size()),
           "identity snapshot reserves bytes");
    lease.Commit(std::move(tokens), std::make_unique<FakeSnapshot>(value, 8));
  };
  save({1}, first_identity, 1);
  save({1, 2}, first_identity, 2);
  save({1, 2}, second_identity, 3);
  for (const auto& identity : {first_identity, second_identity}) {
    auto hit = cache.Acquire(
        std::vector<gufo::server::ContinuationToken>{1, 2, 3}, {}, identity);
    Expect(hit.cached_tokens() == 2 &&
               dynamic_cast<FakeState&>(hit.state()).value ==
                   (identity == first_identity ? 2 : 3),
           "longest snapshot matches both tokens and input identity");
  }
  auto no_identity =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3});
  Expect(!no_identity.cache_hit(),
         "text-only input cannot reuse an image snapshot");
}

void TestRejectedSnapshotIsDestroyedBeforeReentrantAdmission() {
  using gufo::server::SnapshotEventAction;
  using gufo::server::SnapshotEventReason;
  std::atomic<std::size_t> live_bytes{0};
  std::vector<std::size_t> invalidations(3);
  std::size_t next_id = 0;
  std::promise<void> restore_started;
  std::promise<void> finish_restore;
  auto finish = finish_restore.get_future();
  gufo::server::ContinuationCache::Lease observer;
  bool callback_ran = false;
  bool destruction_checked = false;
  gufo::server::ContinuationCache cache(
      3, [&] { return std::make_unique<FakeState>(next_id++, &invalidations); },
      {
          .restore =
              [&](auto&, const auto&) {
                restore_started.set_value();
                finish.wait();
              },
          .capacity_bytes = [] { return 16; },
          .on_event =
              [&](const auto& event) {
                if (event.action != SnapshotEventAction::kSkipped ||
                    event.reason != SnapshotEventReason::kEntryCapacity)
                  return;
                callback_ran = true;
                Expect(observer.TryReserveSnapshot(8, 1),
                       "callback may reserve only after rejected payload is "
                       "destroyed");
                auto third = std::make_shared<TrackedSnapshot>(live_bytes, 8);
                Expect(live_bytes <= 16,
                       "reentrant callback cannot exceed the physical payload "
                       "budget");
                third.reset();
                observer.SkipSnapshot(SnapshotEventReason::kCaptureFailure, 8,
                                      1);
              },
          .entry_capacity = 1,
      });
  {
    auto root = cache.Acquire(std::vector<gufo::server::ContinuationToken>{1});
    Expect(root.TryReserveSnapshot(8, 1), "root snapshot reserves bytes");
    root.Commit({1}, std::make_shared<TrackedSnapshot>(live_bytes, 8));
  }
  auto restoring = std::async(std::launch::async, [&] {
    auto lease =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2});
  });
  restore_started.get_future().wait();
  observer = cache.Acquire(std::vector<gufo::server::ContinuationToken>{9});
  auto writer = cache.Acquire(std::vector<gufo::server::ContinuationToken>{8});
  Expect(writer.TryReserveSnapshot(8, 1), "candidate reserves remaining bytes");
  Expect(
      writer.Commit(
          {8},
          std::make_shared<TrackedSnapshot>(
              live_bytes, 8,
              [&] {
                destruction_checked = true;
                Expect(cache.retained_snapshot_bytes() +
                               cache.reserved_snapshot_bytes() ==
                           16,
                       "rejected candidate stays charged during destruction");
                Expect(!observer.TryReserveSnapshot(8, 1),
                       "destructor cannot spend the rejected candidate bytes");
              })) == 0,
      "pinned entry rejects the candidate");
  Expect(callback_ran && destruction_checked && live_bytes == 8 &&
             cache.reserved_snapshot_bytes() == 0,
         "rejection and reentrant callback release candidate reservations");
  finish_restore.set_value();
  restoring.get();
  observer.Invalidate();
}

void TestEvictedSnapshotStaysChargedThroughDestruction() {
  using gufo::server::SnapshotEventReason;
  for (const bool byte_eviction : {true, false}) {
    std::atomic<std::size_t> live_bytes{0};
    std::vector<std::size_t> invalidations(2);
    std::size_t next_id = 0;
    gufo::server::ContinuationCache::Lease observer;
    bool destruction_checked = false;
    gufo::server::ContinuationCache cache(
        2,
        [&] { return std::make_unique<FakeState>(next_id++, &invalidations); },
        {
            .restore = [](auto&, const auto&) {},
            .capacity_bytes = [] { return 16; },
            .on_event = {},
            .entry_capacity = 1,
        });
    {
      auto root =
          cache.Acquire(std::vector<gufo::server::ContinuationToken>{1});
      const std::size_t bytes = byte_eviction ? 16 : 8;
      Expect(root.TryReserveSnapshot(bytes, 1),
             "retired snapshot reserves bytes");
      root.Commit(
          {1}, std::make_shared<TrackedSnapshot>(live_bytes, bytes, [&] {
            destruction_checked = true;
            Expect(cache.retained_snapshot_bytes() +
                           cache.reserved_snapshot_bytes() ==
                       16,
                   "retired payload stays accounted during destruction");
            Expect(!observer.TryReserveSnapshot(16, 1),
                   "destructor cannot reuse bytes still occupied by payloads");
            Expect(live_bytes <= 16,
                   "destruction respects the physical budget");
          }));
    }
    observer = cache.Acquire(std::vector<gufo::server::ContinuationToken>{9});
    auto writer =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{8});
    Expect(writer.TryReserveSnapshot(8, 1),
           "writer reserves after byte reclamation");
    Expect(writer.Commit({8},
                         std::make_shared<TrackedSnapshot>(live_bytes, 8)) == 8,
           "replacement is retained after retired payload destruction");
    Expect(destruction_checked && live_bytes == 8 &&
               cache.retained_snapshot_bytes() == 8 &&
               cache.reserved_snapshot_bytes() == 0,
           "retired bytes are released exactly once after destruction");
    Expect(observer.TryReserveSnapshot(8, 1),
           "destruction completion makes bytes available to other requests");
    observer.SkipSnapshot(SnapshotEventReason::kCaptureFailure, 8, 1);
    observer.Invalidate();
  }
}

struct SharedStorageSnapshot final : gufo::server::ContinuationSnapshot {
  std::vector<gufo::server::ContinuationSnapshotStorageOwner> owners;
  explicit SharedStorageSnapshot(
      std::vector<gufo::server::ContinuationSnapshotStorageOwner> storage)
      : owners(std::move(storage)) {}
  std::size_t PayloadBytes() const noexcept override { return 12; }
  std::span<const gufo::server::ContinuationSnapshotStorageOwner>
  StorageOwners() const noexcept override {
    return owners;
  }
};

void TestPhysicalOwnerDeduplicationAndExternalLifetime() {
  std::vector<std::size_t> invalidations(1);
  gufo::server::ContinuationCache cache(
      1, [&] { return std::make_unique<FakeState>(0, &invalidations); },
      {.restore = [](auto&, const auto&) {},
       .capacity_bytes = [] { return 40; },
       .on_event = {},
       .entry_capacity = 2});
  auto common = std::make_shared<const std::array<std::uint8_t, 8>>();
  const std::weak_ptr<const void> weak_common = common;
  const auto capture = [&](unsigned token) {
    auto lease =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{token});
    Expect(lease.TryReserveSnapshot(12, 1),
           "full conservative allocation fits");
    auto capsule = std::make_shared<const std::array<std::uint8_t, 4>>();
    auto snapshot = std::make_shared<SharedStorageSnapshot>(
        std::vector<gufo::server::ContinuationSnapshotStorageOwner>{
            {common, sizeof(*common)}, {capsule, sizeof(*capsule)}});
    Expect(lease.Commit({token}, snapshot) == 12,
           "logical payload remains complete while shared storage is "
           "deduplicated");
    return snapshot;
  };
  auto first = capture(1);
  auto second = capture(2);
  Expect(cache.retained_snapshot_bytes() == 16,
         "two real capsules share one charged immutable history allocation");
  common.reset();
  second.reset();
  auto replacement =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{9});
  Expect(replacement.TryReserveSnapshot(24, 1),
         "replacement reserves full upper bound");
  replacement.Commit({9}, std::make_unique<FakeSnapshot>(9, 24));
  Expect(cache.retained_snapshot_bytes() == 40 && !weak_common.expired(),
         "external reader keeps evicted physical owners charged");
  first.reset();
  Expect(cache.retained_snapshot_bytes() == 36,
         "external reader releases capsule but surviving child owns shared "
         "history");
  auto last = cache.Acquire(std::vector<gufo::server::ContinuationToken>{10});
  Expect(last.TryReserveSnapshot(40, 1), "eviction reclaims all unique owners");
  Expect(
      weak_common.expired() && cache.retained_snapshot_bytes() == 0,
      "sole-owned accounting entries are reclaimed after their final readers");
  last.SkipSnapshot(gufo::server::SnapshotEventReason::kCaptureFailure, 40, 1);
}

void TestDeltaReservationAdmitsGrowingSharedHistory() {
  // Mirrors the 2026-09-26 A/B failure: a conversation whose earlier
  // snapshot is retained needs only its tail delta, but a reservation for
  // the full allocation exceeded the byte budget and every capture was
  // skipped. With the incremental estimate the same growth is admitted.
  std::vector<std::size_t> invalidations(1);
  gufo::server::ContinuationCache cache(
      1, [&] { return std::make_unique<FakeState>(0, &invalidations); },
      {.restore = [](auto&, const auto&) {},
       .capacity_bytes = [] { return 40; },
       .on_event = {},
       .entry_capacity = 2});
  auto history = std::make_shared<const std::array<std::uint8_t, 8>>();
  const auto capture = [&](unsigned token, std::size_t reservation,
                           std::size_t tail_bytes) {
    auto lease =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{token});
    // The runner reserves the incremental estimate (shared history minus
    // inherited bytes plus headroom); the full allocation would not fit.
    Expect(lease.TryReserveSnapshot(reservation, 1),
           "reservation fits the byte budget");
    auto tail = std::make_shared<const std::array<std::uint8_t, 8>>();
    auto snapshot = std::make_shared<SharedStorageSnapshot>(
        std::vector<gufo::server::ContinuationSnapshotStorageOwner>{
            {history, sizeof(*history)}, {tail, tail_bytes}});
    Expect(lease.Commit({token}, snapshot) == 12,
           "logical payload stays complete across delta reservations");
  };
  // First capture: no retained parent, so the reservation covers everything
  // the snapshot adds (shared history owner + tail).
  capture(1, 12, 4);
  // Second turn: the full-allocation reservation would exceed the 40-byte
  // budget beside the 12 retained bytes; the delta (8-byte tail) fits
  // because the shared history owner is already charged.
  capture(2, 8, 8);
  Expect(cache.retained_snapshot_bytes() == 20,
         "retained accounting charges the shared history once");
  auto late = cache.Acquire(std::vector<gufo::server::ContinuationToken>{3});
  Expect(!late.TryReserveSnapshot(40, 1),
         "a full-allocation reservation still cannot fit the spent budget");
  late.SkipSnapshot(gufo::server::SnapshotEventReason::kCaptureFailure, 0, 1);
}

void TestProtectedPublishSurvivesPromptAdmissionAndSlotPressure() {
  for (const std::size_t entries : {1, 2}) {
    SnapshotCacheFixture fixture(1, 24, entries);
    auto& cache = fixture.cache;
    auto lease =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3});
    Expect(lease.TryReserveSnapshot(8, 1, true, false),
           "early capture admits with regular snapshot headroom");
    Expect(
        lease.PublishSnapshot({1}, std::make_unique<FakeSnapshot>(1, 8)) == 8,
        "early snapshot publishes without releasing mutable lease");
    Expect(!lease.TryReserveSnapshot(24, 3),
           "later reservation cannot evict protected early fallback");
    Expect(lease.TryReserveSnapshot(8, 3),
           "ordinary prompt reserves remaining bytes");
    const auto retained = lease.Commit(
        {1, 2, 3}, std::make_unique<FakeSnapshot>(3, 8), {1, 2, 3, 4});
    Expect(retained == (entries == 1 ? 0 : 8),
           "entry pressure also honors protected source until publication");
    auto branch =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 9});
    Expect(branch.cached_tokens() == 1,
           "successful or refused prompt retains viable earlier branch state");
  }
}

void TestLearnedPrefixesBoundIdentityAgeAndExactTokens() {
  SnapshotCacheFixture fixture(1, 64);
  fixture.Save({1, 2, 3, 4}, 4);
  const std::vector<gufo::server::ContinuationToken> branch{1, 2, 3, 9};
  auto learned = fixture.cache.LearnPrefixBoundaries(branch, {}, 2);
  Expect(learned == std::vector<std::size_t>{3},
         "compatible divergence trains exact RAM prefix metadata");
  const std::vector<std::uint8_t> image{1};
  Expect(fixture.cache.LearnPrefixBoundaries(branch, image, 2).empty(),
         "learned boundaries never cross input identities");
  Expect(
      fixture.cache
          .LearnPrefixBoundaries(
              std::vector<gufo::server::ContinuationToken>{8, 2, 3, 9}, {}, 2)
          .empty(),
      "same lengths do not imply exact prefix identity");
  Expect(fixture.cache.LearnPrefixBoundaries(branch, {}, 4).empty(),
         "minimum saved-work threshold rejects short prefixes");
  // Remove the original training source then age the metadata independently.
  auto evict =
      fixture.cache.Acquire(std::vector<gufo::server::ContinuationToken>{8});
  Expect(evict.TryReserveSnapshot(64, 1), "training source can be evicted");
  evict.Commit({8}, std::make_unique<FakeSnapshot>(8, 64));
  for (std::size_t i = 0; i < 129; ++i)
    (void)fixture.cache.LearnPrefixBoundaries(
        std::vector<gufo::server::ContinuationToken>{7}, {}, 2);
  Expect(fixture.cache.LearnPrefixBoundaries(branch, {}, 2).empty(),
         "aged learned metadata cannot accumulate indefinitely");
}

void TestPeerCannotEvictProtectedCaptureSource() {
  SnapshotCacheFixture fixture(2, 24, 2);
  fixture.Save({1}, 1);
  auto source =
      fixture.cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2});
  Expect(source.TryReserveSnapshot(8, 2, true), "source pins before capture");
  auto peer =
      fixture.cache.Acquire(std::vector<gufo::server::ContinuationToken>{9});
  Expect(!peer.TryReserveSnapshot(24, 1),
         "peer cannot evict pinned fallback or spend pending allocation");
  source.SkipSnapshot(gufo::server::SnapshotEventReason::kCaptureFailure, 8, 2);
  Expect(!peer.TryReserveSnapshot(24, 1),
         "capture failure keeps fallback protected for the lease lifetime");
  Expect(source.HasSnapshotFor(std::vector<gufo::server::ContinuationToken>{1}),
         "valid fallback survives failed peer admission");
  source.Invalidate();
  Expect(peer.TryReserveSnapshot(24, 1),
         "lease release removes source pin so unrelated work can progress");
  peer.SkipSnapshot(gufo::server::SnapshotEventReason::kCaptureFailure, 24, 1);
}

void TestRejectedExternallyHeldCandidateRemainsCharged() {
  SnapshotCacheFixture fixture(2, 24, 1);
  fixture.Save({1}, 1);
  auto source =
      fixture.cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2});
  Expect(source.TryReserveSnapshot(8, 2, true), "candidate reserves bytes");
  auto snapshot = std::make_shared<FakeSnapshot>(2, 8);
  Expect(source.PublishSnapshot({1, 2}, snapshot) == 0,
         "protected sole entry rejects candidate without replacing source");
  Expect(fixture.cache.retained_snapshot_bytes() == 16,
         "rejected externally held payload is still physically charged");
  auto peer =
      fixture.cache.Acquire(std::vector<gufo::server::ContinuationToken>{9});
  Expect(!peer.TryReserveSnapshot(16, 1),
         "external rejected payload cannot become unaccounted admission room");
  snapshot.reset();
  Expect(fixture.cache.retained_snapshot_bytes() == 8 &&
             peer.TryReserveSnapshot(16, 1),
         "last external release frees the rejected candidate exactly once");
  peer.SkipSnapshot(gufo::server::SnapshotEventReason::kCaptureFailure, 16, 1);
}

void TestExternalFinalDestructorCannotSpendItsOwnBytes() {
  std::vector<std::size_t> invalidations(2);
  std::size_t next_id = 0;
  std::atomic<std::size_t> live{0};
  gufo::server::ContinuationCache::Lease observer;
  bool checked = false;
  gufo::server::ContinuationCache cache(
      2, [&] { return std::make_unique<FakeState>(next_id++, &invalidations); },
      {.restore = [](auto&, const auto&) {},
       .capacity_bytes = [] { return 16; },
       .on_event = {},
       .entry_capacity = 1});
  auto external = std::make_shared<TrackedSnapshot>(live, 8, [&] {
    checked = true;
    Expect(
        live == 16 && cache.retained_snapshot_bytes() == 16,
        "external final destructor retains its physical charge until return");
    Expect(!observer.TryReserveSnapshot(8, 1, true, false),
           "external destructor cannot admit bytes occupied by itself");
  });
  auto initial = cache.Acquire(std::vector<gufo::server::ContinuationToken>{1});
  Expect(initial.TryReserveSnapshot(8, 1), "external root reserves bytes");
  initial.Commit({1}, external);
  observer = cache.Acquire(std::vector<gufo::server::ContinuationToken>{9});
  auto next = cache.Acquire(std::vector<gufo::server::ContinuationToken>{2});
  Expect(next.TryReserveSnapshot(8, 1), "replacement reserves remaining bytes");
  next.Commit({2}, std::make_shared<TrackedSnapshot>(live, 8));
  external.reset();
  Expect(cache.retained_snapshot_bytes() == 8 && checked && live == 8,
         "controlled sole-owner reclamation destroys then releases physical "
         "bytes");
  observer.Invalidate();
}

void TestControlBlockIdentityRejectsRawPointerAliasing() {
  std::vector<std::size_t> invalidations(1);
  gufo::server::ContinuationCache cache(
      1, [&] { return std::make_unique<FakeState>(0, &invalidations); },
      {.restore = [](auto&, const auto&) {},
       .capacity_bytes = [] { return 32; },
       .on_event = {},
       .entry_capacity = 4});
  int same_address = 0;
  auto a = std::make_shared<const std::array<std::uint8_t, 8>>();
  auto b = std::make_shared<const std::array<std::uint8_t, 8>>();
  const std::shared_ptr<const void> alias_a(a, &same_address);
  const std::shared_ptr<const void> alias_b(b, &same_address);
  const auto save = [&](unsigned token, std::shared_ptr<const void> owner) {
    auto lease =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{token});
    Expect(lease.TryReserveSnapshot(8, 1),
           "control-block owner reserves bytes");
    lease.Commit(
        {token},
        std::make_shared<SharedStorageSnapshot>(
            std::vector<gufo::server::ContinuationSnapshotStorageOwner>{
                {owner, 8}}));
  };
  save(1, alias_a);
  save(2, alias_b);
  Expect(
      cache.retained_snapshot_bytes() == 16,
      "equal raw addresses with different control blocks remain independent");
  const std::shared_ptr<const void> alias_same_owner(a, &(*a)[1]);
  save(3, alias_same_owner);
  Expect(cache.retained_snapshot_bytes() == 16,
         "different alias pointers with one control block are charged once");
}

void TestZeroSnapshotEntryCapacityIsRejectedBeforeStateAllocation() {
  std::size_t states_created = 0;
  std::vector<std::size_t> invalidations(1);
  bool rejected = false;
  try {
    gufo::server::ContinuationCache cache(
        1,
        [&] {
          return std::make_unique<FakeState>(states_created++, &invalidations);
        },
        {
            .restore = [](auto&, const auto&) {},
            .capacity_bytes = [] { return 64; },
            .on_event = {},
            .entry_capacity = 0,
        });
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Expect(rejected && states_created == 0,
         "invalid snapshot entry capacity cannot allocate model state");
}

}  // namespace

void TestImageIdentityIsolation() {
  for (const bool snapshot_mode : {false, true}) {
    std::vector<std::size_t> invalidations(2);
    std::size_t next_id = 0;
    gufo::server::ContinuationCache::SnapshotSupport support;
    if (snapshot_mode) {
      support.capacity_bytes = [] { return std::size_t{1024}; };
      support.restore = [](auto& state, const auto& snapshot) {
        dynamic_cast<FakeState&>(state).value =
            dynamic_cast<const FakeSnapshot&>(snapshot).value;
      };
    }
    gufo::server::ContinuationCache cache(
        2,
        [&] { return std::make_unique<FakeState>(next_id++, &invalidations); },
        std::move(support));
    const std::vector<gufo::server::ContinuationToken> prompt{1, 248056, 3};
    const std::vector<std::uint8_t> a{1, 2}, b{1, 3};
    const auto save = [&](auto identity, std::size_t value) {
      auto lease = cache.Acquire(prompt, {}, identity);
      Expect(!lease.cache_hit(), "different image is a cold input");
      dynamic_cast<FakeState&>(lease.state()).value = value;
      if (snapshot_mode) {
        Expect(lease.TryReserveSnapshot(sizeof(std::size_t), prompt.size()),
               "snapshot admitted");
        lease.Commit(prompt, std::make_unique<FakeSnapshot>(value));
      } else {
        lease.Commit(prompt);
      }
    };
    save(a, 10);
    save(b, 20);
    for (const auto& [identity, value] :
         std::vector<std::pair<std::vector<std::uint8_t>, std::size_t>>{
             {a, 10}, {b, 20}}) {
      auto lease = cache.Acquire(prompt, {}, identity);
      Expect(lease.cache_hit(), "same image reuses its own prefix");
      Expect(dynamic_cast<FakeState&>(lease.state()).value == value,
             "image state cannot cross requests");
      if (!snapshot_mode)
        lease.Commit(prompt);
    }
    auto text = cache.Acquire(prompt);
    Expect(!text.cache_hit(),
           "literal image-pad text cannot reuse image state");
  }
}

int main() {
  TestExternalFinalDestructorCannotSpendItsOwnBytes();
  TestControlBlockIdentityRejectsRawPointerAliasing();
  TestPeerCannotEvictProtectedCaptureSource();
  TestRejectedExternallyHeldCandidateRemainsCharged();
    TestPhysicalOwnerDeduplicationAndExternalLifetime();
    TestDeltaReservationAdmitsGrowingSharedHistory();
  TestProtectedPublishSurvivesPromptAdmissionAndSlotPressure();
  TestLearnedPrefixesBoundIdentityAgeAndExactTokens();
  TestImageIdentityIsolation();
  TestColdMissThenExactExtensionHit();
  TestDivergenceInvalidatesOldState();
  TestUncommittedLeaseIsInvalidated();
  TestLongestAvailablePrefixWins();
  TestWaitingAcquireCanBeCancelled();
  TestSnapshotCanBranchIntoTwoIndependentStateSlots();
  TestByteCapacityEvictsBeforeSnapshotAllocation();
  TestConcurrentReservationsCannotOvercommitBudget();
  TestImpossibleReservationPreservesRetainedEntries();
  TestAbandonedReservationIsReleased();
  TestReservationMismatchSkipsRetentionWithoutFailingCommit();
  TestEntryReplacementLogsRemovedSnapshot();
  TestOneStateRetainsThreeInterleavedConversations();
  TestIndependentEntryAndByteBoundsUseLru();
  TestLiveHitTouchesFallbackAndFailedCapturePreservesIt();
  TestExactReplacementDoesNotConsumeAnotherSnapshotSlot();
  TestSnapshotSlotsCannotBecomeMutableStates();
  TestRestoringSnapshotStaysAccountedUntilReaderFinishes();
  TestZeroSnapshotEntryCapacityIsRejectedBeforeStateAllocation();
  TestZeroByteBudgetRetainsOnlyTheLiveFrontier();
  TestLeaseSourceIdentitySurvivesSnapshotSlotReuse();
  TestIndependentSnapshotsKeepIdentityAndLongestPrefixIsolation();
  TestRejectedSnapshotIsDestroyedBeforeReentrantAdmission();
  TestEvictedSnapshotStaysChargedThroughDestruction();
  std::cout << "All continuation cache tests passed\n";
  return 0;
}
