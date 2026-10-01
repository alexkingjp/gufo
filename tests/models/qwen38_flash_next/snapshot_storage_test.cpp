#include "src/models/qwen38_flash_next/snapshot_storage.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

namespace qfn = gufo::models::qwen38_flash_next;
namespace ss = qfn::snapshot;

namespace {

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

// Header/recurrent, target K, raw ring, target V, pooled keys, MTP/capsule.
std::vector<ss::Region> Layout(std::size_t rows, std::size_t ring) {
  std::vector<ss::Region> layout;
  std::size_t offset = 0;
  const auto add = [&](std::size_t bytes, std::uint32_t key = 0) {
    if (bytes != 0) {
      layout.push_back({offset, bytes, key});
      offset += bytes;
    }
  };
  add(32);
  add(rows * 16, 1);
  add(ring * 8);
  add(rows * 16, 2);
  add(rows / 4 * 4, 3);
  add(24);
  return layout;
}

std::vector<std::uint8_t> Contents(const std::vector<ss::Region>& layout,
                                   std::uint8_t boundary,
                                   std::size_t common_rows = 0,
                                   std::uint8_t branch = 0) {
  std::vector<std::uint8_t> bytes(layout.back().offset + layout.back().bytes);
  for (const auto& r : layout) {
    for (std::size_t i = 0; i < r.bytes; ++i) {
      const auto common =
          r.history_key == 3 ? common_rows / 4 * 4 : common_rows * 16;
      bytes[r.offset + i] =
          r.history_key ? static_cast<std::uint8_t>(i + r.history_key +
                                                    (i >= common ? branch : 0))
                        : boundary;
    }
  }
  return bytes;
}

std::shared_ptr<const ss::Storage> Capture(
    const std::vector<ss::Region>& layout,
    const std::vector<std::uint8_t>& bytes, const ss::Storage* parent,
    std::size_t* copied = nullptr, std::size_t fail_after = ~std::size_t{0}) {
  std::size_t calls = 0;
  return ss::Storage::Capture(
      layout, parent, [&](auto offset, auto destination) {
        if (calls++ >= fail_after)
          return false;
        if (copied)
          *copied += destination.size();
        std::memcpy(destination.data(), bytes.data() + offset,
                    destination.size());
        return true;
      });
}

void CheckBytes(const ss::Storage& storage,
                const std::vector<std::uint8_t>& expected) {
  std::vector<std::uint8_t> copied(storage.size());
  Require(storage.CopyTo(0, copied) && copied == expected,
          "copy changed bytes");
  std::vector<std::uint8_t> streamed;
  storage.StreamTo([&](auto span) {
    streamed.insert(streamed.end(), span.begin(), span.end());
  });
  Require(streamed == expected, "stream changed serialization order");
  for (const auto view : {ss::View(storage), ss::View(expected)}) {
    std::array<std::uint8_t, 19> section{};
    for (std::size_t offset = 0; offset + section.size() <= expected.size();
         offset += 13) {
      Require(view.CopyTo(offset, section), "partial read failed");
      Require(
          std::equal(section.begin(), section.end(), expected.begin() + offset),
          "partial read crossed a segment incorrectly");
    }
    Require(!view.CopyTo(expected.size(), section),
            "out-of-bounds read accepted");
  }
}

std::size_t UniqueBytes(std::initializer_list<const ss::Storage*> stores) {
  std::map<std::shared_ptr<const void>, std::size_t, std::owner_less<>> owners;
  for (const auto* store : stores) {
    for (const auto& owner : store->StorageOwners()) {
      const auto [it, inserted] = owners.emplace(owner.owner, owner.bytes);
      Require(inserted || it->second == owner.bytes, "unstable owner charge");
    }
  }
  std::size_t total = 0;
  for (const auto& [identity, bytes] : owners) {
    Require(identity != nullptr && bytes != 0, "empty owner");
    total += bytes;
  }
  return total;
}

void TestBranchesAndLifetime() {
  const auto first_layout = Layout(128, 3);
  const auto first_bytes = Contents(first_layout, 17);
  std::size_t copied = 0;
  auto first = Capture(first_layout, first_bytes, nullptr, &copied);
  Require(first && copied == first_bytes.size(), "first capture incomplete");
  CheckBytes(*first, first_bytes);
  const auto next_layout = Layout(132, 1);
  auto next_bytes = Contents(next_layout, 23);
  copied = 0;
  auto next = Capture(next_layout, next_bytes, first.get(), &copied);
  const auto inherited = 128 * 32 + 128 / 4 * 4;
  Require(next && copied == next_bytes.size() - inherited,
          "capture recopied immutable prefix or reused local capsule");
  CheckBytes(*next, next_bytes);
  Require(UniqueBytes({first.get(), next.get()}) ==
              UniqueBytes({first.get()}) + UniqueBytes({next.get()}) -
                  inherited - 3 * sizeof(ss::Buffer),
          "unique accounting did not deduplicate shared physical allocations");
  Require(UniqueBytes({next.get()}) <= ss::Storage::AllocationUpperBound(
                                           next->size(), next_layout.size()),
          "physical accounting exceeds capture reservation");
  const auto branch_layout = Layout(136, 2);
  const auto branch_bytes = Contents(branch_layout, 31, 128, 99);
  auto branch = Capture(branch_layout, branch_bytes, first.get());
  Require(branch != nullptr, "branch capture failed");
  CheckBytes(*branch, branch_bytes);
  CheckBytes(*next, next_bytes);
  const auto unrelated_bytes = Contents(next_layout, 9, 0, 77);
  auto unrelated = Capture(next_layout, unrelated_bytes, nullptr);
  Require(UniqueBytes({next.get(), unrelated.get()}) ==
              UniqueBytes({next.get()}) + UniqueBytes({unrelated.get()}),
          "same position shared unrelated lineage");
  // Failed/cancelled capture leaves the only valid fallback and its owners
  // intact.
  const auto before = UniqueBytes({first.get()});
  for (std::size_t fail = 0; fail < 6; ++fail) {
    auto cancelled =
        Capture(next_layout, next_bytes, first.get(), nullptr, fail);
    Require(!cancelled, "cancelled capture published a snapshot");
    Require(UniqueBytes({first.get()}) == before,
            "cancellation changed fallback");
  }
  std::weak_ptr<const void> capsule = first->StorageOwners().front().owner;
  std::weak_ptr<const ss::Storage> weak_first = first;
  first.reset();
  Require(weak_first.expired() && capsule.expired(),
          "child retained ancestor object or recurrent capsule");
  CheckBytes(*next, next_bytes);
  CheckBytes(*branch, branch_bytes);
  next_bytes.assign(next_bytes.size(), 0);
  Require(next->CopyTo(0, std::span(next_bytes).first(next_bytes.size() - 1)),
          "valid partial copy failed");
  std::vector<qfn::SnapshotStorageOwner> pinned(next->StorageOwners().begin(),
                                                next->StorageOwners().end());
  std::weak_ptr<const void> pinned_owner = pinned.front().owner;
  next.reset();
  Require(!pinned_owner.expired(), "reader pin did not preserve allocation");
  pinned.clear();
  Require(pinned_owner.expired(),
          "allocation did not release after last reader");
}

void TestInheritedBytesEstimate() {
  const auto parent_layout = Layout(128, 3);
  auto parent = Capture(parent_layout, Contents(parent_layout, 17), nullptr);
  Require(parent != nullptr, "estimate parent capture failed");
  // A growing child at 132 rows inherits every parent history byte.
  const auto child_layout = Layout(132, 1);
  Require(parent->InheritedBytes(child_layout) == 128 * 32 + 128 / 4 * 4,
          "growing-child estimate did not mirror capture inheritance");
  // A rolled-back position (parent longer than the region) inherits nothing,
  // mirroring the capture-side geometry rejection.
  Require(parent->InheritedBytes(Layout(96, 1)) == 0,
          "shrinking region must not estimate inherited bytes");
  // Boundary-local regions and the header are never inherited.
  const auto local_only = std::vector<ss::Region>{
      {0, 32}, {32, 64}, {96, 8}};
  Require(parent->InheritedBytes(local_only) == 0,
          "boundary-local regions must not estimate inherited bytes");
  // The estimate must bound the shared charge from below: every shared piece
  // the capture will reuse costs its parent bytes plus one Buffer header,
  // and the reservation covers data only while descriptor headroom absorbs
  // the headers.
  const auto child_bytes = Contents(child_layout, 23);
  auto child = Capture(child_layout, child_bytes, parent.get());
  Require(child != nullptr, "estimate child capture failed");
  const auto shared_charge =
      UniqueBytes({parent.get()}) + UniqueBytes({child.get()}) -
      UniqueBytes({parent.get(), child.get()});
  Require(parent->InheritedBytes(child_layout) <= shared_charge,
          "inherited-data estimate exceeded the shared physical charge");
}

void TestFrozenReader() {
  const auto layout = Layout(32, 3);
  const auto bytes = Contents(layout, 42);
  auto store = Capture(layout, bytes, nullptr);
  Require(store != nullptr, "reader fixture capture failed");
  std::size_t offset = 0;
  bool threw = false;
  try {
    store->StreamTo([](auto) { throw std::runtime_error("cancelled writer"); });
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Require(threw, "stream swallowed writer failure");
  CheckBytes(*store, bytes);
  auto reader = store;
  std::weak_ptr<const ss::Storage> weak = store;
  store.reset();
  Require(!weak.expired(), "restore reader did not retain snapshot");
  reader->StreamTo([&](auto section) {
    Require(std::equal(section.begin(), section.end(), bytes.begin() + offset),
            "eviction disturbed streaming reader");
    offset += section.size();
  });
  Require(offset == bytes.size(), "reader truncated payload");
  reader.reset();
  Require(weak.expired(), "read completion retained snapshot");
  auto fallback = Capture(layout, bytes, weak.lock().get());
  Require(fallback != nullptr, "expired weak parent broke full capture");
  CheckBytes(*fallback, bytes);
}

void TestLineageAndBoundedSegments() {
  const std::array<std::int32_t, 3> parent{1, 2, 3};
  const std::array<std::int32_t, 4> child{1, 2, 3, 4}, edited{1, 9, 3, 4};
  const std::array<std::uint8_t, 2> image{5, 8}, other_image{5, 9};
  Require(ss::ExactPrefix<std::int32_t>(parent, child, image, image),
          "exact extension rejected");
  Require(
      !ss::ExactPrefix<std::int32_t>(parent, edited, image, image) &&
          !ss::ExactPrefix<std::int32_t>(parent, child, image, other_image) &&
          !ss::ExactPrefix<std::int32_t>(child, parent, image, image),
      "token/image/length mismatch accepted");
  std::shared_ptr<const ss::Storage> previous;
  for (std::size_t n = 1; n <= 2 * ss::Storage::kMaxSegments + 5; ++n) {
    const std::vector<ss::Region> layout{{0, 16}, {16, n * 8, 1}};
    auto bytes = Contents(layout, static_cast<std::uint8_t>(n));
    auto next = Capture(layout, bytes, previous.get());
    Require(next != nullptr, "capture chain failed");
    CheckBytes(*next, bytes);
    Require(next->StorageOwners().size() <= ss::Storage::kMaxSegments + 2,
            "unbounded history metadata");
    Require(UniqueBytes({next.get()}) <=
                ss::Storage::AllocationUpperBound(next->size(), layout.size()),
            "capture bound excludes inherited metadata");
    previous = std::move(next);
  }
  const std::vector<ss::Region> invalid{{1, 8, 1}};
  bool rejected = false;
  try {
    (void)ss::Storage::Capture(invalid, nullptr,
                               [](auto, auto) { return true; });
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Require(rejected, "malformed section layout accepted");
}

}  // namespace

int main() {
  try {
    TestBranchesAndLifetime();
    TestFrozenReader();
    TestInheritedBytesEstimate();
    TestLineageAndBoundedSegments();
    std::cout << "snapshot storage CPU tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
