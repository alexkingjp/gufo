#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_SNAPSHOT_STORAGE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_SNAPSHOT_STORAGE_HPP_

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace gufo::models::qwen38_flash_next {

/// Stable physical allocation identity is the shared_ptr control block, not
/// get(): compare with owner_less. Bytes include requested backing storage and
/// known object/container storage, not malloc bookkeeping or page rounding.
/// Holding owner pins the allocation; every alias reports the same full charge.
struct SnapshotStorageOwner {
  std::shared_ptr<const void> owner;
  std::size_t bytes;
};

namespace snapshot {

template<class Token>
inline bool ExactPrefix(std::span<const Token> previous,
                        std::span<const Token> current,
                        std::span<const std::uint8_t> previous_identity,
                        std::span<const std::uint8_t> current_identity) {
  return previous.size() <= current.size() &&
         std::equal(previous.begin(), previous.end(), current.begin()) &&
         std::equal(previous_identity.begin(), previous_identity.end(),
                    current_identity.begin(), current_identity.end());
}

class Buffer final {
public:
  template<class Read>
  static std::shared_ptr<const Buffer> Capture(std::size_t size, Read&& read) {
    auto buffer = std::shared_ptr<Buffer>(new Buffer(size));
    if (!read(std::span(buffer->data_.get(), size)))
      return nullptr;
    return buffer;
  }

  [[nodiscard]] std::span<const std::uint8_t> bytes() const noexcept {
    return {data_.get(), size_};
  }

  static SnapshotStorageOwner Owner(const std::shared_ptr<const Buffer>& p) {
    return {std::shared_ptr<const void>(p, p->data_.get()),
            p->size_ + sizeof(Buffer)};
  }

private:
  explicit Buffer(std::size_t size)
      : data_(new std::uint8_t[size]), size_(size) {
    // Preserve the snapshot UMA policy: first touch must not trigger huge-page
    // compaction alongside resident weights. Advise only wholly owned pages.
    const long page = sysconf(_SC_PAGESIZE);
    if (page > 0) {
      const auto address = reinterpret_cast<std::uintptr_t>(data_.get());
      const auto skip = (page - address % page) % page;
      if (size > skip) {
        const auto length = (size - skip) / page * page;
        if (length != 0)
          (void)madvise(data_.get() + skip, length, MADV_NOHUGEPAGE);
      }
    }
  }

  std::unique_ptr<std::uint8_t[]> data_;
  std::size_t size_;
};

/// Serialization-order regions. A nonzero key denotes an append-only history
/// section; zero denotes boundary-local state, which is never inherited.
struct Region {
  std::uint64_t offset;
  std::size_t bytes;
  std::uint32_t history_key{0};
};

/// A flat immutable graph of leaf allocations: children do not own parent
/// snapshots or their recurrent capsules. No mutable device aliases escape.
class Storage final {
public:
  // Bound metadata across long capture chains. Rare consolidation recopies one
  // history section without retaining an unbounded ancestor graph.
  static constexpr std::size_t kMaxSegments = 64;

  template<class Read>
  static std::shared_ptr<const Storage> Capture(std::span<const Region> layout,
                                                const Storage* parent,
                                                Read&& read) {
    auto out = std::shared_ptr<Storage>(new Storage);
    std::size_t private_bytes = 0;
    std::unordered_set<std::uint32_t> keys;
    for (const auto& region : layout) {
      if (region.offset != out->size_ || region.bytes == 0 ||
          region.bytes > std::numeric_limits<std::size_t>::max() - out->size_ ||
          (region.history_key != 0 && !keys.insert(region.history_key).second))
        throw std::invalid_argument("invalid snapshot region layout");
      out->size_ += region.bytes;
      if (region.history_key == 0)
        private_bytes += region.bytes;
    }
    std::shared_ptr<const Buffer> capsule;
    if (private_bytes != 0) {
      capsule = Buffer::Capture(private_bytes, [&](auto destination) {
        std::size_t offset = 0;
        for (const auto& region : layout) {
          if (region.history_key == 0) {
            if (!read(region.offset, destination.subspan(offset, region.bytes)))
              return false;
            offset += region.bytes;
          }
        }
        return true;
      });
      if (!capsule)
        return nullptr;
    }
    std::size_t private_offset = 0;
    out->regions_.reserve(layout.size());
    for (const auto& region : layout) {
      const auto first = out->pieces_.size();
      if (region.history_key == 0) {
        out->pieces_.push_back(
            {region.offset, region.bytes, private_offset, capsule});
        private_offset += region.bytes;
      } else {
        std::size_t inherited = 0;
        if (parent != nullptr) {
          for (const auto& old : parent->regions_) {
            if (old.region.history_key != region.history_key ||
                old.region.bytes > region.bytes ||
                (old.region.bytes < region.bytes && old.count >= kMaxSegments))
              continue;
            for (std::size_t i = old.first; i < old.first + old.count; ++i) {
              auto piece = parent->pieces_[i];
              piece.offset = region.offset + (piece.offset - old.region.offset);
              out->pieces_.push_back(std::move(piece));
            }
            inherited = old.region.bytes;
            break;
          }
        }
        if (inherited != region.bytes) {
          const auto offset = region.offset + inherited;
          const auto bytes = region.bytes - inherited;
          auto tail = Buffer::Capture(bytes, [&](auto destination) {
            return read(offset, destination);
          });
          if (!tail)
            return nullptr;
          out->pieces_.push_back({offset, bytes, 0, std::move(tail)});
        }
      }
      out->regions_.push_back({region, first, out->pieces_.size() - first});
    }
    std::unordered_set<const Buffer*> seen;
    out->owners_.reserve(out->pieces_.size() + 1);
    for (const auto& piece : out->pieces_) {
      if (seen.insert(piece.buffer.get()).second)
        out->owners_.push_back(Buffer::Owner(piece.buffer));
    }
    // An independent accounting identity avoids a Storage -> owner -> Storage
    // cycle. It never keeps uncharged history allocations alive on its own.
    out->metadata_ = std::make_shared<const std::uint8_t>(0);
    out->owners_.push_back(
        {out->metadata_,
         sizeof(Storage) + 1 + out->regions_.capacity() * sizeof(Section) +
             out->pieces_.capacity() * sizeof(Piece) +
             out->owners_.capacity() * sizeof(SnapshotStorageOwner)});
    return out;
  }

  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] std::span<const SnapshotStorageOwner> StorageOwners()
      const noexcept {
    return owners_;
  }

  /// Data bytes of every layout region this storage could serve as a capture
  /// parent, mirroring Capture's inheritance matching exactly (same
  /// history_key, parent bytes within the region, segment cap). Lets callers
  /// reserve the incremental cost of a capture before performing it.
  [[nodiscard]] std::size_t InheritedBytes(
      std::span<const Region> layout) const noexcept {
    std::size_t inherited = 0;
    for (const auto& region : layout) {
      if (region.history_key == 0)
        continue;
      for (const auto& old : regions_) {
        if (old.region.history_key != region.history_key ||
            old.region.bytes > region.bytes ||
            (old.region.bytes < region.bytes && old.count >= kMaxSegments))
          continue;
        inherited += old.region.bytes;
        break;
      }
    }
    return inherited;
  }

  /// Safe full-payload reservation even if a weak capture parent expires.
  /// Additional inherited descriptors are small but must also be reserved.
  static std::uint64_t AllocationUpperBound(std::uint64_t bytes,
                                            std::size_t regions) {
    const auto pieces = regions * kMaxSegments;
    return bytes + sizeof(Storage) + 1 + regions * sizeof(Section) +
           2 * pieces * sizeof(Piece) +
           (pieces + 1) * (sizeof(Buffer) + sizeof(SnapshotStorageOwner));
  }

  template<class Visit>
  bool VisitRange(std::uint64_t offset, std::size_t bytes,
                  Visit&& visit) const {
    if (offset > size_ || bytes > size_ - offset)
      return false;
    auto it = std::lower_bound(pieces_.begin(), pieces_.end(), offset,
                               [](const Piece& p, std::uint64_t start) {
                                 return p.offset + p.bytes <= start;
                               });
    while (bytes != 0 && it != pieces_.end()) {
      const auto skip = static_cast<std::size_t>(offset - it->offset);
      const auto take = std::min(bytes, it->bytes - skip);
      if (!visit(it->buffer->bytes().subspan(it->buffer_offset + skip, take)))
        return false;
      bytes -= take;
      offset += take;
      ++it;
    }
    return bytes == 0;
  }

  [[nodiscard]] bool CopyTo(std::uint64_t offset,
                            std::span<std::uint8_t> destination) const {
    return VisitRange(offset, destination.size(), [&](auto source) {
      std::memcpy(destination.data(), source.data(), source.size());
      destination = destination.subspan(source.size());
      return true;
    });
  }

  void StreamTo(
      const std::function<void(std::span<const std::uint8_t>)>& sink) const {
    for (const auto& piece : pieces_)
      sink(piece.buffer->bytes().subspan(piece.buffer_offset, piece.bytes));
  }

private:
  struct Piece {
    std::uint64_t offset;
    std::size_t bytes;
    std::size_t buffer_offset;
    std::shared_ptr<const Buffer> buffer;
  };
  struct Section {
    Region region;
    std::size_t first;
    std::size_t count;
  };

  Storage() = default;
  // Destroy last so metadata remains charged while descriptor vectors unwind.
  std::shared_ptr<const void> metadata_;
  std::size_t size_{0};
  std::vector<Section> regions_;
  std::vector<Piece> pieces_;
  std::vector<SnapshotStorageOwner> owners_;
};

/// Reads persistent contiguous bytes and shared RAM segments through the same
/// restore validation path, without materializing a second full payload.
class View {
public:
  explicit View(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}
  explicit View(const Storage& storage) : storage_(&storage) {}
  [[nodiscard]] std::size_t size() const noexcept {
    return storage_ ? storage_->size() : bytes_.size();
  }
  template<class Visit>
  bool VisitRange(std::uint64_t offset, std::size_t bytes,
                  Visit&& visit) const {
    if (storage_)
      return storage_->VisitRange(offset, bytes, std::forward<Visit>(visit));
    return offset <= bytes_.size() && bytes <= bytes_.size() - offset &&
           (bytes == 0 || visit(bytes_.subspan(offset, bytes)));
  }
  [[nodiscard]] bool CopyTo(std::uint64_t offset,
                            std::span<std::uint8_t> destination) const {
    return VisitRange(offset, destination.size(), [&](auto source) {
      std::memcpy(destination.data(), source.data(), source.size());
      destination = destination.subspan(source.size());
      return true;
    });
  }

private:
  std::span<const std::uint8_t> bytes_;
  const Storage* storage_{nullptr};
};

}  // namespace snapshot
}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_SNAPSHOT_STORAGE_HPP_
