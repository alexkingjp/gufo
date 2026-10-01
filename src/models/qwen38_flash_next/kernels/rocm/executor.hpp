#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXECUTOR_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXECUTOR_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "src/core/session_mode.hpp"
#include "src/models/qwen/hip/ops/token.hpp"
#include "src/models/qwen/vision/device_input.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/blaslt.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/history_arena.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "src/models/qwen38_flash_next/snapshot_storage.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

class Executor;
struct ArgmaxCandidate;
struct SnapshotHeader;

/// Per-sequence state on the device: recurrent SSM state, KV and indexer
/// caches, PLE conv history, plus the host-side n-gram window. A
/// speculative forward additionally keeps per-token snapshots of every
/// recurrent buffer so the batch can be cut back to its accepted prefix.
class Session {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  [[nodiscard]] bool mtp_enabled() const noexcept { return mtp_enabled_; }
  [[nodiscard]] std::uint32_t position() const noexcept { return position_; }
  [[nodiscard]] std::uint32_t max_context() const noexcept {
    return max_context_;
  }
  /// Drops every token; the next Forward starts at position 0.
  void Reset();
  void SetCancellationCheck(std::function<bool()> check);
  [[nodiscard]] bool CheckCancellation(std::string* error) const;
  [[nodiscard]] bool Cancelled() const noexcept { return cancelled_; }
  [[nodiscard]] std::uint64_t MutationEpoch() const noexcept {
    return mutation_epoch_;
  }
  [[nodiscard]] std::size_t AllocatedBytes() const noexcept;
  /// Live bytes of the elastic history families (K/V and pooled-key caches,
  /// trunk and MTP). Zero when history preallocation is fixed at
  /// max_context.
  [[nodiscard]] std::size_t HistoryBytes() const noexcept {
    return history_bytes_;
  }
  /// Positions the history families currently hold. Equal to max_context
  /// outside elastic mode.
  [[nodiscard]] std::uint32_t kv_capacity() const noexcept {
    return kv_capacity_;
  }
  void ConfigureVision(std::shared_ptr<const qwen::vision::Prompt> prompt,
                       std::shared_ptr<qwen::vision::Encoder> encoder,
                       hipStream_t stream);
  void RestoreVisionLayout(const qwen::vision::RopeLayout& layout,
                           hipStream_t stream);
  [[nodiscard]] const qwen::vision::RopeLayout& VisionLayout() const noexcept {
    return vision_input_.layout();
  }

private:
  friend class Executor;
  Session() = default;

  struct LinearState {
    float* conv_state{nullptr};    ///< [kernel-1][channels]
    float* state{nullptr};         ///< [v_heads][d][d]
    RollbackRows conv_snapshots;   ///< [max_spec-1][kernel-1][channels]
    RollbackRows state_snapshots;  ///< [max_spec-1][v_heads][d][d]
  };
  struct AttentionState {
    const qwen::vision::DeviceRope* rope{nullptr};
    __half* k_cache{nullptr};  ///< [max_context][kv_heads*d]
    __half* v_cache{nullptr};  ///< [max_context][kv_heads*d]
    /// Packed q8_0 stores (kv_quant.hpp); non-null in quantized mode,
    /// where k_cache/v_cache are null.
    std::byte* k_store{nullptr};
    std::byte* v_store{nullptr};
    float* index_k{nullptr};   ///< [index_capacity_][indexer_dim] raw ring
    __half* block_k{nullptr};  ///< [max_context/ratio][indexer_dim]
  };
  /// Attention view over the MTP draft history. The single constructor for
  /// draft-side views — the batched catch-up path and MtpForward must not
  /// hand-build these, or a new cache family (as k_store/v_store were)
  /// silently misses one site and faults only under multi-session batches.
  [[nodiscard]] Session::AttentionState MtpAttentionView() const noexcept {
    Session::AttentionState view;
    view.rope = vision_input_.rope();
    view.k_cache = mtp_.k_cache;
    view.v_cache = mtp_.v_cache;
    view.k_store = mtp_.k_store;
    view.v_store = mtp_.v_store;
    view.index_k = mtp_.index_k;
    view.block_k = mtp_.block_k;
    return view;
  }
  /// Per-launch values the kernels read from device memory, so a captured
  /// graph replays at any position.
  struct Control {
    std::uint32_t position;      ///< first position of the trunk batch
    std::uint32_t blocks;        ///< indexer blocks pooled so far
    std::uint32_t mtp_position;  ///< first position of the draft batch
    std::int32_t hidden_row;     ///< kept trunk row the draft reads, or -1
    std::uint32_t mtp_blocks;    ///< complete predictor indexer blocks
  };
  struct MtpState {
    __half* k_cache{nullptr};
    __half* v_cache{nullptr};
    std::byte* k_store{nullptr};
    std::byte* v_store{nullptr};
    float* index_k{nullptr};
    __half* block_k{nullptr};
    std::uint32_t blocks{0};
    float* h{nullptr};  ///< [hc_dim] wide residual handed to the next draft
    float* target_hidden{nullptr};  ///< last max_speculative trunk rows
    std::uint32_t position{0};
  };

  mutable bool cancelled_{false};
  std::uint64_t mutation_epoch_{0};
  const Executor* owner_{nullptr};
  qwen::vision::DeviceInput vision_input_;
  std::uint32_t max_context_{0};
  bool mtp_enabled_{false};
  std::uint32_t index_capacity_{0};  ///< power-of-two raw indexer ring rows
  std::uint32_t position_{0};
  std::vector<LinearState> linear_;
  std::vector<AttentionState> attention_;
  float* ple_history_{nullptr};  ///< [PleConvHistory()][hc_dim]
  RollbackRows ple_snapshots_;   ///< [max_spec-1][PleConvHistory()][hc_dim]
  NgramHistory ngram_;
  std::vector<NgramHistory> ngram_snapshots_;
  std::uint32_t spec_base_{0};    ///< position before the speculative batch
  std::uint32_t spec_tokens_{0};  ///< tokens of the pending speculative batch
  MtpState mtp_;
  Control* control_{nullptr};
  std::uint32_t blocks_{0};  ///< host shadow of Control::blocks
  /// Captured decode graphs by batch shape, and the shapes that ran eagerly
  /// once (the GEMM tier's arena must be grown before capture).
  std::unordered_map<std::uint64_t, hipGraphExec_t> graphs_;
  std::unordered_set<std::uint64_t> warmed_;
  std::vector<void*> allocations_;
  std::size_t allocated_bytes_{0};
  /// History buffers allocated by growth, with their byte sizes. Grown
  /// buffers live in the executor's history slab (or, if the slab could
  /// not be reserved, in driver allocations) and are returned through
  /// ReleaseHistorySpan / hipFree at destruction — never freed mid-flight.
  struct GrownSpan {
    void* pointer;
    std::size_t bytes;
  };
  std::vector<GrownSpan> grown_spans_;
  /// Set when a growth failed part-way (some families repointed, others
  /// not): the session is unusable until Reset and every entry point
  /// refuses it.
  bool growth_corrupt_{false};
  /// Live elastic history bytes; released against the executor budget on
  /// destruction. Zero outside elastic mode.
  std::size_t history_bytes_{0};
  /// Positions the history families hold; equals max_context_ outside
  /// elastic mode.
  std::uint32_t kv_capacity_{0};
  /// Arena-carved history regions backing the position-scaled families.
  /// Releasing the handles (session destruction) returns the spans to the
  /// executor's pool; the family pointers above alias them.
  std::vector<std::shared_ptr<history::HistoryArena::Region>> history_regions_;
  /// True when history_bytes_ is arena-backed: the elastic budget was
  /// never charged, so destruction must not release against it.
  bool history_arena_backed_{false};
  std::size_t rollback_bytes_{0};
  std::function<bool()> is_cancelled_;
  std::vector<float*> rollback_allocations_;
  std::uint32_t rollback_depth_{0};
  void TrimRollback(std::uint32_t depth) noexcept;
};

/// Runs the trunk graph on the GPU for one session at a time. Buffers are
/// sized once for `max_batch` tokens; longer prompts are fed in chunks.
class Executor {
public:
  struct Options {
    std::uint32_t max_batch{1};
    /// Rows of logits (and hidden states) a Forward call may return.
    std::uint32_t max_logit_rows{1};
    /// Longest speculative batch; bounds the recurrent snapshot storage.
    std::uint32_t max_speculative{1};
    /// Elastic session history: 0 keeps the legacy full-context
    /// preallocation. A nonzero byte budget sizes every session's K/V and
    /// pooled-key caches to their content — sessions start at
    /// `history_initial_positions` positions and double on demand — while
    /// the executor-wide live total stays within the budget. Frees the
    /// device memory that fixed-shape sessions waste on unused context and
    /// is what lets more lineages park warm on this host.
    std::size_t history_budget_bytes{0};
    std::uint32_t history_initial_positions{32768};
    /// Shared history arena (P1 stage 2): 0 keeps per-session allocation
    /// (fixed or elastic per history_budget_bytes above). A nonzero byte
    /// budget preallocates one executor-wide region pool at creation and
    /// every session's position-scaled K/V and pooled-key families are
    /// carved from it as refcounted regions — no device allocation ever
    /// runs during serving, and a released session's regions return to the
    /// pool for the next lineage. Sessions are full-capacity; growth
    /// (EnsureHistoryCapacity) fails closed in this mode. Mutually
    /// exclusive with history_budget_bytes.
    std::size_t history_arena_bytes{0};
    /// Quantize session K/V caches to packed q8_0 blocks (kv_quant.hpp):
    /// ~47% less history memory and halved attention-read traffic at
    /// depth, at a small bounded per-element quantization error. Fused
    /// prefill projection routes are bypassed when enabled. Snapshots
    /// taken under this mode are rejected by executors running f16 (and
    /// vice versa) via the payload header.
    bool kv_quant{false};
    /// Optional sink for elastic-history events (growth, budget refusal).
    /// The server layer injects its logger; the executor stays free of
    /// logging dependencies. Called from request threads; must be
    /// thread-safe. Messages are engine diagnostics only.
    std::function<void(std::string_view)> history_logger;
  };

  ~Executor();
  [[nodiscard]] hipStream_t stream() const noexcept { return stream_; }
  /// True when session history comes from the shared preallocated arena
  /// (Options::history_arena_bytes).
  [[nodiscard]] bool history_arena_enabled() const noexcept {
    return history_arena_ != nullptr;
  }
  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;

  [[nodiscard]] static std::unique_ptr<Executor> Create(
      const DeviceModel& model, NgramTable* ngram, Options options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      core::SessionMode mode, std::uint32_t max_context,
      std::string* error_msg = nullptr) const;
  [[nodiscard]] bool EnsureRollback(Session& session, std::uint32_t depth,
                                    std::string* error_msg) const;
  [[nodiscard]] std::size_t SessionBytes(
      core::SessionMode mode, std::uint32_t max_context,
      std::uint32_t rollback_depth) const noexcept;
  [[nodiscard]] std::size_t DeferredScratchBytes() const;

  enum class ForwardMode { kDecode, kVerify, kPrefill };

  /// Appends `tokens` (at most max_batch) to the session and returns the
  /// logits of the last `n_logits` tokens in `logits` (n_logits * vocab
  /// floats, host memory). A null `logits` keeps the rows on the GPU for
  /// verification. The final wide residual of those tokens stays on
  /// the device for MtpForward. kVerify permits Rollback (at most
  /// max_speculative rows). kPrefill uses consistent prompt arithmetic at
  /// every chunk width.
  [[nodiscard]] bool Forward(Session& session,
                             std::span<const std::int32_t> tokens,
                             std::uint32_t n_logits, float* logits,
                             ForwardMode mode, std::string* error_msg) const;

  struct BatchItem {
    Session* session;
    std::span<const std::int32_t> tokens;
    bool speculative;
  };
  /// Packs independent decode chains for shared projections. Every chain
  /// keeps its own recurrent/KV state and the decode arithmetic (<=8 rows).
  /// Logit rows remain in batch order until SelectBatchLogits is called.
  [[nodiscard]] bool ForwardBatch(std::span<const BatchItem> items,
                                  std::string* error_msg) const;
  [[nodiscard]] bool SelectBatchLogits(std::uint32_t offset, std::uint32_t rows,
                                       float* logits,
                                       std::string* error_msg) const;

  /// Keeps the first `keep` (1..n) tokens of the last speculative batch and
  /// discards the rest. If `logits` is supplied, copies the kept frontier
  /// into it using the same synchronization as rollback.
  [[nodiscard]] bool Rollback(Session& session, std::uint32_t keep,
                              std::string* error_msg, float* logits = nullptr,
                              bool wait = true) const;

  /// Runs the draft block over `tokens` (at most max_batch) at the session's
  /// MTP position. The hidden input of token i is the trunk residual of row
  /// `hidden_row + i` of the last Forward batch, or, with hidden_row < 0
  /// (single token), the draft block's own residual from the previous call.
  /// Only requested outputs are computed. Production requests a greedy
  /// token or compact candidate logits. Catch-up skips the output head
  /// when only the draft state is needed.
  struct MtpOutput {
    std::int32_t* token{nullptr};
    MtpCandidateLogits* candidates{nullptr};
    MtpTrace* trace{nullptr};  ///< final-row diagnostic; disables graph capture
  };
  struct MtpHeadItem {
    Session* session;
    MtpOutput output;
  };
  struct MtpBatchItem {
    Session* session;
    std::span<const std::int32_t> tokens;
    std::int32_t hidden_row;
  };
  /// Runs independent short predictor chains with shared projections and
  /// private attention caches, positions and carried hidden states.
  [[nodiscard]] bool MtpForwardBatch(std::span<const MtpBatchItem> items,
                                     std::string* error_msg) const;
  /// Projects each session's carried draft hidden state with shared weights.
  [[nodiscard]] bool MtpHeads(std::span<const MtpHeadItem> items,
                              std::string* error_msg) const;
  [[nodiscard]] bool MtpForward(Session& session,
                                std::span<const std::int32_t> tokens,
                                std::int32_t hidden_row, MtpOutput output,
                                std::string* error_msg,
                                const float* hidden_source = nullptr) const;
  /// Leading kept trunk rows, for independent predictor qualification.
  [[nodiscard]] bool CopyTrunkHidden(const Session& session,
                                     std::span<float> hidden,
                                     std::string* error_msg) const;

  /// Plain greedy verification keeps full logit rows on the GPU.
  [[nodiscard]] bool GreedyMtpPredictions(
      std::span<ArgmaxCandidate> predictions, std::string* error_msg) const;

  /// A session's complete context as one host byte payload: recurrent and
  /// PLE state, KV and indexer caches up to the position, and the draft
  /// block's caches plus the `hidden_rows` most recent kept trunk rows.
  /// The payload restores into any session of this executor whose context
  /// holds the position; the pending speculative batch must be empty.
  struct SnapshotInfo {
    std::uint32_t position{0};
    std::uint32_t hidden_rows{0};
  };
  [[nodiscard]] std::uint64_t SnapshotBytes(const Session& session,
                                            std::uint32_t hidden_rows) const;
  [[nodiscard]] bool SaveSnapshot(const Session& session,
                                  std::uint32_t hidden_rows,
                                  std::span<std::uint8_t> payload,
                                  std::string* error_msg) const;
  /// Parent must come from this session's exact immutable capture lineage.
  /// Only append-only KV/pooled keys share storage; raw rings are local.
  [[nodiscard]] std::shared_ptr<const snapshot::Storage> SaveSharedSnapshot(
      const Session& session, std::uint32_t hidden_rows,
      const snapshot::Storage* parent, std::string* error_msg) const;
  /// Data bytes of the capture layout that `parent` would serve without new
  /// allocations: the same parent-geometry validation and layout walk that
  /// SaveSharedSnapshot applies, without touching device memory. Returns 0
  /// when the parent cannot contribute. Never an upper bound on its own —
  /// subtract from SnapshotAllocationBytes for the incremental reservation.
  [[nodiscard]] std::uint64_t SnapshotInheritedBytes(
      const Session& session, std::uint32_t hidden_rows,
      const snapshot::Storage* parent) const;
  [[nodiscard]] std::uint64_t SnapshotAllocationBytes(
      const Session& session, std::uint32_t hidden_rows) const;
  [[nodiscard]] bool RestoreSnapshot(Session& session,
                                     const snapshot::Storage& payload,
                                     SnapshotInfo* info, std::string* error_msg,
                                     std::uint32_t next_drafts = 0) const;
  /// Reuse at most the rollback rows needed by the restored operation;
  /// restoration never grows scratch. Callers derive this bound from the
  /// restored policy (or the concrete verifier width in a diagnostic).
  [[nodiscard]] bool RestoreSnapshot(Session& session,
                                     std::span<const std::uint8_t> payload,
                                     SnapshotInfo* info, std::string* error_msg,
                                     std::uint32_t next_drafts = 0) const;

  /// Rewinds the draft block's own context.
  void MtpRewind(Session& session, std::uint32_t position) const noexcept {
    session.mtp_.position = position;
    session.mtp_.blocks =
        std::min(session.mtp_.blocks, position / config().compress_ratio);
  }
  [[nodiscard]] std::uint32_t MtpPosition(
      const Session& session) const noexcept {
    return session.mtp_.position;
  }

  [[nodiscard]] const Config& config() const noexcept {
    return model_->config();
  }
  [[nodiscard]] std::uint32_t max_batch() const noexcept {
    return options_.max_batch;
  }
  [[nodiscard]] std::uint32_t max_speculative() const noexcept {
    return options_.max_speculative;
  }
  [[nodiscard]] bool has_mtp() const noexcept { return model_->has_mtp(); }
  [[nodiscard]] bool kv_quant() const noexcept { return options_.kv_quant; }
  /// Bytes of one K (or V) cache row per position: f16 pairs, or packed
  /// q8_0 when the session stores are quantized.
  [[nodiscard]] std::size_t KvRowBytes() const noexcept;
  /// Device bytes the elastic history families need at `positions` (K/V and
  /// pooled-key caches, trunk and — for speculative sessions — MTP). The
  /// indexer ring is bounded by top-k and the batch, not the context, so it
  /// is excluded.
  [[nodiscard]] std::size_t ElasticHistoryBytes(
      core::SessionMode mode, std::uint32_t positions) const noexcept;
  [[nodiscard]] std::uint32_t history_initial_positions() const noexcept {
    return options_.history_initial_positions;
  }

private:
  Executor() = default;

  friend struct Session;

  /// Grows the session's history families to hold at least `needed`
  /// positions (doubling from the current capacity), doubling first and
  /// copying the live prefix device-to-device. Invalidates captured decode
  /// graphs — the kernels take the family pointers as launch arguments —
  /// and fails closed without touching the session when the executor-wide
  /// elastic budget cannot cover the growth. No-op outside elastic mode.
  [[nodiscard]] bool EnsureHistoryCapacity(Session& session,
                                           std::uint32_t needed,
                                           std::string* error_msg) const;
  [[nodiscard]] bool ReserveHistoryBytes(std::size_t delta,
                                         std::string* error_msg) const;
  void ReleaseHistoryBytes(std::size_t bytes) const noexcept;

  [[nodiscard]] bool RestoreSnapshotView(Session& session,
                                         snapshot::View payload,
                                         SnapshotInfo* info,
                                         std::string* error_msg,
                                         std::uint32_t next_drafts) const;
  [[nodiscard]] bool ValidateSnapshotCapture(const Session& session,
                                             std::uint32_t hidden_rows,
                                             std::string* error_msg) const;
  /// Visits every device region in payload order with (device pointer or null
  /// when sizing, payload offset, bytes, name, append-only history key or 0).
  /// Returns the payload size, or 0 once a visit failed.
  template<typename Visit>
  static std::uint64_t WalkSnapshot(const SnapshotHeader& h,
                                    const Session* session, Visit&& visit);

  /// An activation batch quantized once for the decode GEMVs; `data` is
  /// null when the batch is wide enough for the tiled path.
  struct Q8Input {
    const float* x;
    const void* data;
    std::uint32_t n;
    std::uint32_t k;
  };
  bool Quantize(const float* x, std::uint32_t n_tokens, std::uint32_t k,
                Q8Input* q, std::string* error_msg) const;
  bool Dense(const DeviceTensor& w, const Q8Input& q, float* out,
             std::string* error_msg) const;
  bool Dense(const DeviceTensor& w, const float* x, float* out,
             std::uint32_t n_tokens, std::string* error_msg) const;
  /// out = (up . x) * silu(gate . x); s_.shexp_gate is scratch.
  /// With `down` (the projection that consumes the result), a wide batch
  /// leaves only that projection's staged input form (F16 rows in
  /// s_.x_half or the tiled Q8 layout in s_.x_q8t, registered in the input
  /// cache); `out` then holds the gate projection, not the result.
  bool GatedDense(const DeviceTensor& up, const DeviceTensor& gate,
                  const float* x, float* out, std::uint32_t n_tokens,
                  const DeviceTensor* down, std::string* error_msg) const;
  /// Reuse or populate the F16 activation staging buffer.
  void PrepareHalfInput(const float* x, std::uint32_t rows,
                        std::uint32_t cols) const;
  /// Whether a wide dense Q8_0 projection takes the F16 WMMA GEMM.
  bool DenseF16Route(const DeviceTensor& w, std::uint32_t n_tokens) const;
  void RoutedHints(const DeviceTensor& w, std::uint32_t n_tokens) const;
  /// Reads the routing of the current batch back and derives the tile
  /// hints for its expert GEMMs (tiled batches only).
  bool RouteHints(std::uint32_t n_tokens, std::string* error_msg) const;
  /// Routed gate/up projections followed by SwiGLU.
  bool GatedExperts(const DeviceTensor& a, const DeviceTensor& b,
                    const float* x, const std::int32_t* ids, float* out,
                    std::uint32_t n_tokens, std::uint32_t n_used,
                    std::string* error_msg) const;
  bool Experts(const DeviceTensor& w, const float* x, const std::int32_t* ids,
               float* out, std::uint32_t n_rows, std::uint32_t n_used,
               std::uint32_t n_tokens, std::string* error_msg) const;
  /// `normed` says the previous Combine already produced m's grouped norm
  /// of res (F32 in s_.xn, or F16 plus tiled Q8 on the wide route).
  bool HcMix(const DeviceMixer& m, const float* res, bool normed, float* mixed,
             float* inject, std::uint32_t n_tokens,
             std::string* error_msg) const;
  /// Share mixer projections across decode requests without selecting the
  /// prefill arithmetic. Injection rows are compact across the whole batch.
  bool HcMixBatch(const DeviceMixer& m, const float* res, bool normed,
                  float* mixed, float* inject, std::uint32_t rows,
                  std::string* error) const;
  void CombineBatch(float* res, const float* gamma, std::uint32_t rows) const;
  /// Residual update by the block output plus the grouped norm for the next
  /// mixer (`gamma`); wide batches write it as F16 and tiled Q8.
  void Combine(float* res, const float* gamma, std::uint32_t n_tokens) const;
  /// Hashes the batch's n-gram rows and starts reading them from disk, so
  /// the read overlaps the layers before the PLE one.
  bool PleFetch(Session& s, std::span<const std::int32_t> tokens,
                bool speculative, std::string* error_msg) const;
  bool WaitPle(std::string* error_msg) const;
  bool Ple(const DeviceLayer& l, Session& s, std::uint32_t n_tokens, float* res,
           bool speculative, std::string* error_msg,
           bool embeddings_ready = false) const;
  bool LinearAttention(const DeviceLayer& l, Session::LinearState& s,
                       const float* x, float* out, std::uint32_t n_tokens,
                       bool speculative, std::string* error_msg,
                       bool projections_ready = false,
                       bool project_output = true) const;
  /// `pos`/`first_block` are device values; `start_pos` and `pool_grid`
  /// are their host-side counterparts for the eager-only decisions.
  bool Attention(const DeviceLayer& l, Session::AttentionState& s,
                 const float* x, float* out, std::uint32_t n_tokens,
                 const std::uint32_t* pos, const std::uint32_t* first_block,
                 std::uint32_t start_pos, std::uint32_t pool_grid,
                 std::uint32_t max_context, bool sparse, std::string* error_msg,
                 bool last_only = false, bool projections_ready = false,
                 bool project_output = true) const;
  bool Moe(const DeviceLayer& l, const float* x, float* out,
           std::uint32_t n_tokens, std::string* error_msg,
           bool last_only = false) const;
  /// Runs routed experts after the router and shared expert are ready.
  bool MoeExperts(const DeviceLayer& l, const float* x, float* out,
                  std::uint32_t n_tokens, std::string* error_msg) const;
  bool MoeBatch(const DeviceLayer& l, const float* x, float* out,
                std::uint32_t rows, std::string* error) const;
  /// Selects the greedy token or compact candidates from full MTP logits.
  bool MtpHead(const DeviceMixer& head, const float* res, bool token,
               bool candidates, std::string* error_msg) const;
  /// Enqueues one trunk batch (control and token upload through logits).
  bool ForwardBody(Session& session, std::uint32_t n, std::uint32_t n_logits,
                   bool download_logits, bool speculative, bool sparse,
                   std::uint32_t start_pos, std::uint32_t pool_grid,
                   std::uint32_t first_layer, std::uint32_t end_layer,
                   std::string* error_msg) const;
  bool MtpBody(Session& session, std::uint32_t n, std::uint32_t pos, bool token,
               bool candidates, std::string* error_msg, std::uint32_t pool_grid,
               const float* hidden_source, MtpTrace* trace) const;
  /// Runs `body` eagerly, or as the session's captured graph for `key`
  /// when `graph` is set. A prefix may leave its work queued so the host
  /// can wait for disk reads while the GPU computes it.
  bool Run(Session& session, std::uint64_t key, bool graph,
           const std::function<bool()>& body, std::string* error_msg,
           bool synchronize = true) const;

  const DeviceModel* model_{nullptr};
  NgramTable* ngram_{nullptr};
  Options options_;
  hipStream_t stream_{nullptr};
  hipEvent_t counts_ready_{nullptr};
  hipblasHandle_t blas_{nullptr};
  std::unique_ptr<BlasLt> blaslt_;
  // Shared history arena (P1 stage 2). The device wrapper outlives the
  // arena implementation here; regions keep the implementation alive, and
  // sessions already must not outlive the executor (owner_), so region
  // release never races executor destruction.
  std::unique_ptr<history::ArenaDevice> history_arena_device_;
  std::shared_ptr<history::HistoryArena> history_arena_;
  // Elastic history budget accounting. Sessions grow from request threads;
  // the counter guards the executor-wide live total.
  mutable std::mutex history_budget_mutex_;
  mutable std::size_t history_live_bytes_{0};
  [[nodiscard]] bool AllocHistorySpan(std::size_t bytes, void** pointer,
                                      hipStream_t stream,
                                      std::string* error_msg) const;
  void FreeHistorySpan(void* pointer, hipStream_t stream) const noexcept;
  void FreeHistorySpanUnlocked(void* pointer,
                               hipStream_t stream) const noexcept;

  // Scratch, sized for max_batch tokens. Names follow reference.cpp.
  struct Scratch {
    std::int32_t* tokens;
    void* x_half;   ///< activations narrowed to the weight's 16-bit type
    void* x_q8[2];  ///< Q8_1 activations of a decode batch, alternating
    void* x_q8t;    ///< tiled Q8 activations of a wide batch (W8A8 route)
    float* res;
    float* xn;
    __half* xn_half;  ///< xn as F16 on the F16 mixer input route
    void* xn_q8t;     ///< xn as tiled Q8 for the W8A8 mixer down projection
    float* lo;
    float* hc_gate;
    float* mixed;
    float* inject;
    float* block_out;
    // linear attention
    float* qkv;
    float* z;
    float* qkvz;  ///< [t][qkv | z] from the stacked projection
    float* alpha_beta;
    float* conv_scratch;
    float* qn;
    float* kn;
    float* gdn_raw;
    float* gdn_out;
    // attention
    float* qg;  ///< [t][q|gate (; k ; v)]
    float* q;
    float* attn_gate;
    float* k;
    float* v;
    float* iq;
    float* ik;
    std::uint32_t* mask;
    float* scores;
    float* ctx;
    float* attn_partials;  ///< split-key partials of a narrow batch
    // ple
    float* ple_emb;
    float* ple_key;
    float* ple_value;
    float* ple_query;
    float* ple_gated;
    float* ple_norm;
    float* ple_conv;
    float* ple_history_scratch;
    // moe
    float* router;
    std::int32_t* ids;
    std::uint32_t* expert_counts;
    // Routed WMMA route: 16-row padded bucket bounds, scatter cursors, the
    // compact row -> (token, slot) maps and the tiled Q8 gathered rows.
    std::int32_t* routed_bounds;
    std::int32_t* routed_cursors;
    std::int32_t* rows_token;
    std::int32_t* rows_slot;
    std::int32_t* routed_tiles;  ///< (expert | tile << 16) per launched tile
    float* weights;
    float* gate_e;
    float* up_e;
    float* down_e;
    float* shexp_gate;
    float* shexp_up;
    float* shexp_out;
    /// Shared-expert SwiGLU rows narrowed for the F16 down projection, so
    /// s_.x_half keeps the routed experts' token rows.
    __half* shexp_half;
    // head and hidden rows kept for the draft block
    float* logits;
    // mtp
    float* mtp_h;
    float* mtp_embd;
    float* mtp_eproj;
    float* mtp_res;
    ArgmaxCandidate* mtp_argmax;
    std::int32_t* mtp_token;
    std::uint32_t* mtp_ids;
    std::uint32_t* mtp_scratch_ids;
    float* mtp_scores;
  };
  mutable Scratch s_{};
  [[nodiscard]] Scratch RowScratch(const Scratch& base,
                                   std::uint32_t offset) const;
  void UseScratch(const Scratch& scratch) const;
  bool DenseBatch(const DeviceTensor& w, const float* x, float* out,
                  std::uint32_t rows, std::string* error_msg) const;
  bool GatedDenseBatch(const DeviceTensor& up, const DeviceTensor& gate,
                       const float* x, float* out, std::uint32_t rows,
                       std::string* error) const;
  bool QuantizeBatch(const float* x, std::uint32_t rows, std::uint32_t cols,
                     std::string* error) const;
  bool AllocateBatch(std::string* error_msg) const;
  /// Allocated only when concurrent decoding is first requested.
  mutable float* batch_logits_{nullptr};
  /// Pinned per-session frontier-logit staging for batched decode
  /// epilogues (kBatchSessions x vocab), so one stream sync covers every
  /// session's rollback download.
  mutable float* batch_frontier_host_{nullptr};

public:
  /// Pinned staging slot `slot` (0..7) for a batched decode epilogue.
  [[nodiscard]] float* BatchFrontierStaging(std::uint32_t slot,
                                            std::string* error_msg) const;

private:
  mutable Session::Control* batch_controls_{nullptr};
  mutable MtpCandidateLogits* batch_candidates_host_{nullptr};
  // Mapped descriptors, one slice per layer: GPU reads cannot race the host
  // preparing the next layer. ForwardBatch drains before reusing this table.
  mutable GdnBatchItem* batch_gdn_host_{nullptr};
  mutable GdnBatchItem* batch_gdn_{nullptr};
  mutable std::uint32_t batch_rows_{0};
  mutable const float* selected_logits_{nullptr};
  [[nodiscard]] const float* VerificationLogits() const {
    return selected_logits_ != nullptr ? selected_logits_ : s_.logits;
  }
  std::uint32_t mask_words_{0};
  /// Queries per block-selection launch (its score scratch is chunk x
  /// max_blocks floats: 128 MB at the 262k context).
  std::uint32_t select_chunk_{512};
  std::vector<void*> allocations_;
  /// Pinned: the n-gram rows go up with hipMemcpyAsync, and a pageable
  /// source would not be ordered against the kernels behind it.
  float* host_emb_{nullptr};
  mutable std::vector<std::uint32_t> host_rows_;
  mutable bool ple_pending_{false};
  // Pinned host staging the launched (or captured) work reads and writes.
  Session::Control* control_host_{nullptr};
  std::int32_t* tokens_host_{nullptr};
  std::uint32_t* counts_host_{nullptr};
  std::int32_t* tiles_host_{nullptr};         ///< routed tile map staging
  mutable std::uint32_t routed_max_rows_{0};  ///< 0 = no readback yet
  mutable std::uint32_t routed_n_tiles_{0};   ///< down projection tiles
  mutable std::uint32_t routed_64_tiles_{0};  ///< appended 64-token tiles
  mutable std::uint32_t routed_pair_offset_{0};
  mutable std::uint32_t routed_pair_tiles_{0};
  mutable std::uint32_t routed_pair_rows_{64};
  mutable std::uint32_t routed_tile_rows_{48};  ///< token rows per tile
  mutable int routed_tile_cols_{0};
  float* logits_host_{nullptr};
  std::int32_t* mtp_token_host_{nullptr};
  MtpCandidateLogits* mtp_candidates_host_{nullptr};
  /// The model geometry allows the wide mixer route (see Combine).
  bool wide_mixer_{false};
  /// Set by Moe when its epilogue is left for the combine that follows.
  mutable bool moe_pending_{false};
  /// Set by GatedDense when s_.shexp_half holds the SwiGLU rows its F16
  /// down projection reads.
  mutable bool shexp_half_ready_{false};
  // What s_.x_q8t / s_.x_half currently hold (input pointer, rows, cols,
  // and the half type), so a projection over the same rows skips its
  // activation pass. Cleared whenever the source buffer is rewritten.
  mutable const float* q8t_src_{nullptr};
  mutable std::uint32_t q8t_rows_{0};
  mutable std::size_t q8t_cols_{0};
  mutable const float* half_src_{nullptr};
  mutable std::uint32_t half_rows_{0};
  mutable std::size_t half_cols_{0};
  mutable bool half_bf16_{false};
  /// Set by a combine that wrote s_.xn_half / s_.xn_q8t instead of s_.xn.
  mutable bool xn_half_{false};
  /// Partial sums per inject logit the last HcMix left in s_.inject.
  mutable std::uint32_t inject_parts_{1};
  mutable unsigned q8_slot_{0};
};

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXECUTOR_HPP_
