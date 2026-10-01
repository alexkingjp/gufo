#include "src/cli/serve/text_generation_scheduler.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace gufo::server {
namespace {

struct OutputBudget {
  explicit OutputBudget(std::size_t byte_limit) : limit(byte_limit) {}

  [[nodiscard]] bool TryReserve(std::size_t bytes) noexcept {
    std::size_t current = buffered_bytes.load(std::memory_order_relaxed);
    while (current <= limit && bytes <= limit - current) {
      const std::size_t updated = current + bytes;
      if (buffered_bytes.compare_exchange_weak(current, updated,
                                               std::memory_order_acq_rel,
                                               std::memory_order_relaxed)) {
        std::size_t high_water =
            max_buffered_bytes.load(std::memory_order_relaxed);
        while (high_water < updated &&
               !max_buffered_bytes.compare_exchange_weak(
                   high_water, updated, std::memory_order_relaxed,
                   std::memory_order_relaxed)) {
        }
        return true;
      }
    }
    return false;
  }

  void Release(std::size_t bytes) noexcept {
    if (bytes > 0) {
      buffered_bytes.fetch_sub(bytes, std::memory_order_acq_rel);
    }
  }

  const std::size_t limit;
  std::atomic<std::size_t> buffered_bytes{0};
  std::atomic<std::size_t> max_buffered_bytes{0};
};

struct ScheduledRequest {
  std::uint64_t id{0};
  std::string client_id{"anonymous"};
  std::vector<TextRunnerToken> prompt;
  std::shared_ptr<const TextPromptContext> prompt_context;
  std::shared_ptr<const sampling::TokenConstraint> json_constraint;
  std::vector<TextCacheBoundary> cache_boundaries;
  bool cache_prompt{true};
  std::size_t cache_prefix_tokens{0};
  std::size_t token_limit{1};
  sampling::SamplingConfig sampling;
  TextGenerationScheduler::CancellationCheck external_cancellation;
  bool publish_token_pieces{false};
  TextGenerationScheduler::Clock::time_point request_start;
  std::optional<TextGenerationScheduler::Clock::time_point> deadline;
  std::optional<TextGenerationScheduler::Clock::time_point> queue_deadline;
  TextGenerationScheduler::ClockSource clock;
  TextGenerationScheduler::Clock::time_point submitted_at;
  std::optional<TextGenerationScheduler::Clock::time_point> admitted_at;
  std::optional<TextGenerationScheduler::Clock::time_point> decode_ready_at;
  std::optional<TextGenerationScheduler::Clock::time_point> wait_start;
  bool waiting_for_snapshot{false};
  std::size_t admission_bypasses{0};
  std::size_t reserved_prefill_tokens{0};
  bool accounted{false};
  std::size_t max_output_bytes{0};
  std::size_t max_buffered_output_bytes{0};
  std::shared_ptr<OutputBudget> output_budget;

  std::atomic<bool> cancellation_requested{false};
  std::atomic<TextRequestPhase> phase{TextRequestPhase::kQueued};

  TextRunnerPool::Request runner_request;
  TextGenerationScheduler::Result result;
  std::optional<TextGenerationScheduler::Clock::time_point> previous_token;
  std::chrono::duration<double, std::milli> inter_token_total{0};
  std::size_t inter_token_samples{0};
  bool decode_due{false};
  std::optional<TextRunnerToken> preview_token;
  bool advance_pending{false};
  bool preview_stops{false};
  std::size_t generated_output_bytes{0};

  std::mutex output_mutex;
  std::condition_variable output_condition;
  std::deque<std::string> output_pieces;
  std::size_t buffered_output_bytes{0};
  std::exception_ptr failure;
  bool terminal{false};
};

struct PendingClient {
  std::string client_id;
  std::deque<std::shared_ptr<ScheduledRequest>> requests;
};

bool IsTerminal(const std::shared_ptr<ScheduledRequest>& request) noexcept {
  return request->phase.load(std::memory_order_acquire) ==
         TextRequestPhase::kTerminal;
}

void ReleaseBufferedOutputLocked(
    const std::shared_ptr<ScheduledRequest>& request) noexcept {
  request->output_pieces.clear();
  request->output_budget->Release(request->buffered_output_bytes);
  request->buffered_output_bytes = 0;
}

void PublishTerminal(const std::shared_ptr<ScheduledRequest>& request,
                     std::exception_ptr failure = {},
                     bool discard_pending_output = false) noexcept {
  {
    const std::lock_guard<std::mutex> lock(request->output_mutex);
    if (discard_pending_output) {
      ReleaseBufferedOutputLocked(request);
    }
    request->failure = std::move(failure);
    request->terminal = true;
    request->phase.store(TextRequestPhase::kTerminal,
                         std::memory_order_release);
  }
  request->output_condition.notify_all();
}

[[nodiscard]] bool PublishPiece(
    const std::shared_ptr<ScheduledRequest>& request, std::string piece) {
  if (!request->publish_token_pieces) {
    return true;
  }
  const std::size_t piece_bytes = piece.size();
  {
    const std::lock_guard<std::mutex> lock(request->output_mutex);
    if (request->buffered_output_bytes > request->max_buffered_output_bytes ||
        piece_bytes > request->max_buffered_output_bytes -
                          request->buffered_output_bytes) {
      return false;
    }
    if (!request->output_budget->TryReserve(piece_bytes)) {
      return false;
    }
    try {
      request->buffered_output_bytes += piece_bytes;
      request->result.max_buffered_output_bytes =
          std::max(request->result.max_buffered_output_bytes,
                   request->buffered_output_bytes);
      request->output_pieces.push_back(std::move(piece));
    } catch (...) {
      request->output_budget->Release(piece_bytes);
      request->buffered_output_bytes -= piece_bytes;
      throw;
    }
  }
  request->output_condition.notify_one();
  return true;
}

bool CancellationRequested(const std::shared_ptr<ScheduledRequest>& request) {
  if (request->cancellation_requested.load(std::memory_order_acquire)) {
    return true;
  }
  return request->external_cancellation && request->external_cancellation();
}

bool DeadlineExceeded(const std::shared_ptr<ScheduledRequest>& request) {
  return request->deadline.has_value() &&
         request->clock() >= *request->deadline;
}

}  // namespace

struct TextGenerationScheduler::Request::Impl {
  explicit Impl(std::shared_ptr<ScheduledRequest> scheduled_request)
      : request(std::move(scheduled_request)) {}

  std::shared_ptr<ScheduledRequest> request;
  bool waited{false};
};

struct TextGenerationScheduler::Impl {
  Impl(std::shared_ptr<TextRunnerPool> model_runner_pool,
       TextPrefillPolicy model_prefill_policy,
       TextSchedulerPolicy model_scheduler_policy, ClockSource clock_source)
      : runner_pool(std::move(model_runner_pool)),
        prefill_policy(model_prefill_policy),
        scheduler_policy(model_scheduler_policy),
        clock(std::move(clock_source)),
        output_budget(std::make_shared<OutputBudget>(
            model_scheduler_policy.max_buffered_output_bytes_total)) {
    if (runner_pool == nullptr) {
      throw std::invalid_argument(
          "text generation scheduler runner pool must not be null");
    }
    if (!clock || prefill_policy.decode_active_tokens == 0 ||
        prefill_policy.target_chunk_time.count() <= 0 ||
        prefill_policy.min_chunk_tokens == 0 ||
        prefill_policy.min_chunk_tokens >
            std::min(prefill_policy.decode_active_tokens,
                     kDefaultDecodeActivePrefillTokens)) {
      throw std::invalid_argument(
          "active-decode prefill budget must be at least one token");
    }
    if (scheduler_policy.max_pending_requests == 0 ||
        scheduler_policy.max_pending_requests_per_client == 0 ||
        scheduler_policy.max_pending_requests_per_client >
            scheduler_policy.max_pending_requests ||
        scheduler_policy.max_output_bytes_per_request == 0 ||
        scheduler_policy.max_buffered_output_bytes_per_request == 0 ||
        scheduler_policy.max_buffered_output_bytes_total == 0 ||
        scheduler_policy.request_timeout.count() < 0 ||
        scheduler_policy.decode_burst.count() < 0 ||
        scheduler_policy.queue_timeout.count() < 0 ||
        scheduler_policy.admission_aging.count() < 0 ||
        scheduler_policy.max_decode_steps_per_burst == 0) {
      throw std::invalid_argument("invalid text scheduler limits");
    }
    incremental_prefill_supported =
        runner_pool->runner().Descriptor().capabilities.incremental_prefill;
    final_token_advance_required =
        runner_pool->runner()
            .Descriptor()
            .capabilities.final_token_advance_required;
    incremental_text_is_exact = runner_pool->runner()
                                    .Descriptor()
                                    .capabilities.incremental_text_is_exact;
    multi_token_decode =
        runner_pool->runner().Descriptor().capabilities.multi_token_decode;
    batched_multi_token_decode = runner_pool->runner()
                                     .Descriptor()
                                     .capabilities.batched_multi_token_decode;
    batched_multi_token_decode_max_width =
        runner_pool->runner()
            .Descriptor()
            .capabilities.batched_multi_token_decode_max_width;
    worker = std::jthread(
        [this](const std::stop_token& stop_token) { Run(stop_token); });
  }

  ~Impl() {
    {
      const std::lock_guard<std::mutex> lock(queue_mutex);
      stopping = true;
    }
    worker.request_stop();
    queue_condition.notify_all();
    if (worker.joinable()) {
      worker.join();
    }
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;

  [[nodiscard]] std::shared_ptr<ScheduledRequest> PopQueued() {
    const std::lock_guard<std::mutex> lock(queue_mutex);
    if (queued_clients.empty()) {
      return {};
    }
    auto selected = queued_clients.begin();
    // Lineage-sticky admission: a queued request whose input identity
    // matches a free slot's resident continuation adopts that slot without
    // any restore, and the conversation keeps serving its live state in
    // place. Bounded: after capacity() consecutive sticky admissions one
    // unrelated head goes through, so nothing starves.
    bool sticky_pop = false;
    if (scheduler_policy.sticky_admission &&
        sticky_admission_run < runner_pool->capacity()) {
      const auto lineages = runner_pool->FreeSlotLineages();
      if (!lineages.empty()) {
        auto match = queued_clients.end();
        std::size_t match_live = 0;
        for (auto candidate = queued_clients.begin();
             candidate != queued_clients.end(); ++candidate) {
          const auto& head = candidate->requests.front();
          if (head->prompt_context == nullptr)
            continue;
          const auto& identity = head->prompt_context->cache_identity;
          if (identity.empty())
            continue;
          for (const auto& lineage : lineages) {
            if (lineage.live_tokens > match_live &&
                lineage.identity.size() == identity.size() &&
                std::equal(identity.begin(), identity.end(),
                           lineage.identity.begin())) {
              match = candidate;
              match_live = lineage.live_tokens;
              break;
            }
          }
        }
        if (match != queued_clients.end()) {
          selected = match;
          sticky_pop = true;
        }
      }
    }
    if (sticky_pop) {
      ++sticky_admission_run;
    } else {
      sticky_admission_run = 0;
    }
    if (scheduler_policy.prefer_short_prefill) {
      const auto now = clock();
      auto forced = queued_clients.end();
      for (auto candidate = queued_clients.begin();
           candidate != queued_clients.end(); ++candidate) {
        const auto& request = candidate->requests.front();
        const bool aged =
            scheduler_policy.admission_aging.count() > 0 &&
            now - request->submitted_at >= scheduler_policy.admission_aging;
        if (aged || request->admission_bypasses >=
                        scheduler_policy.max_admission_bypasses) {
          if (forced == queued_clients.end() ||
              request->id < forced->requests.front()->id)
            forced = candidate;
        }
        // Only compare client heads: short jobs cannot jump their own FIFO.
        const auto cost = request->result.estimated_prefill_tokens.value_or(
            request->result.prompt_tokens);
        const auto best = selected->requests.front();
        if (cost < best->result.estimated_prefill_tokens.value_or(
                       best->result.prompt_tokens))
          selected = candidate;
      }
      if (forced != queued_clients.end())
        selected = forced;
      for (auto& client : queued_clients) {
        auto& head = client.requests.front();
        if (head != selected->requests.front() &&
            head->admission_bypasses < scheduler_policy.max_admission_bypasses)
          ++head->admission_bypasses;
      }
    }
    auto client = std::move(*selected);
    queued_clients.erase(selected);
    auto request = std::move(client.requests.front());
    client.requests.pop_front();
    --queued_count;
    if (!client.requests.empty()) {
      queued_clients.push_back(std::move(client));
    }
    return request;
  }

  [[nodiscard]] bool RemoveQueued(
      const std::shared_ptr<ScheduledRequest>& request) {
    const std::lock_guard<std::mutex> lock(queue_mutex);
    for (auto client = queued_clients.begin(); client != queued_clients.end();
         ++client) {
      const auto queued =
          std::find(client->requests.begin(), client->requests.end(), request);
      if (queued == client->requests.end()) {
        continue;
      }
      client->requests.erase(queued);
      --queued_count;
      if (client->requests.empty()) {
        queued_clients.erase(client);
      }
      return true;
    }
    return false;
  }

  [[nodiscard]] std::vector<std::shared_ptr<ScheduledRequest>> QueuedSnapshot()
      const {
    const std::lock_guard<std::mutex> lock(queue_mutex);
    std::vector<std::shared_ptr<ScheduledRequest>> snapshot;
    snapshot.reserve(queued_count);
    for (const auto& client : queued_clients) {
      snapshot.insert(snapshot.end(), client.requests.begin(),
                      client.requests.end());
    }
    return snapshot;
  }

  static double Milliseconds(Clock::duration duration) {
    return std::max(
        0.0, std::chrono::duration<double, std::milli>(duration).count());
  }

  void AccountWait(const std::shared_ptr<ScheduledRequest>& request) {
    if (!request->wait_start)
      return;
    const double elapsed = Milliseconds(clock() - *request->wait_start);
    if (request->waiting_for_snapshot)
      request->result.snapshot_wait_ms += elapsed;
    else
      request->result.resident_wait_ms += elapsed;
    request->wait_start.reset();
  }

  void BeginWork(const std::shared_ptr<ScheduledRequest>& request) {
    AccountWait(request);
    request->waiting_for_snapshot = false;
  }

  void EndWork(const std::shared_ptr<ScheduledRequest>& request) {
    if (IsTerminal(request))
      return;
    request->wait_start = clock();
    request->waiting_for_snapshot = request->runner_request.SnapshotPending();
    if (request->runner_request.prefill_complete() && !request->decode_ready_at)
      request->decode_ready_at = clock();
  }

  void ReservePrefill(const std::shared_ptr<ScheduledRequest>& request,
                      std::size_t tokens) {
    const std::lock_guard<std::mutex> lock(queue_mutex);
    reserved_prefill_tokens -= request->reserved_prefill_tokens;
    request->reserved_prefill_tokens = tokens;
    reserved_prefill_tokens += tokens;
  }

  void FinalizeTiming(const std::shared_ptr<ScheduledRequest>& request) {
    AccountWait(request);
    const auto now = clock();
    request->result.queue_ms = request->result.queue_admission_ms =
        Milliseconds(request->admitted_at.value_or(now) -
                     request->request_start);
    request->result.total_generation_wall_ms =
        Milliseconds(now - request->request_start);
    if (request->decode_ready_at)
      request->result.decode_wall_ms =
          Milliseconds(now - *request->decode_ready_at);
    const std::lock_guard<std::mutex> lock(queue_mutex);
    if (request->accounted) {
      --inflight_count;
      auto client = inflight_clients.find(request->client_id);
      if (--client->second == 0)
        inflight_clients.erase(client);
      if (request->admitted_at)
        --resident_count;
      reserved_prefill_tokens -= request->reserved_prefill_tokens;
      request->reserved_prefill_tokens = 0;
      request->accounted = false;
    }
  }

  static void CopyCheckpointMetrics(
      const std::shared_ptr<ScheduledRequest>& request,
      const TextRunnerPool::Request::CommitMetrics& metrics) {
    if (metrics.prefix_checkpoint_tokens != 0) {
      switch (metrics.prefix_checkpoint_kind) {
        case TextCacheBoundaryKind::kSystemEnd:
          request->result.prefix_checkpoint_kind = "system_end";
          break;
        case TextCacheBoundaryKind::kLastUserStart:
          request->result.prefix_checkpoint_kind = "last_user_start";
          break;
        case TextCacheBoundaryKind::kHistoryEnd:
          request->result.prefix_checkpoint_kind = "history_end";
          break;
        case TextCacheBoundaryKind::kFullPrompt:
          request->result.prefix_checkpoint_kind = "full_prompt";
          break;
        case TextCacheBoundaryKind::kLearnedPrefix:
          request->result.prefix_checkpoint_kind = "learned_prefix";
          break;
      }
    }
    request->result.prefix_checkpoint_tokens = metrics.prefix_checkpoint_tokens;
    request->result.prefix_checkpoint_bytes = metrics.prefix_checkpoint_bytes;
    request->result.prefix_checkpoint_ms = metrics.prefix_checkpoint_ms;
    request->result.prefix_checkpoint_failures =
        metrics.prefix_checkpoint_failures;
  }

  void FinalizeResult(const std::shared_ptr<ScheduledRequest>& request,
                      TextGenerationBackend::FinishReason finish_reason) {
    request->result.completion_tokens = request->result.tokens.size();
    request->result.finish_reason = finish_reason;
    if (request->inter_token_samples > 0) {
      request->result.mean_inter_token_ms =
          request->inter_token_total.count() /
          static_cast<double>(request->inter_token_samples);
    }
    if (!incremental_text_is_exact) {
      request->result.text =
          runner_pool->runner().Decode(request->result.tokens);
    }
    FinalizeTiming(request);
  }

  void CompleteCancelled(
      const std::shared_ptr<ScheduledRequest>& request) noexcept {
    try {
      BeginWork(request);
      if (request->runner_request) {
        const auto retained = request->runner_request.Cancel();
        CopyCheckpointMetrics(request, retained);
        request->result.cache_snapshot_bytes = retained.snapshot_bytes;
        request->result.cache_snapshot_ms = retained.snapshot_ms;
        request->result.cache_disk_queued_bytes = retained.disk_queued_bytes;
        request->result.cache_disk_enqueue_ms = retained.disk_enqueue_ms;
      }
      request->result.cancelled = true;
      FinalizeResult(request, TextGenerationBackend::FinishReason::kCancelled);
      PublishTerminal(request, {}, true);
    } catch (...) {
      CompleteFailure(request, std::current_exception());
    }
  }

  void CompleteFailure(const std::shared_ptr<ScheduledRequest>& request,
                       std::exception_ptr failure) noexcept {
    BeginWork(request);
    if (request->runner_request) {
      request->runner_request.Invalidate();
    }
    request->result.completion_tokens = request->result.tokens.size();
    FinalizeTiming(request);
    PublishTerminal(request, std::move(failure), true);
  }

  void CompleteDeadline(
      const std::shared_ptr<ScheduledRequest>& request) noexcept {
    CompleteFailure(request, std::make_exception_ptr(TextGenerationError(
                                 TextGenerationErrorCode::kDeadlineExceeded,
                                 "text generation request deadline exceeded")));
  }

  void CompleteAdmissionTimeout(
      const std::shared_ptr<ScheduledRequest>& request) {
    CompleteFailure(request, std::make_exception_ptr(TextGenerationError(
                                 TextGenerationErrorCode::kAdmissionTimeout,
                                 "text generation admission wait exceeded")));
  }

  [[nodiscard]] bool QueueExpired(
      const std::shared_ptr<ScheduledRequest>& request) const {
    return !request->admitted_at && request->queue_deadline &&
           clock() >= *request->queue_deadline;
  }

  [[nodiscard]] bool CompleteIfStopped(
      const std::shared_ptr<ScheduledRequest>& request) noexcept {
    try {
      if (DeadlineExceeded(request)) {
        CompleteDeadline(request);
        return true;
      }
      if (CancellationRequested(request)) {
        CompleteCancelled(request);
        return true;
      }
      if (QueueExpired(request)) {
        CompleteAdmissionTimeout(request);
        return true;
      }
      return false;
    } catch (...) {
      CompleteFailure(request, std::current_exception());
      return true;
    }
  }

  void CompleteSuccess(const std::shared_ptr<ScheduledRequest>& request,
                       TextGenerationBackend::FinishReason finish_reason) {
    const auto cache_commit = request->runner_request.Commit();
    CopyCheckpointMetrics(request, cache_commit);
    request->result.cache_snapshot_bytes = cache_commit.snapshot_bytes;
    request->result.cache_snapshot_ms = cache_commit.snapshot_ms;
    request->result.cache_disk_queued_bytes = cache_commit.disk_queued_bytes;
    request->result.cache_disk_enqueue_ms = cache_commit.disk_enqueue_ms;
    request->result.cache_shared_prefix_snapshots =
        cache_commit.shared_prefix_snapshots;
    request->result.cache_shared_prefix_bytes =
        cache_commit.shared_prefix_bytes;
    request->result.cache_shared_prefix_ms = cache_commit.shared_prefix_ms;
    FinalizeResult(request, finish_reason);
    PublishTerminal(request);
  }

  void ProcessQueuedCancellations() {
    for (const auto& request : QueuedSnapshot()) {
      try {
        const bool deadline_exceeded = DeadlineExceeded(request);
        const bool cancelled =
            !deadline_exceeded && CancellationRequested(request);
        const bool queue_expired = QueueExpired(request);
        if (!(deadline_exceeded || cancelled || queue_expired) ||
            !RemoveQueued(request)) {
          continue;
        }
        if (deadline_exceeded) {
          CompleteDeadline(request);
        } else if (cancelled) {
          CompleteCancelled(request);
        } else {
          CompleteAdmissionTimeout(request);
        }
      } catch (...) {
        if (RemoveQueued(request)) {
          CompleteFailure(request, std::current_exception());
        }
      }
    }
  }

  void Admit(std::deque<std::shared_ptr<ScheduledRequest>>& prefilling,
             std::deque<std::shared_ptr<ScheduledRequest>>& decoding,
             std::size_t capturing, const std::stop_token& stop_token) {
    // Captures retain their runner lease until decoding resumes.
    while (!stop_token.stop_requested() &&
           prefilling.size() + decoding.size() + capturing <
               runner_pool->capacity()) {
      auto request = PopQueued();
      if (request == nullptr) {
        return;
      }
      try {
        if (CompleteIfStopped(request)) {
          continue;
        }

        request->admitted_at = clock();
        {
          const std::lock_guard<std::mutex> lock(queue_mutex);
          ++resident_count;
        }
        const std::weak_ptr<ScheduledRequest> weak_request = request;
        request->runner_request = runner_pool->Acquire(
            std::move(request->prompt), request->sampling,
            [weak_request, stop_token] {
              const auto request = weak_request.lock();
              return stop_token.stop_requested() || request == nullptr ||
                     CancellationRequested(request) ||
                     DeadlineExceeded(request);
            },
            std::move(request->prompt_context), request->cache_prompt,
            request->cache_prefix_tokens, request->json_constraint,
            std::move(request->cache_boundaries));
        if (!request->runner_request) {
          CompleteCancelled(request);
          continue;
        }

        request->result.cache_hit = request->runner_request.cache_hit();
        const auto lookup = request->runner_request.cache_lookup();
        request->result.cache_miss_reason = lookup.miss_reason;
        request->result.cache_common_prefix_tokens =
            lookup.common_prefix_tokens;
        request->result.cache_checkpoint_tokens = lookup.checkpoint_tokens;
        request->result.cached_prompt_tokens =
            request->runner_request.cached_prompt_tokens();
        request->result.cache_restore_bytes =
            request->runner_request.cache_restore_bytes();
        request->result.cache_restore_ms =
            request->runner_request.cache_restore_ms();
        request->result.cache_disk_hit =
            request->runner_request.cache_disk_hit();
        request->result.incremental_prefill_supported =
            incremental_prefill_supported;
        request->result.queue_ms = request->result.queue_admission_ms =
            Milliseconds(*request->admitted_at - request->request_start);
        request->result.effective_prefill_tokens =
            request->result.prompt_tokens -
            request->result.cached_prompt_tokens;
        ReservePrefill(request, request->result.effective_prefill_tokens);
        EndWork(request);
        request->result.resident_requests_at_admission =
            prefilling.size() + decoding.size() + capturing + 1;
        request->phase.store(TextRequestPhase::kAdmitted,
                             std::memory_order_release);
        request->phase.store(request->runner_request.prefill_complete()
                                 ? TextRequestPhase::kDecodeReady
                                 : TextRequestPhase::kPrefilling,
                             std::memory_order_release);
        if (request->runner_request.prefill_complete()) {
          // Replacements cannot renew a spent burst. Existing due decoders
          // still finish their round before the next prefill chunk.
          const bool spent =
              scheduler_policy.decode_burst.count() > 0 &&
              (decode_burst_ms >= scheduler_policy.decode_burst.count() ||
               decode_burst_steps >=
                   scheduler_policy.max_decode_steps_per_burst);
          request->decode_due = !spent;
          decoding.push_back(std::move(request));
        } else {
          prefilling.push_back(std::move(request));
        }
      } catch (...) {
        CompleteFailure(request, std::current_exception());
      }
    }
  }

  void StepPrefill(const std::shared_ptr<ScheduledRequest>& request,
                   bool decoder_runnable, bool snapshot_pending = false) {
    BeginWork(request);
    try {
      if (CompleteIfStopped(request)) {
        return;
      }

      request->phase.store(TextRequestPhase::kPrefilling,
                           std::memory_order_release);
      // A published first token is also latency-sensitive while its frozen
      // prompt is being captured. Bound other prefill work so we can poll
      // that capture promptly. Spare slots alone do not change lone prefill.
      const bool bounded_prefill = incremental_prefill_supported &&
                                   (decoder_runnable || snapshot_pending);
      std::size_t budget = bounded_prefill
                               ? prefill_policy.decode_active_tokens
                               : request->runner_request.prompt_tokens();
      if (bounded_prefill && prefill_policy.adaptive_chunking) {
        const auto cap = std::min(budget, kDefaultDecodeActivePrefillTokens);
        budget = cap;
        if (measured_prefill_ms_per_token > 0) {
          const double estimated = prefill_policy.target_chunk_time.count() /
                                   measured_prefill_ms_per_token;
          budget = static_cast<std::size_t>(std::clamp(
              estimated, static_cast<double>(prefill_policy.min_chunk_tokens),
              static_cast<double>(cap)));
        }
      }
      if ((decoder_runnable || runner_pool->capacity() > 1) &&
          !incremental_prefill_supported) {
        request->result.prefill_fallback_reason =
            "incremental_prefill_unavailable";
      }
      const auto start = clock();
      TextPrefillStep step;
      try {
        step = request->runner_request.Prefill(budget);
      } catch (...) {
        request->result.prefill_ms += Milliseconds(clock() - start);
        throw;
      }
      const double elapsed = Milliseconds(clock() - start);
      request->result.prefill_ms += elapsed;
      request->result.max_prefill_chunk_ms =
          std::max(request->result.max_prefill_chunk_ms, elapsed);
      if (step.consumed_tokens > 0 && elapsed > 0) {
        const double sample = elapsed / step.consumed_tokens;
        // React immediately to slower work, recover conservatively when faster.
        measured_prefill_ms_per_token = std::max(
            sample, 0.8 * measured_prefill_ms_per_token + 0.2 * sample);
      }
      ReservePrefill(request, request->reserved_prefill_tokens -
                                  std::min(request->reserved_prefill_tokens,
                                           step.consumed_tokens));
      request->result.prefill_tokens += step.consumed_tokens;
      ++request->result.prefill_chunks;
      request->result.max_prefill_chunk_tokens = std::max(
          request->result.max_prefill_chunk_tokens, step.consumed_tokens);

      if (decoder_runnable) {
        ++request->result.active_decode_prefill_chunks;
        ++consecutive_active_prefill_chunks;
        request->result.max_consecutive_active_prefill_chunks =
            std::max(request->result.max_consecutive_active_prefill_chunks,
                     consecutive_active_prefill_chunks);
      } else {
        consecutive_active_prefill_chunks = 0;
      }

      if (CompleteIfStopped(request)) {
        return;
      }

      if (step.decode_ready && !request->decode_ready_at)
        request->decode_ready_at = clock();
      request->phase.store(step.decode_ready ? TextRequestPhase::kDecodeReady
                                             : TextRequestPhase::kPrefilling,
                           std::memory_order_release);
    } catch (...) {
      const auto failure = std::current_exception();
      if (!CompleteIfStopped(request)) {
        CompleteFailure(request, failure);
      }
    }
  }

  template<class Work>
  auto MeasureDecode(const std::shared_ptr<ScheduledRequest>& request,
                     Work work) {
    const auto start = clock();
    try {
      if constexpr (std::is_void_v<std::invoke_result_t<Work>>) {
        work();
        const double elapsed = Milliseconds(clock() - start);
        request->result.decode_ms += elapsed;
        decode_service_ms += elapsed;
      } else {
        auto result = work();
        const double elapsed = Milliseconds(clock() - start);
        request->result.decode_ms += elapsed;
        decode_service_ms += elapsed;
        return result;
      }
    } catch (...) {
      const double elapsed = Milliseconds(clock() - start);
      request->result.decode_ms += elapsed;
      decode_service_ms += elapsed;
      throw;
    }
  }

  [[nodiscard]] std::optional<Clock::time_point> PrepareDecode(
      const std::shared_ptr<ScheduledRequest>& request) {
    if (CompleteIfStopped(request)) {
      return std::nullopt;
    }

    if (request->advance_pending) {
      if (!request->runner_request.PreparePromptSnapshot())
        return std::nullopt;
      request->advance_pending = false;
      if (!final_token_advance_required &&
          request->result.tokens.size() >= request->token_limit) {
        CompleteSuccess(request, TextGenerationBackend::FinishReason::kLength);
        return std::nullopt;
      }
      // Snapshot capture and work for other requests are outside this step.
      return clock();
    }
    request->phase.store(TextRequestPhase::kDecoding,
                         std::memory_order_release);
    const auto selection = MeasureDecode(
        request, [&] { return request->runner_request.SelectNext(); });
    if (selection.stop) {
      CompleteSuccess(request, TextGenerationBackend::FinishReason::kStop);
      return std::nullopt;
    }

    if (!PublishSelection(request, selection)) {
      return std::nullopt;
    }
    request->advance_pending = true;
    return PrepareDecode(request);
  }

  [[nodiscard]] bool PublishSelection(
      const std::shared_ptr<ScheduledRequest>& request,
      const TextDecodeSelection& selection) {
    if (selection.piece.size() >
        request->max_output_bytes - request->generated_output_bytes) {
      CompleteFailure(request,
                      std::make_exception_ptr(TextGenerationError(
                          TextGenerationErrorCode::kOutputLimit,
                          "text generation output byte limit exceeded")));
      return false;
    }
    const auto now = clock();
    if (!request->previous_token.has_value()) {
      request->result.first_token_emitted = true;
      request->result.ttft_ms = std::chrono::duration<double, std::milli>(
                                    now - request->request_start)
                                    .count();
    } else {
      const auto inter_token = now - *request->previous_token;
      request->inter_token_total += inter_token;
      request->result.max_inter_token_ms = std::max(
          request->result.max_inter_token_ms,
          std::chrono::duration<double, std::milli>(inter_token).count());
      ++request->inter_token_samples;
    }
    request->previous_token = now;
    request->generated_output_bytes += selection.piece.size();
    request->result.tokens.push_back(selection.token);
    if (!PublishPiece(request, selection.piece)) {
      CompleteFailure(request,
                      std::make_exception_ptr(TextGenerationError(
                          TextGenerationErrorCode::kOutputBackpressure,
                          "text generation buffered output limit exceeded")));
      return false;
    }
    if (incremental_text_is_exact) {
      request->result.text += selection.piece;
    }
    if (CompleteIfStopped(request)) {
      return false;
    }
    return true;
  }

  void FinishAdvanced(const std::shared_ptr<ScheduledRequest>& request) {
    if (CompleteIfStopped(request)) {
      return;
    }
    if (request->result.tokens.size() >= request->token_limit) {
      CompleteSuccess(request, TextGenerationBackend::FinishReason::kLength);
    }
  }

  void StepDecode(const std::shared_ptr<ScheduledRequest>& request) {
    consecutive_active_prefill_chunks = 0;
    if (multi_token_decode) {
      StepMultiTokenDecode(request);
      return;
    }
    try {
      const auto decode_start = PrepareDecode(request);
      if (!decode_start.has_value()) {
        return;
      }
      MeasureDecode(request, [&] { request->runner_request.Advance(); });
      FinishAdvanced(request);
    } catch (...) {
      const auto failure = std::current_exception();
      if (!CompleteIfStopped(request)) {
        CompleteFailure(request, failure);
      }
    }
  }

  void StepMultiTokenDecode(const std::shared_ptr<ScheduledRequest>& request) {
    consecutive_active_prefill_chunks = 0;
    try {
      if (CompleteIfStopped(request)) {
        return;
      }

      request->phase.store(TextRequestPhase::kDecoding,
                           std::memory_order_release);
      if (!PrepareFirstSnapshot(request))
        return;
      const std::size_t remaining =
          request->token_limit - request->result.tokens.size() +
          (request->preview_token.has_value() ? 1 : 0);
      const auto step = MeasureDecode(request, [&] {
        return request->runner_request.DecodeStep(remaining);
      });
      request->result.draft_tokens += step.draft_tokens;
      request->result.draft_accepted_tokens += step.draft_accepted_tokens;

      CheckPreviewResult(request, step);
      for (const auto& selection : step.selections) {
        if (!PublishDecodedSelection(request, selection)) {
          return;
        }
      }

      if (step.stop) {
        CompleteSuccess(request, TextGenerationBackend::FinishReason::kStop);
        return;
      }
      if (request->result.tokens.size() >= request->token_limit) {
        CompleteSuccess(request, TextGenerationBackend::FinishReason::kLength);
      }
    } catch (...) {
      const auto failure = std::current_exception();
      if (!CompleteIfStopped(request)) {
        CompleteFailure(request, failure);
      }
    }
  }

  bool PrepareFirstSnapshot(const std::shared_ptr<ScheduledRequest>& request) {
    if (request->result.tokens.empty() && !request->preview_stops) {
      const auto preview = MeasureDecode(
          request, [&] { return request->runner_request.PreviewFirstToken(); });
      if (preview && !preview->stop) {
        if (!PublishSelection(request, *preview))
          return false;
        request->preview_token = preview->token;
      }
      request->preview_stops = preview && preview->stop;
    }
    if (!request->runner_request.PreparePromptSnapshot())
      return false;
    if (request->preview_stops) {
      CompleteSuccess(request, TextGenerationBackend::FinishReason::kStop);
      return false;
    }
    return !CompleteIfStopped(request);
  }

  static void CheckPreviewResult(
      const std::shared_ptr<ScheduledRequest>& request,
      const TextDecodeStep& step) {
    if (request->preview_token &&
        (step.selections.empty() ||
         step.selections.front().token != *request->preview_token)) {
      throw std::runtime_error("first-token preview disagrees with decoding");
    }
  }

  bool PublishDecodedSelection(const std::shared_ptr<ScheduledRequest>& request,
                               const TextDecodeSelection& selection) {
    if (request->preview_token) {
      request->preview_token.reset();
      return true;
    }
    return PublishSelection(request, selection);
  }

  void StepDecodeBatch(
      const std::vector<std::shared_ptr<ScheduledRequest>>& requests) {
    const auto selected_plan = runner_pool->SelectDecodePlan(requests.size());
    if (batched_multi_token_decode &&
        (batched_multi_token_decode_max_width == 0 ||
         selected_plan.physical_width <=
             batched_multi_token_decode_max_width)) {
      StepMultiTokenDecodeBatch(requests);
      return;
    }
    if (requests.size() == 1) {
      StepDecode(requests.front());
      return;
    }

    consecutive_active_prefill_chunks = 0;
    struct PreparedRequest {
      std::shared_ptr<ScheduledRequest> request;
    };
    std::vector<PreparedRequest> prepared;
    prepared.reserve(requests.size());
    for (const auto& request : requests) {
      try {
        const auto decode_start = PrepareDecode(request);
        if (decode_start.has_value()) {
          prepared.push_back({
              .request = request,
          });
        }
      } catch (...) {
        CompleteFailure(request, std::current_exception());
      }
    }
    if (prepared.empty()) {
      return;
    }
    if (prepared.size() == 1) {
      try {
        MeasureDecode(prepared.front().request, [&] {
          prepared.front().request->runner_request.Advance();
        });
        FinishAdvanced(prepared.front().request);
      } catch (...) {
        CompleteFailure(prepared.front().request, std::current_exception());
      }
      return;
    }

    const auto plan = runner_pool->SelectDecodePlan(prepared.size());
    if (plan.kind != TextExecutionPlanKind::kBatched) {
      for (const auto& item : prepared) {
        try {
          MeasureDecode(item.request,
                        [&] { item.request->runner_request.Advance(); });
          FinishAdvanced(item.request);
        } catch (...) {
          CompleteFailure(item.request, std::current_exception());
        }
      }
      return;
    }

    std::vector<TextRunnerPool::Request*> runner_requests;
    runner_requests.reserve(prepared.size());
    for (const auto& item : prepared) {
      runner_requests.push_back(&item.request->runner_request);
    }
    std::vector<std::exception_ptr> failures;
    const auto batch_start = clock();
    try {
      failures = runner_pool->AdvanceBatch(runner_requests, plan);
    } catch (...) {
      const auto failure = std::current_exception();
      const double elapsed = Milliseconds(clock() - batch_start);
      decode_service_ms += elapsed;
      for (const auto& item : prepared) {
        item.request->result.decode_ms += elapsed;
        if (!CompleteIfStopped(item.request))
          CompleteFailure(item.request, failure);
      }
      return;
    }

    const auto batch_end = clock();
    const double batch_ms = Milliseconds(batch_end - batch_start);
    decode_service_ms += batch_ms;
    for (const auto& item : prepared)
      item.request->result.decode_ms += batch_ms;
    const std::string execution_plan =
        "batched-w" + std::to_string(plan.physical_width);
    for (std::size_t i = 0; i < prepared.size(); ++i) {
      const auto& item = prepared[i];
      if (failures[i]) {
        if (!CompleteIfStopped(item.request))
          CompleteFailure(item.request, failures[i]);
        continue;
      }
      item.request->result.physical_execution_width = std::max(
          item.request->result.physical_execution_width, plan.physical_width);
      item.request->result.execution_plan = execution_plan;
      FinishAdvanced(item.request);
    }
  }

  void StepMultiTokenDecodeBatch(
      const std::vector<std::shared_ptr<ScheduledRequest>>& requests) {
    if (requests.size() == 1) {
      StepMultiTokenDecode(requests.front());
      return;
    }

    consecutive_active_prefill_chunks = 0;
    struct PreparedRequest {
      std::shared_ptr<ScheduledRequest> request;
      std::size_t remaining;
    };
    std::vector<PreparedRequest> prepared;
    prepared.reserve(requests.size());
    for (const auto& request : requests) {
      if (CompleteIfStopped(request)) {
        continue;
      }
      request->phase.store(TextRequestPhase::kDecoding,
                           std::memory_order_release);
      try {
        if (!PrepareFirstSnapshot(request))
          continue;
      } catch (...) {
        CompleteFailure(request, std::current_exception());
        continue;
      }
      prepared.push_back({
          .request = request,
          .remaining = request->token_limit - request->result.tokens.size() +
                       (request->preview_token.has_value() ? 1 : 0),
      });
    }
    if (prepared.empty()) {
      return;
    }
    if (prepared.size() == 1) {
      StepMultiTokenDecode(prepared.front().request);
      return;
    }

    const auto plan = runner_pool->SelectDecodePlan(prepared.size());
    if (plan.kind != TextExecutionPlanKind::kBatched) {
      for (const auto& item : prepared) {
        StepMultiTokenDecode(item.request);
      }
      return;
    }

    std::vector<TextRunnerPool::Request*> runner_requests;
    std::vector<std::size_t> max_tokens;
    runner_requests.reserve(prepared.size());
    max_tokens.reserve(prepared.size());
    for (const auto& item : prepared) {
      runner_requests.push_back(&item.request->runner_request);
      max_tokens.push_back(item.remaining);
    }

    std::vector<TextDecodeStep> steps;
    const auto batch_start = clock();
    try {
      steps = runner_pool->DecodeBatch(runner_requests, max_tokens, plan);
    } catch (...) {
      const auto failure = std::current_exception();
      const double elapsed = Milliseconds(clock() - batch_start);
      decode_service_ms += elapsed;
      for (const auto& item : prepared) {
        item.request->result.decode_ms += elapsed;
        if (!CompleteIfStopped(item.request))
          CompleteFailure(item.request, failure);
      }
      return;
    }

    const auto batch_end = clock();
    const double batch_ms = Milliseconds(batch_end - batch_start);
    decode_service_ms += batch_ms;
    for (const auto& item : prepared)
      item.request->result.decode_ms += batch_ms;
    for (std::size_t index = 0; index < prepared.size(); ++index) {
      const auto& item = prepared[index];
      const auto& step = steps[index];
      if (step.failure) {
        if (!CompleteIfStopped(item.request))
          CompleteFailure(item.request, step.failure);
        continue;
      }
      if (step.execution_plan.physical_width >=
          item.request->result.physical_execution_width) {
        item.request->result.physical_execution_width =
            step.execution_plan.physical_width;
        item.request->result.execution_plan =
            step.execution_plan.kind == TextExecutionPlanKind::kBatched
                ? "batched-w" +
                      std::to_string(step.execution_plan.physical_width)
                : (runner_pool->capacity() == 1 ? "serial-c1"
                                                : "serial-fallback");
      }
      item.request->result.draft_tokens += step.draft_tokens;
      item.request->result.draft_accepted_tokens += step.draft_accepted_tokens;

      bool published = true;
      try {
        CheckPreviewResult(item.request, step);
      } catch (...) {
        CompleteFailure(item.request, std::current_exception());
        continue;
      }
      for (const auto& selection : step.selections) {
        if (!PublishDecodedSelection(item.request, selection)) {
          published = false;
          break;
        }
      }
      if (!published || IsTerminal(item.request)) {
        continue;
      }
      if (step.stop) {
        CompleteSuccess(item.request,
                        TextGenerationBackend::FinishReason::kStop);
      } else if (item.request->result.tokens.size() >=
                 item.request->token_limit) {
        CompleteSuccess(item.request,
                        TextGenerationBackend::FinishReason::kLength);
      }
    }
  }

  [[nodiscard]] static bool HasDueDecoder(
      const std::deque<std::shared_ptr<ScheduledRequest>>& decoding) {
    return std::any_of(decoding.begin(), decoding.end(),
                       [](const auto& request) { return request->decode_due; });
  }

  static void MarkAllDecodersDue(
      std::deque<std::shared_ptr<ScheduledRequest>>& decoding) {
    for (const auto& request : decoding) {
      request->decode_due = true;
    }
  }

  [[nodiscard]] static std::shared_ptr<ScheduledRequest> PopDecoder(
      std::deque<std::shared_ptr<ScheduledRequest>>& decoding,
      bool require_due) {
    const std::size_t candidates = decoding.size();
    for (std::size_t index = 0; index < candidates; ++index) {
      auto request = std::move(decoding.front());
      decoding.pop_front();
      if (!require_due || request->decode_due) {
        return request;
      }
      decoding.push_back(std::move(request));
    }
    auto request = std::move(decoding.front());
    decoding.pop_front();
    return request;
  }

  void CancelRemaining(
      std::deque<std::shared_ptr<ScheduledRequest>>& prefilling,
      std::deque<std::shared_ptr<ScheduledRequest>>& decoding) noexcept {
    std::vector<std::shared_ptr<ScheduledRequest>> remaining_queued;
    {
      const std::lock_guard<std::mutex> lock(queue_mutex);
      remaining_queued.reserve(queued_count);
      for (auto& client : queued_clients) {
        std::move(client.requests.begin(), client.requests.end(),
                  std::back_inserter(remaining_queued));
      }
      queued_clients.clear();
      queued_count = 0;
    }
    for (const auto& request : remaining_queued) {
      CompleteCancelled(request);
    }
    for (const auto& request : prefilling) {
      CompleteCancelled(request);
    }
    for (const auto& request : decoding) {
      CompleteCancelled(request);
    }
    prefilling.clear();
    decoding.clear();
  }

  void Run(const std::stop_token& stop_token) noexcept {
    std::deque<std::shared_ptr<ScheduledRequest>> prefilling;
    std::deque<std::shared_ptr<ScheduledRequest>> decoding;
    std::deque<std::shared_ptr<ScheduledRequest>> capturing;
    while (!stop_token.stop_requested()) {
      ProcessQueuedCancellations();

      for (std::size_t count = capturing.size(); count != 0; --count) {
        auto request = std::move(capturing.front());
        capturing.pop_front();
        if (request->runner_request.SnapshotPending()) {
          capturing.push_back(std::move(request));
        } else if (!CompleteIfStopped(request)) {
          AccountWait(request);
          request->waiting_for_snapshot = false;
          request->wait_start = clock();
          if (request->runner_request.prefill_complete() &&
              !request->decode_ready_at)
            request->decode_ready_at = clock();
          if (request->runner_request.prefill_complete())
            decoding.push_back(std::move(request));
          else
            prefilling.push_back(std::move(request));
        }
      }
      Admit(prefilling, decoding, capturing.size(), stop_token);
      if (prefilling.empty() && decoding.empty()) {
        decode_burst_ms = 0;
        decode_burst_steps = 0;
        std::unique_lock<std::mutex> lock(queue_mutex);
        const auto wake = [&] {
          return stop_token.stop_requested() || stopping ||
                 (queued_count != 0 &&
                  capturing.size() < runner_pool->capacity());
        };
        if (capturing.empty())
          queue_condition.wait(lock, wake);
        else
          queue_condition.wait_for(lock, std::chrono::milliseconds(1), wake);
        continue;
      }

      const bool due_decoder = HasDueDecoder(decoding);
      const std::size_t resident_count = prefilling.size() + decoding.size();
      const bool preparing_multi_token_batch =
          multi_token_decode && resident_count > 1 && !prefilling.empty() &&
          runner_pool->SelectDecodePlan(resident_count).kind ==
              TextExecutionPlanKind::kBatched;
      if (preparing_multi_token_batch) {
        for (const auto& pending : prefilling) {
          pending->runner_request.PrepareBatchExecution();
        }
        for (const auto& ready : decoding) {
          ready->runner_request.PrepareBatchExecution();
        }
      }
      // Give simultaneous new requests one bounded chunk to form their first
      // batch. Once decoding starts, every due decoder runs before more
      // prefill.
      const bool assemble_initial_batch =
          preparing_multi_token_batch &&
          consecutive_active_prefill_chunks == 0 &&
          std::all_of(decoding.begin(), decoding.end(),
                      [](const auto& request) {
                        return request->result.tokens.empty();
                      });
      const bool burst_complete =
          decode_burst_ms >= scheduler_policy.decode_burst.count() ||
          decode_burst_steps >= scheduler_policy.max_decode_steps_per_burst;
      if (!prefilling.empty() && !decoding.empty() &&
          ((!due_decoder && burst_complete) || assemble_initial_batch)) {
        auto request = std::move(prefilling.front());
        prefilling.pop_front();
        const auto started = clock();
        StepPrefill(request, true);
        const double elapsed = Milliseconds(clock() - started);
        for (const auto& peer : decoding)
          peer->result.peer_prefill_ms += elapsed;
        EndWork(request);
        decode_burst_ms = 0;
        decode_burst_steps = 0;
        if (!IsTerminal(request)) {
          if (request->runner_request.SnapshotPending()) {
            capturing.push_back(std::move(request));
          } else if (request->runner_request.prefill_complete()) {
            decoding.push_back(std::move(request));
          } else {
            prefilling.push_back(std::move(request));
          }
        }
        MarkAllDecodersDue(decoding);
        continue;
      }

      if (!decoding.empty()) {
        const std::size_t candidate_count =
            due_decoder
                ? static_cast<std::size_t>(std::count_if(
                      decoding.begin(), decoding.end(),
                      [](const auto& request) { return request->decode_due; }))
                : decoding.size();
        const auto plan = runner_pool->SelectDecodePlan(candidate_count);
        const std::size_t batch_size =
            plan.kind == TextExecutionPlanKind::kBatched
                ? std::min(candidate_count, plan.physical_width)
                : 1;
        std::vector<std::shared_ptr<ScheduledRequest>> batch;
        batch.reserve(batch_size);
        for (std::size_t index = 0; index < batch_size; ++index) {
          auto request = PopDecoder(decoding, due_decoder);
          request->decode_due = false;
          batch.push_back(std::move(request));
        }
        for (const auto& request : batch)
          BeginWork(request);
        const double service_before = decode_service_ms;
        StepDecodeBatch(batch);
        decode_burst_ms += decode_service_ms - service_before;
        ++decode_burst_steps;
        for (auto& request : batch) {
          EndWork(request);
          if (!IsTerminal(request)) {
            if (request->runner_request.SnapshotPending())
              capturing.push_back(std::move(request));
            else
              decoding.push_back(std::move(request));
          }
        }
        continue;
      }

      auto request = std::move(prefilling.front());
      prefilling.pop_front();
      StepPrefill(request, false, !capturing.empty());
      EndWork(request);
      decode_burst_ms = 0;
      decode_burst_steps = 0;
      if (!IsTerminal(request)) {
        if (request->runner_request.SnapshotPending()) {
          capturing.push_back(std::move(request));
        } else if (request->runner_request.prefill_complete()) {
          request->decode_due = true;
          decoding.push_back(std::move(request));
        } else {
          prefilling.push_back(std::move(request));
        }
      }
    }
    for (auto& request : capturing)
      decoding.push_back(std::move(request));
    CancelRemaining(prefilling, decoding);
  }

  std::shared_ptr<TextRunnerPool> runner_pool;
  TextPrefillPolicy prefill_policy;
  TextSchedulerPolicy scheduler_policy;
  ClockSource clock;
  std::shared_ptr<OutputBudget> output_budget;
  bool incremental_prefill_supported{false};
  bool final_token_advance_required{true};
  bool incremental_text_is_exact{false};
  bool multi_token_decode{false};
  bool batched_multi_token_decode{false};
  std::size_t batched_multi_token_decode_max_width{0};
  mutable std::mutex queue_mutex;
  std::condition_variable queue_condition;
  std::deque<PendingClient> queued_clients;
  std::size_t queued_count{0};
  std::size_t inflight_count{0};
  std::size_t resident_count{0};
  std::size_t sticky_admission_run{0};
  std::size_t reserved_prefill_tokens{0};
  std::unordered_map<std::string, std::size_t> inflight_clients;
  double measured_prefill_ms_per_token{0};
  double decode_service_ms{0};
  double decode_burst_ms{0};
  std::size_t decode_burst_steps{0};
  bool stopping{false};
  std::size_t consecutive_active_prefill_chunks{0};
  std::atomic<std::uint64_t> next_request_id{1};
  std::jthread worker;
};

TextGenerationScheduler::Request::Request() = default;

TextGenerationScheduler::Request::Request(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

TextGenerationScheduler::Request::~Request() {
  Cancel();
}

TextGenerationScheduler::Request::Request(Request&&) noexcept = default;

TextGenerationScheduler::Request& TextGenerationScheduler::Request::operator=(
    Request&& other) noexcept {
  if (this != &other) {
    Cancel();
    impl_ = std::move(other.impl_);
  }
  return *this;
}

TextGenerationScheduler::Request::operator bool() const noexcept {
  return impl_ != nullptr && impl_->request != nullptr;
}

std::uint64_t TextGenerationScheduler::Request::id() const noexcept {
  return *this ? impl_->request->id : 0;
}

TextRequestPhase TextGenerationScheduler::Request::phase() const noexcept {
  return *this ? impl_->request->phase.load(std::memory_order_acquire)
               : TextRequestPhase::kTerminal;
}

TextGenerationScheduler::Result TextGenerationScheduler::Request::Wait(
    const TokenCallback& on_token) {
  if (!*this) {
    throw std::logic_error("text scheduler request is empty");
  }
  if (impl_->waited) {
    throw std::logic_error("text scheduler request was already consumed");
  }
  impl_->waited = true;

  bool deliver_pieces = true;
  bool consumer_cancelled = false;
  std::exception_ptr callback_failure;
  Result result;
  std::exception_ptr scheduler_failure;

  while (true) {
    std::string piece;
    bool terminal = false;
    {
      std::unique_lock<std::mutex> lock(impl_->request->output_mutex);
      impl_->request->output_condition.wait(lock, [&] {
        return impl_->request->terminal ||
               !impl_->request->output_pieces.empty();
      });
      if (!impl_->request->output_pieces.empty()) {
        const std::size_t piece_bytes =
            impl_->request->output_pieces.front().size();
        piece = std::move(impl_->request->output_pieces.front());
        impl_->request->output_pieces.pop_front();
        impl_->request->buffered_output_bytes -= piece_bytes;
        impl_->request->output_budget->Release(piece_bytes);
      } else if (impl_->request->terminal) {
        result = impl_->request->result;
        scheduler_failure = impl_->request->failure;
        terminal = true;
      }
    }

    if (!piece.empty() && deliver_pieces && on_token) {
      try {
        if (!on_token(piece)) {
          consumer_cancelled = true;
          deliver_pieces = false;
          Cancel();
        }
      } catch (...) {
        callback_failure = std::current_exception();
        deliver_pieces = false;
        Cancel();
      }
    }
    if (terminal) {
      break;
    }
  }

  if (callback_failure != nullptr) {
    std::rethrow_exception(callback_failure);
  }
  if (scheduler_failure != nullptr) {
    std::rethrow_exception(scheduler_failure);
  }
  if (consumer_cancelled) {
    result.cancelled = true;
    result.finish_reason = TextGenerationBackend::FinishReason::kCancelled;
  }
  return result;
}

void TextGenerationScheduler::Request::Cancel() noexcept {
  if (*this && !IsTerminal(impl_->request)) {
    impl_->request->cancellation_requested.store(true,
                                                 std::memory_order_release);
  }
}

TextGenerationScheduler::TextGenerationScheduler(
    std::shared_ptr<TextRunnerPool> runner_pool,
    TextPrefillPolicy prefill_policy, TextSchedulerPolicy scheduler_policy,
    ClockSource clock)
    : impl_(std::make_unique<Impl>(std::move(runner_pool), prefill_policy,
                                   scheduler_policy, std::move(clock))) {}

TextGenerationScheduler::~TextGenerationScheduler() = default;

const TextModelRunner& TextGenerationScheduler::runner() const noexcept {
  return impl_->runner_pool->runner();
}

std::size_t TextGenerationScheduler::capacity() const noexcept {
  return impl_->runner_pool->capacity();
}

std::size_t TextGenerationScheduler::buffered_output_bytes() const noexcept {
  return impl_->output_budget->buffered_bytes.load(std::memory_order_relaxed);
}

std::size_t TextGenerationScheduler::max_buffered_output_bytes()
    const noexcept {
  return impl_->output_budget->max_buffered_bytes.load(
      std::memory_order_relaxed);
}

TextGenerationScheduler::Request TextGenerationScheduler::Submit(
    std::vector<TextRunnerToken> prompt, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling,
    const CancellationCheck& is_cancelled, bool publish_token_pieces) {
  return Submit(std::move(prompt), max_tokens, sampling, is_cancelled,
                publish_token_pieces, RequestMetadata{});
}

TextGenerationScheduler::Request TextGenerationScheduler::Submit(
    std::vector<TextRunnerToken> prompt, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling,
    const CancellationCheck& is_cancelled, bool publish_token_pieces,
    RequestMetadata metadata) {
  if (prompt.empty()) {
    throw std::invalid_argument("text scheduler prompt must not be empty");
  }
  sampling.Validate();
  if (metadata.json_constraint && sampling.temperature != 0)
    throw std::invalid_argument(
        "JSON constrained decoding requires temperature 0");
  if (metadata.json_constraint &&
      !impl_->runner_pool->runner().Descriptor().capabilities.json_constraints)
    throw std::invalid_argument(
        "model does not support JSON constrained decoding");

  if (metadata.estimated_prefill_tokens &&
      *metadata.estimated_prefill_tokens > prompt.size())
    throw std::invalid_argument("prefill estimate exceeds prompt size");
  if (!metadata.cache_prompt)
    metadata.estimated_prefill_tokens = prompt.size();
  auto request = std::make_shared<ScheduledRequest>();
  request->id = impl_->next_request_id.fetch_add(1, std::memory_order_relaxed);
  request->client_id =
      metadata.client_id.empty() ? "anonymous" : std::move(metadata.client_id);
  request->result.request_id = request->id;
  request->result.estimated_prefill_tokens = metadata.estimated_prefill_tokens;
  request->result.prompt_tokens = prompt.size();
  request->result.client_id = request->client_id;
  request->result.configured_active_prefill_tokens =
      impl_->prefill_policy.decode_active_tokens;
  request->result.requested_logical_concurrency =
      impl_->runner_pool->capacity();
  request->result.execution_plan =
      impl_->runner_pool->capacity() == 1 ? "serial-c1" : "serial-fallback";
  request->prompt = std::move(prompt);
  request->prompt_context = std::move(metadata.prompt_context);
  request->json_constraint = std::move(metadata.json_constraint);
  request->cache_boundaries = std::move(metadata.cache_boundaries);
  request->cache_prompt = metadata.cache_prompt;
  request->cache_prefix_tokens = metadata.cache_prefix_tokens;
  request->token_limit = max_tokens > 0 ? max_tokens : 1;
  request->sampling = sampling;
  request->external_cancellation = is_cancelled;
  request->publish_token_pieces = publish_token_pieces;
  request->clock = impl_->clock;
  request->submitted_at = impl_->clock();
  request->request_start = metadata.request_start;
  if (impl_->scheduler_policy.queue_timeout.count() > 0)
    request->queue_deadline =
        request->submitted_at + impl_->scheduler_policy.queue_timeout;
  request->deadline = metadata.deadline;
  if (!request->deadline.has_value() &&
      impl_->scheduler_policy.request_timeout.count() > 0) {
    request->deadline =
        request->request_start + impl_->scheduler_policy.request_timeout;
  }
  request->max_output_bytes =
      impl_->scheduler_policy.max_output_bytes_per_request;
  request->max_buffered_output_bytes =
      impl_->scheduler_policy.max_buffered_output_bytes_per_request;
  request->output_budget = impl_->output_budget;

  {
    const std::lock_guard<std::mutex> lock(impl_->queue_mutex);
    if (impl_->stopping) {
      throw TextGenerationError(TextGenerationErrorCode::kSchedulerStopping,
                                "text generation scheduler is stopping");
    }
    if (impl_->queued_count >= impl_->scheduler_policy.max_pending_requests) {
      throw TextGenerationError(TextGenerationErrorCode::kQueueFull,
                                "text generation pending queue is full");
    }
    const auto inflight_client =
        impl_->inflight_clients.find(request->client_id);
    const std::size_t client_inflight =
        inflight_client == impl_->inflight_clients.end()
            ? 0
            : inflight_client->second;
    const auto client_limit =
        impl_->scheduler_policy.max_inflight_requests_per_client;
    if (client_limit > 0 && client_inflight >= client_limit)
      throw TextGenerationError(
          TextGenerationErrorCode::kClientInflightFull,
          "text generation client inflight limit exceeded");
    const auto estimate = request->result.estimated_prefill_tokens;
    const auto token_limit =
        impl_->scheduler_policy.max_inflight_prefill_tokens;
    if (token_limit > 0 && estimate &&
        (*estimate > token_limit ||
         impl_->reserved_prefill_tokens > token_limit - *estimate))
      throw TextGenerationError(
          TextGenerationErrorCode::kPrefillBudgetFull,
          "text generation known prefill budget exceeded");
    auto client =
        std::find_if(impl_->queued_clients.begin(), impl_->queued_clients.end(),
                     [&](const PendingClient& pending) {
                       return pending.client_id == request->client_id;
                     });
    if (client != impl_->queued_clients.end() &&
        client->requests.size() >=
            impl_->scheduler_policy.max_pending_requests_per_client) {
      throw TextGenerationError(TextGenerationErrorCode::kClientQueueFull,
                                "text generation client pending queue is full");
    }
    if (client == impl_->queued_clients.end()) {
      impl_->queued_clients.push_back({
          .client_id = request->client_id,
          .requests = {},
      });
      client = std::prev(impl_->queued_clients.end());
    }
    request->result.queue_depth_at_submit = impl_->queued_count + 1;
    request->result.client_queue_depth_at_submit = client->requests.size() + 1;
    client->requests.push_back(request);
    ++impl_->queued_count;
    request->accounted = true;
    request->reserved_prefill_tokens = estimate.value_or(0);
    impl_->reserved_prefill_tokens += request->reserved_prefill_tokens;
    request->result.resident_requests_at_submit = impl_->resident_count;
    request->result.inflight_requests_at_submit = ++impl_->inflight_count;
    request->result.client_inflight_requests_at_submit =
        ++impl_->inflight_clients[request->client_id];
  }
  impl_->queue_condition.notify_one();
  return Request(std::make_unique<Request::Impl>(std::move(request)));
}

TextGenerationScheduler::Request TextGenerationScheduler::Submit(
    std::vector<TextRunnerToken> prompt, std::size_t max_tokens,
    float temperature, const CancellationCheck& is_cancelled,
    bool publish_token_pieces) {
  sampling::SamplingConfig config;
  config.temperature = temperature;
  return Submit(std::move(prompt), max_tokens, config, is_cancelled,
                publish_token_pieces);
}

TextGenerationScheduler::Request TextGenerationScheduler::Submit(
    std::vector<TextRunnerToken> prompt, std::size_t max_tokens,
    float temperature, const CancellationCheck& is_cancelled,
    bool publish_token_pieces, RequestMetadata metadata) {
  sampling::SamplingConfig config;
  config.temperature = temperature;
  return Submit(std::move(prompt), max_tokens, config, is_cancelled,
                publish_token_pieces, std::move(metadata));
}

}  // namespace gufo::server
