#ifndef GUFO_SERVER_TEXT_GENERATION_SCHEDULER_HPP_
#define GUFO_SERVER_TEXT_GENERATION_SCHEDULER_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/cli/serve/text_model_runner.hpp"

namespace gufo::server {

enum class TextRequestPhase : std::uint8_t {
  kQueued,
  kAdmitted,
  kPrefilling,
  kDecodeReady,
  kDecoding,
  kTerminal,
};

inline constexpr std::size_t kDefaultDecodeActivePrefillTokens = 512;
inline constexpr std::size_t kDefaultMaxOutputBytes =
    static_cast<std::size_t>(1024) * 1024;
inline constexpr std::size_t kDefaultMaxBufferedOutputBytes =
    static_cast<std::size_t>(64) * 1024;
inline constexpr std::size_t kDefaultMaxBufferedOutputBytesTotal =
    static_cast<std::size_t>(256) * 1024;

struct TextPrefillPolicy {
  std::size_t decode_active_tokens{kDefaultDecodeActivePrefillTokens};
  std::chrono::milliseconds target_chunk_time{100};
  std::size_t min_chunk_tokens{1};
  /// Opt-in until changed physical prefill shapes pass model numerical gates.
  bool adaptive_chunking{false};
};

struct TextSchedulerPolicy {
  std::size_t max_pending_requests{16};
  std::size_t max_pending_requests_per_client{4};
  std::size_t max_output_bytes_per_request{kDefaultMaxOutputBytes};
  std::size_t max_buffered_output_bytes_per_request{
      kDefaultMaxBufferedOutputBytes};
  std::size_t max_buffered_output_bytes_total{
      kDefaultMaxBufferedOutputBytesTotal};
  std::chrono::milliseconds request_timeout{0};
  /// Zero retains the legacy one-round decode policy; checked at work
  /// boundaries.
  std::chrono::milliseconds decode_burst{0};
  std::size_t max_decode_steps_per_burst{32};
  /// Zero disables queue expiry. Admitted requests use request_timeout instead.
  std::chrono::milliseconds queue_timeout{0};
  /// Includes pending and resident requests. Zero disables this additional cap.
  std::size_t max_inflight_requests_per_client{0};
  bool prefer_short_prefill{false};
  std::size_t max_admission_bypasses{2};
  std::chrono::milliseconds admission_aging{1000};
  /// Bounds known remaining work only; unknown cache eligibility is not a miss.
  std::size_t max_inflight_prefill_tokens{0};
  /// Prefer a queued request whose input identity matches the resident
  /// lineage of a free runner slot: the slot is adopted without a restore
  /// and the live state keeps serving the same conversation in place.
  /// Per-client FIFO is untouched; unrelated heads wait behind at most
  /// capacity() consecutive sticky admissions.
  bool sticky_admission{true};
};

/// Single-owner scheduler for opaque text-model runner states.
///
/// Submitters never execute model code. One scheduler thread owns admission,
/// runner leases, prefill/decode work units, cancellation, and reclamation.
class TextGenerationScheduler {
public:
  using Clock = std::chrono::steady_clock;
  /// Must be monotonic, thread-safe and non-throwing; injectable for CPU tests.
  using ClockSource = std::function<Clock::time_point()>;
  using Result = TextGenerationBackend::Result;
  using CancellationCheck = TextGenerationBackend::CancellationCheck;
  using TokenCallback = TextGenerationBackend::TokenCallback;

  struct RequestMetadata {
    std::string client_id{"anonymous"};
    std::optional<Clock::time_point> deadline;
    Clock::time_point request_start{Clock::now()};
    std::shared_ptr<const TextPromptContext> prompt_context;
    bool cache_prompt{true};
    std::size_t cache_prefix_tokens{0};
    std::shared_ptr<const sampling::TokenConstraint> json_constraint;
    std::vector<TextCacheBoundary> cache_boundaries;
    /// Source-verified eligible uncached work, not a diagnostic longest prefix.
    /// Missing means unknown and cannot cause a token-budget rejection.
    std::optional<std::size_t> estimated_prefill_tokens;
  };

  class Request {
  public:
    Request();
    ~Request();

    Request(const Request&) = delete;
    Request& operator=(const Request&) = delete;
    Request(Request&&) noexcept;
    Request& operator=(Request&&) noexcept;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] std::uint64_t id() const noexcept;
    [[nodiscard]] TextRequestPhase phase() const noexcept;

    /// Consumes queued output pieces on the calling thread and waits for the
    /// scheduler-owned request to become terminal.
    Result Wait(const TokenCallback& on_token = {});
    void Cancel() noexcept;

  private:
    friend class TextGenerationScheduler;
    struct Impl;

    explicit Request(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
  };

  explicit TextGenerationScheduler(std::shared_ptr<TextRunnerPool> runner_pool,
                                   TextPrefillPolicy prefill_policy = {},
                                   TextSchedulerPolicy scheduler_policy = {},
                                   ClockSource clock = Clock::now);
  ~TextGenerationScheduler();

  TextGenerationScheduler(const TextGenerationScheduler&) = delete;
  TextGenerationScheduler& operator=(const TextGenerationScheduler&) = delete;
  TextGenerationScheduler(TextGenerationScheduler&&) = delete;
  TextGenerationScheduler& operator=(TextGenerationScheduler&&) = delete;

  [[nodiscard]] const TextModelRunner& runner() const noexcept;
  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] std::size_t buffered_output_bytes() const noexcept;
  [[nodiscard]] std::size_t max_buffered_output_bytes() const noexcept;

  [[nodiscard]] Request Submit(std::vector<TextRunnerToken> prompt,
                               std::size_t max_tokens,
                               const sampling::SamplingConfig& sampling,
                               const CancellationCheck& is_cancelled = {},
                               bool publish_token_pieces = false);

  [[nodiscard]] Request Submit(std::vector<TextRunnerToken> prompt,
                               std::size_t max_tokens,
                               const sampling::SamplingConfig& sampling,
                               const CancellationCheck& is_cancelled,
                               bool publish_token_pieces,
                               RequestMetadata metadata);

  [[nodiscard]] Request Submit(std::vector<TextRunnerToken> prompt,
                               std::size_t max_tokens, float temperature,
                               const CancellationCheck& is_cancelled = {},
                               bool publish_token_pieces = false);

  [[nodiscard]] Request Submit(std::vector<TextRunnerToken> prompt,
                               std::size_t max_tokens, float temperature,
                               const CancellationCheck& is_cancelled,
                               bool publish_token_pieces,
                               RequestMetadata metadata);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

using TextRequestMetadata = TextGenerationScheduler::RequestMetadata;

}  // namespace gufo::server

#endif  // GUFO_SERVER_TEXT_GENERATION_SCHEDULER_HPP_
