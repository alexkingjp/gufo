#ifndef GUFO_SERVER_GENERATION_METRICS_HPP_
#define GUFO_SERVER_GENERATION_METRICS_HPP_

#include <iomanip>
#include <sstream>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/core/json.hpp"

namespace gufo::server {

inline double PrefillTokensPerSecond(
    const TextGenerationBackend::Result& result) {
  return result.prefill_ms > 0.0 ? static_cast<double>(result.prefill_tokens) *
                                       1000.0 / result.prefill_ms
                                 : 0.0;
}

inline double ActiveDecodeTokensPerSecond(
    const TextGenerationBackend::Result& result) {
  return result.decode_ms > 0.0
             ? 1000.0 * result.completion_tokens / result.decode_ms
             : 0.0;
}

inline double DecodeWallTokensPerSecond(
    const TextGenerationBackend::Result& result) {
  return result.decode_wall_ms > 0.0
             ? 1000.0 * result.completion_tokens / result.decode_wall_ms
             : 0.0;
}

inline double GenerationWallTokensPerSecond(
    const TextGenerationBackend::Result& result) {
  return result.total_generation_wall_ms > 0.0
             ? 1000.0 * result.completion_tokens /
                   result.total_generation_wall_ms
             : 0.0;
}

inline std::string GenerationLogDetails(
    const TextGenerationBackend::Result& result) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(1)
      << "prompt_tokens=" << result.prompt_tokens
      << " prefill_tokens=" << result.prefill_tokens
      << " generated_tokens=" << result.completion_tokens << " finish="
      << (result.cancelled ? "cancelled"
          : result.finish_reason == TextGenerationBackend::FinishReason::kLength
              ? "length"
              : "stop")
      << " cache="
      << (result.cache_disk_hit ? "disk"
          : result.cache_hit    ? "memory"
                                : "miss")
      << " cached_tokens=" << result.cached_prompt_tokens
      << " cache_restore_ms=" << result.cache_restore_ms
      << " queue_depth=" << result.queue_depth_at_submit
      << " resident_at_admission=" << result.resident_requests_at_admission
      << " request_id=" << result.request_id
      << " resident_at_submit=" << result.resident_requests_at_submit
      << " inflight_at_submit=" << result.inflight_requests_at_submit
      << " client_inflight_at_submit="
      << result.client_inflight_requests_at_submit
      << " effective_prefill_tokens=" << result.effective_prefill_tokens
      << " queue_ms=" << result.queue_ms
      << " queue_admission_ms=" << result.queue_admission_ms
      << " resident_wait_ms=" << result.resident_wait_ms
      << " peer_prefill_ms=" << result.peer_prefill_ms
      << " snapshot_wait_ms=" << result.snapshot_wait_ms
      << " total_generation_wall_ms=" << result.total_generation_wall_ms
      << " decode_wall_ms=" << result.decode_wall_ms
      << " first_token_emitted=" << result.first_token_emitted
      << " ttft_ms=" << result.ttft_ms
      << " max_prefill_chunk_ms=" << result.max_prefill_chunk_ms
      << " prefill_tps=" << PrefillTokensPerSecond(result)
      << " active_decode_tps=" << ActiveDecodeTokensPerSecond(result)
      << " decode_wall_tps=" << DecodeWallTokensPerSecond(result)
      << " generation_wall_tps=" << GenerationWallTokensPerSecond(result)
      << " batch_width=" << result.physical_execution_width
      << " plan=" << result.execution_plan
      << " draft_accepted=" << result.draft_accepted_tokens
      << " draft_proposed=" << result.draft_tokens;
  if (result.draft_tokens > 0)
    out << " acceptance_pct="
        << 100.0 * result.draft_accepted_tokens / result.draft_tokens;
  if (result.cache_snapshot_bytes > 0)
    out << " cache_snapshot_bytes=" << result.cache_snapshot_bytes;
  if (result.cache_disk_queued_bytes > 0)
    out << " cache_disk_queued_bytes=" << result.cache_disk_queued_bytes;
  if (!result.cache_hit && !result.cache_miss_reason.empty())
    out << " cache_miss_reason=" << result.cache_miss_reason
        << " common_prefix_tokens=" << result.cache_common_prefix_tokens
        << " nearest_checkpoint_tokens=" << result.cache_checkpoint_tokens;
  return out.str();
}

/// Timed prefill counts work actually executed; usage counts the full prompt.
inline json::Value GenerationTimings(
    const TextGenerationBackend::Result& result) {
  json::Value timings = json::Value::object();
  timings["prompt_n"] = result.prefill_tokens;
  timings["prompt_ms"] = result.prefill_ms;
  timings["prompt_per_token_ms"] =
      result.prefill_tokens > 0
          ? result.prefill_ms / static_cast<double>(result.prefill_tokens)
          : 0.0;
  timings["prompt_per_second"] = PrefillTokensPerSecond(result);
  timings["predicted_n"] = result.completion_tokens;
  timings["predicted_ms"] = result.decode_ms;
  timings["predicted_per_token_ms"] =
      result.completion_tokens > 0
          ? result.decode_ms / static_cast<double>(result.completion_tokens)
          : 0.0;
  timings["predicted_per_second"] =
      result.decode_ms > 0.0 ? static_cast<double>(result.completion_tokens) *
                                   1000.0 / result.decode_ms
                             : 0.0;
  // Keep legacy active-work fields for compatibility, but label their basis.
  timings["predicted_time_basis"] = "active_decode";
  timings["active_decode_ms"] = result.decode_ms;
  timings["active_decode_per_second"] = ActiveDecodeTokensPerSecond(result);
  timings["decode_wall_ms"] = result.decode_wall_ms;
  timings["decode_wall_per_second"] = DecodeWallTokensPerSecond(result);
  timings["total_generation_wall_ms"] = result.total_generation_wall_ms;
  timings["generation_wall_per_second"] = GenerationWallTokensPerSecond(result);
  timings["queue_admission_ms"] = result.queue_admission_ms;
  timings["resident_wait_ms"] = result.resident_wait_ms;
  timings["peer_prefill_ms"] = result.peer_prefill_ms;
  timings["snapshot_wait_ms"] = result.snapshot_wait_ms;
  timings["first_token_emitted"] = result.first_token_emitted;
  timings["ttft_ms"] = result.ttft_ms;
  timings["cache_n"] = result.cached_prompt_tokens;
  timings["cache_restore_ms"] = result.cache_restore_ms;
  timings["cache_snapshot_ms"] = result.cache_snapshot_ms;
  timings["cache_disk_enqueue_ms"] = result.cache_disk_enqueue_ms;
  timings["draft_n"] = result.draft_tokens;
  timings["draft_n_accepted"] = result.draft_accepted_tokens;
  return timings;
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_GENERATION_METRICS_HPP_
