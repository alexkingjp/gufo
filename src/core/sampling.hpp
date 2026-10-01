#ifndef GUFO_CORE_SAMPLING_HPP_
#define GUFO_CORE_SAMPLING_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "src/core/json_constraint.hpp"

namespace gufo::sampling {

using TokenId = std::uint32_t;

/// Immutable tokenizer binding shared by requests, never a decoding cursor.
class TokenConstraint {
public:
  struct Piece {
    std::string bytes;
    bool stop{false};
    bool special{false};
  };
  using Vocabulary = std::vector<Piece>;
  struct Options {
    bool starts_in_reasoning{false};
    bool allow_tool_calls{false};
    bool require_tool_call{false};
  };
  struct Cursor {
    enum class Phase {
      kReasoning,
      kAnswerStart,
      kJson,
      kToolStart,
      kToolBody,
      kAfterTool
    };
    JsonConstraint grammar;
    Phase phase;
    std::string pending;
    enum class ToolPhase {
      kStart,
      kJson,
      kFunctionPrefix,
      kFunctionName,
      kBetweenParameters,
      kParameterName,
      kParameterBody,
      kClose
    };
    ToolPhase tool_phase{ToolPhase::kStart};
    std::vector<char> json_closers;
    bool json_string{false};
    bool json_escape{false};
  };

  TokenConstraint(std::shared_ptr<const JsonConstraint> prototype,
                  std::shared_ptr<const Vocabulary> vocabulary);
  TokenConstraint(std::shared_ptr<const JsonConstraint> prototype,
                  std::shared_ptr<const Vocabulary> vocabulary,
                  Options options);
  [[nodiscard]] Cursor Start() const;
  [[nodiscard]] bool Complete(const Cursor& cursor) const;
  [[nodiscard]] bool Allows(const Cursor& cursor, TokenId token) const;
  void Accept(Cursor& cursor, TokenId token) const;

private:
  void Advance(Cursor& cursor, std::string_view bytes) const;
  void AdvanceToolBody(Cursor& cursor, char byte) const;
  std::shared_ptr<const JsonConstraint> prototype_;
  std::shared_ptr<const Vocabulary> vocabulary_;
  Options options_;
};

/// Model-independent controls for selecting a token from a logit row.
///
/// Filters run in this order: penalties, temperature, top-k, top-p, min-p.
/// The defaults preserve Gufo's existing greedy decoding behavior.
struct SamplingConfig {
  float temperature{0.0F};
  std::int32_t top_k{0};
  float top_p{1.0F};
  float min_p{0.0F};
  std::size_t min_keep{0};
  std::int64_t seed{-1};
  float repeat_penalty{1.0F};
  std::size_t repeat_last_n{64};
  float frequency_penalty{0.0F};
  float presence_penalty{0.0F};

  void Validate() const;

  [[nodiscard]] bool penalties_enabled() const noexcept;
  [[nodiscard]] bool uses_random_sampling() const noexcept;
  [[nodiscard]] bool can_use_unmodified_argmax() const noexcept;
};

/// Unique token counts: repetition window and full generated response.
struct TokenPenalty {
  TokenId token{0};
  std::uint32_t generated_count{0};
  std::uint32_t repeated{0};
};

struct Probability {
  TokenId token{0};
  double value{0.0};
};

/// A normalized, post-filter sampling distribution.
class SamplingDistribution {
public:
  SamplingDistribution() = default;
  explicit SamplingDistribution(std::vector<Probability> entries);

  [[nodiscard]] std::span<const Probability> entries() const noexcept;
  [[nodiscard]] TokenId best_token() const;
  [[nodiscard]] double probability(TokenId token) const noexcept;
  [[nodiscard]] TokenId Sample(std::uint64_t* rng_state) const;
  [[nodiscard]] TokenId SampleResidual(
      std::span<const TokenId> candidate_ids,
      std::span<const float> candidate_probabilities,
      std::uint64_t* rng_state) const;

private:
  friend class SamplerState;
  // Entries generated internally have unique IDs; retain their stable order.
  SamplingDistribution(std::vector<Probability> entries, double total);
  std::vector<Probability> entries_;
};

/// Builds the exact distribution from prompt and request-generated tokens.
[[nodiscard]] SamplingDistribution BuildDistribution(
    std::span<const float> logits, const SamplingConfig& config,
    std::span<const TokenId> prompt_tokens = {},
    std::span<const TokenId> generated_tokens = {});

/// Request-owned controls, repetition window, full generated counts and RNG.
///
/// Call Accept only for committed tokens. The type is copyable so speculative
/// decode can work on a tentative state and discard it on rollback.
class SamplerState {
public:
  explicit SamplerState(SamplingConfig config = {},
                        std::span<const TokenId> initial_history = {},
                        std::shared_ptr<const TokenConstraint> constraint = {});
  ~SamplerState() = default;
  SamplerState(const SamplerState& other);
  SamplerState& operator=(const SamplerState& other);
  SamplerState(SamplerState&&) noexcept = default;
  SamplerState& operator=(SamplerState&&) noexcept = default;

  [[nodiscard]] const SamplingConfig& config() const noexcept;
  [[nodiscard]] std::span<const TokenId> history() const noexcept;
  [[nodiscard]] std::span<const TokenPenalty> penalties() const noexcept {
    return penalty_counts_;
  }
  [[nodiscard]] std::uint64_t rng_state() const noexcept;
  [[nodiscard]] std::uint64_t* mutable_rng_state() noexcept;
  void SetRngState(std::uint64_t state) noexcept;
  /// Publish RNG and any deferred draw without accepting tentative history.
  void CopyDrawStateFrom(const SamplerState& other) noexcept;
  /// Attaching a grammar starts a new cursor and discards a deferred draw.
  void SetConstraint(std::shared_ptr<const TokenConstraint> constraint);
  [[nodiscard]] bool has_constraint() const noexcept;
  [[nodiscard]] bool constraint_complete() const;
  struct DrawState {
    std::uint64_t rng;
    std::optional<TokenId> pending;
  };
  [[nodiscard]] DrawState SaveDrawState() const noexcept {
    return {rng_state_, pending_sample_};
  }
  void RestoreDrawState(DrawState state) noexcept {
    rng_state_ = state.rng;
    pending_sample_ = state.pending;
  }

  /// Starts a request: tokens initialize repetition only; generated counts
  /// reset.
  void ResetHistory(std::span<const TokenId> tokens);
  void Accept(TokenId token);
  void Accept(std::span<const TokenId> tokens);

  [[nodiscard]] SamplingDistribution Distribution(
      std::span<const float> logits) const;
  /// Compact logits use token_ids for penalties; returned IDs index logits.
  [[nodiscard]] SamplingDistribution Distribution(
      std::span<const float> logits, std::span<const TokenId> token_ids) const;
  /// Retain a speculative residual draw for the next Sample call. Copies
  /// preserve this draw along with the RNG; resetting history discards it.
  void DeferSample(TokenId token);
  [[nodiscard]] TokenId Sample(std::span<const float> logits);
  [[nodiscard]] TokenId SampleResidual(
      std::span<const float> target_logits,
      std::span<const TokenId> candidate_ids,
      std::span<const float> candidate_probabilities);
  [[nodiscard]] double Uniform();

private:
  void TrimHistory();
  void RebuildPenaltyCounts();
  [[nodiscard]] double AdjustedLogit(TokenId token, float logit) const noexcept;
  [[nodiscard]] TokenId SampleGreedy(std::span<const float> logits) const;
  [[nodiscard]] TokenId SampleConstrainedGreedy(
      std::span<const float> logits,
      std::span<const TokenId> token_ids = {}) const;
  [[nodiscard]] SamplingDistribution LinearDistribution(
      std::span<const float> logits) const;
  void PrepareSelected(std::span<const float> logits);

  SamplingConfig config_;
  std::vector<TokenId> history_;
  std::vector<TokenPenalty> penalty_counts_;
  std::vector<Probability> candidate_scratch_;
  std::uint64_t rng_state_{0};
  std::optional<TokenId> pending_sample_;
  std::shared_ptr<const TokenConstraint> constraint_;
  std::unique_ptr<TokenConstraint::Cursor> constraint_cursor_;
};

[[nodiscard]] std::uint64_t NextRandom(std::uint64_t* state);
[[nodiscard]] double Uniform(std::uint64_t* state);

// Compatibility helpers for callers that have not yet adopted SamplerState.
[[nodiscard]] TokenId SampleLogits(std::span<const float> logits,
                                   float temperature, std::uint64_t* rng_state);
[[nodiscard]] double TokenProbability(std::span<const float> logits,
                                      float temperature, TokenId token);
[[nodiscard]] TokenId SampleResidual(
    std::span<const float> target_logits, float temperature,
    std::span<const TokenId> candidate_ids,
    std::span<const float> candidate_probabilities, std::uint64_t* rng_state);

}  // namespace gufo::sampling

#endif  // GUFO_CORE_SAMPLING_HPP_
