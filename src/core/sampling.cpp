#include "src/core/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace gufo::sampling {
namespace {

struct Candidate {
  TokenId token;
  double logit;
};

[[nodiscard]] bool IsBetterCandidate(const Candidate& left,
                                     const Candidate& right) noexcept {
  if (left.logit == right.logit) {
    return left.token < right.token;
  }
  return left.logit > right.logit;
}

[[nodiscard]] bool IsBetterProbability(const Probability& left,
                                       const Probability& right) noexcept {
  if (left.value == right.value) {
    return left.token < right.token;
  }
  return left.value > right.value;
}

[[nodiscard]] std::size_t MinimumKept(const SamplingConfig& config,
                                      std::size_t size) {
  return std::min(size, std::max<std::size_t>(config.min_keep, 1));
}

void SortCandidates(std::vector<Candidate>* candidates) {
  std::ranges::sort(*candidates, IsBetterCandidate);
}

[[nodiscard]] std::vector<double> SoftmaxWeights(
    std::span<const Candidate> candidates, double temperature) {
  if (candidates.empty()) {
    throw std::runtime_error("sampling filters removed every token");
  }
  const double maximum = candidates.front().logit;
  std::vector<double> weights;
  weights.reserve(candidates.size());
  double sum = 0.0;
  for (const auto& candidate : candidates) {
    const double weight = std::exp((candidate.logit - maximum) / temperature);
    weights.push_back(weight);
    sum += weight;
  }
  if (!(sum > 0.0) || !std::isfinite(sum)) {
    throw std::runtime_error("logit softmax normalization failed");
  }
  for (double& weight : weights) {
    weight /= sum;
  }
  return weights;
}

double Penalize(double logit, const SamplingConfig& config,
                const TokenPenalty& penalty) {
  if (penalty.repeated && config.repeat_penalty != 1.0F) {
    logit = logit <= 0 ? logit * config.repeat_penalty
                       : logit / config.repeat_penalty;
  }
  logit -=
      static_cast<double>(config.frequency_penalty) * penalty.generated_count;
  if (penalty.generated_count != 0)
    logit -= config.presence_penalty;
  return logit;
}

void ApplyPenalties(std::vector<Candidate>* candidates,
                    const SamplingConfig& config,
                    std::span<const TokenPenalty> penalties) {
  if (penalties.empty())
    return;
  for (auto& candidate : *candidates) {
    const auto found = std::ranges::lower_bound(penalties, candidate.token, {},
                                                &TokenPenalty::token);
    if (found != penalties.end() && found->token == candidate.token)
      candidate.logit = Penalize(candidate.logit, config, *found);
  }
}

void ApplyTopK(std::vector<Candidate>* candidates,
               const SamplingConfig& config) {
  if (config.top_k <= 0) {
    SortCandidates(candidates);
    return;
  }
  const auto configured = static_cast<std::size_t>(config.top_k);
  const std::size_t keep =
      std::min(candidates->size(),
               std::max(configured, MinimumKept(config, candidates->size())));
  if (keep < candidates->size()) {
    std::partial_sort(candidates->begin(),
                      candidates->begin() + static_cast<std::ptrdiff_t>(keep),
                      candidates->end(), IsBetterCandidate);
  } else {
    SortCandidates(candidates);
  }
  candidates->resize(keep);
}

void ApplyTopP(std::vector<Candidate>* candidates,
               const SamplingConfig& config) {
  if (config.top_p >= 1.0F || candidates->size() <= 1) {
    return;
  }
  const auto probabilities =
      SoftmaxWeights(*candidates, static_cast<double>(config.temperature));
  const std::size_t minimum = MinimumKept(config, candidates->size());
  double cumulative = 0.0;
  std::size_t keep = 0;
  while (keep < candidates->size()) {
    cumulative += probabilities[keep];
    ++keep;
    if (keep >= minimum && cumulative >= static_cast<double>(config.top_p)) {
      break;
    }
  }
  candidates->resize(keep);
}

void ApplyMinP(std::vector<Candidate>* candidates,
               const SamplingConfig& config) {
  if (config.min_p <= 0.0F || candidates->size() <= 1) {
    return;
  }
  const double threshold = static_cast<double>(config.temperature) *
                           std::log(static_cast<double>(config.min_p));
  const double maximum = candidates->front().logit;
  std::size_t keep = 0;
  while (keep < candidates->size() &&
         candidates->at(keep).logit - maximum >= threshold) {
    ++keep;
  }
  keep = std::max(keep, MinimumKept(config, candidates->size()));
  candidates->resize(std::min(keep, candidates->size()));
}

[[nodiscard]] std::uint64_t InitialRngState(std::int64_t seed) {
  if (seed >= 0) {
    return static_cast<std::uint64_t>(seed);
  }
  std::random_device random_device;
  return (static_cast<std::uint64_t>(random_device()) << 32U) ^
         static_cast<std::uint64_t>(random_device());
}

}  // namespace

TokenConstraint::TokenConstraint(
    std::shared_ptr<const JsonConstraint> prototype,
    std::shared_ptr<const Vocabulary> vocabulary)
    : TokenConstraint(std::move(prototype), std::move(vocabulary), {}) {}

TokenConstraint::TokenConstraint(
    std::shared_ptr<const JsonConstraint> prototype,
    std::shared_ptr<const Vocabulary> vocabulary, Options options)
    : prototype_(std::move(prototype)),
      vocabulary_(std::move(vocabulary)),
      options_(options) {
  if (!prototype_ || !vocabulary_ || vocabulary_->empty())
    throw std::invalid_argument(
        "JSON constraint requires a grammar and vocabulary");
  if (options_.require_tool_call && !options_.allow_tool_calls)
    throw std::invalid_argument("required tool calls must be enabled");
}

TokenConstraint::Cursor TokenConstraint::Start() const {
  return {*prototype_,
          options_.starts_in_reasoning
              ? Cursor::Phase::kReasoning
              : (options_.allow_tool_calls ? Cursor::Phase::kAnswerStart
                                           : Cursor::Phase::kJson),
          {}};
}

bool TokenConstraint::Complete(const Cursor& cursor) const {
  return cursor.phase == Cursor::Phase::kAfterTool ||
         (cursor.phase == Cursor::Phase::kJson && cursor.grammar.Complete());
}

bool TokenConstraint::Allows(const Cursor& cursor, TokenId token) const {
  if (token >= vocabulary_->size())
    return false;
  auto trial = cursor;
  try {
    Accept(trial, token);
    return true;
  } catch (const std::invalid_argument&) {
    return false;
  }
}

void TokenConstraint::Accept(Cursor& cursor, TokenId token) const {
  if (token >= vocabulary_->size())
    throw std::invalid_argument("constrained token exceeds vocabulary");
  const auto& piece = (*vocabulary_)[token];
  if (piece.stop) {
    if (!Complete(cursor))
      throw std::invalid_argument(
          "JSON constraint cannot stop before completion");
    return;
  }
  using Phase = Cursor::Phase;
  const bool transition_special =
      (cursor.phase == Phase::kReasoning && piece.bytes == "</think>") ||
      ((cursor.phase == Phase::kAnswerStart ||
        cursor.phase == Phase::kAfterTool) &&
       options_.allow_tool_calls && piece.bytes == "<tool_call>") ||
      (cursor.phase == Phase::kToolBody &&
       (piece.bytes == "</tool_call>" || piece.bytes == "<function=" ||
        piece.bytes == "</function>" || piece.bytes == "<parameter=" ||
        piece.bytes == "</parameter>"));
  if (piece.bytes.empty() || (piece.special && !transition_special))
    throw std::invalid_argument(
        "JSON constraint rejects special or empty token");
  Advance(cursor, piece.bytes);
}

void TokenConstraint::Advance(Cursor& cursor, std::string_view bytes) const {
  using Phase = Cursor::Phase;
  constexpr std::string_view thought_end = "</think>";
  constexpr std::string_view tool_start = "<tool_call>";
  const auto whitespace = [](char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
  };
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    const char byte = bytes[index];
    switch (cursor.phase) {
      case Phase::kReasoning:
        cursor.pending.push_back(byte);
        while (!thought_end.starts_with(cursor.pending))
          cursor.pending.erase(0, 1);
        if (cursor.pending == thought_end) {
          cursor.phase =
              options_.allow_tool_calls ? Phase::kAnswerStart : Phase::kJson;
          cursor.pending.clear();
        }
        break;
      case Phase::kToolBody:
        AdvanceToolBody(cursor, byte);
        break;
      case Phase::kAnswerStart:
      case Phase::kAfterTool:
        if (whitespace(byte)) {
          if (cursor.pending.size() == 8)
            throw std::invalid_argument(
                "structured output has excessive boundary whitespace");
          cursor.pending.push_back(byte);
        } else if (options_.allow_tool_calls && byte == '<') {
          cursor.phase = Phase::kToolStart;
          cursor.pending = "<";
        } else {
          if (cursor.phase == Phase::kAfterTool || options_.require_tool_call)
            throw std::invalid_argument(
                "structured output requires a tool call");
          cursor.phase = Phase::kJson;
          cursor.grammar.Accept(cursor.pending +
                                std::string(bytes.substr(index)));
          cursor.pending.clear();
          return;
        }
        break;
      case Phase::kToolStart:
        cursor.pending.push_back(byte);
        if (!tool_start.starts_with(cursor.pending))
          throw std::invalid_argument(
              "structured output has an invalid tool marker");
        if (cursor.pending == tool_start) {
          cursor.pending.clear();
          cursor.phase = Phase::kToolBody;
          cursor.tool_phase = Cursor::ToolPhase::kStart;
          cursor.json_closers.clear();
          cursor.json_string = false;
          cursor.json_escape = false;
        }
        break;
      case Phase::kJson:
        cursor.grammar.Accept(bytes.substr(index));
        return;
    }
  }
}

// Framing only: the transport validates tool names, argument types and JSON.
// Delimiters inside JSON strings or XML parameters remain payload bytes.
void TokenConstraint::AdvanceToolBody(Cursor& cursor, char byte) const {
  using ToolPhase = Cursor::ToolPhase;
  constexpr std::string_view function_start = "<function=";
  constexpr std::string_view function_end = "</function>";
  constexpr std::string_view parameter_start = "<parameter=";
  constexpr std::string_view parameter_end = "</parameter>";
  constexpr std::string_view tool_end = "</tool_call>";
  const bool whitespace =
      byte == ' ' || byte == '\t' || byte == '\r' || byte == '\n';
  switch (cursor.tool_phase) {
    case ToolPhase::kStart:
      if (whitespace)
        break;
      if (byte == '{') {
        cursor.json_closers.push_back('}');
        cursor.tool_phase = ToolPhase::kJson;
      } else if (byte == '<') {
        cursor.pending = "<";
        cursor.tool_phase = ToolPhase::kFunctionPrefix;
      } else {
        throw std::invalid_argument(
            "tool call requires a JSON object or function tag");
      }
      break;
    case ToolPhase::kJson:
      if (cursor.json_string) {
        if (cursor.json_escape)
          cursor.json_escape = false;
        else if (byte == '\\')
          cursor.json_escape = true;
        else if (byte == '"')
          cursor.json_string = false;
      } else if (byte == '"') {
        cursor.json_string = true;
      } else if (byte == '{' || byte == '[') {
        cursor.json_closers.push_back(byte == '{' ? '}' : ']');
      } else if (byte == '}' || byte == ']') {
        if (cursor.json_closers.empty() || cursor.json_closers.back() != byte)
          throw std::invalid_argument("tool JSON has mismatched delimiters");
        cursor.json_closers.pop_back();
        if (cursor.json_closers.empty())
          cursor.tool_phase = ToolPhase::kClose;
      }
      break;
    case ToolPhase::kFunctionPrefix:
      cursor.pending.push_back(byte);
      if (!function_start.starts_with(cursor.pending))
        throw std::invalid_argument("invalid tool function prefix");
      if (cursor.pending == function_start) {
        cursor.pending.clear();
        cursor.tool_phase = ToolPhase::kFunctionName;
      }
      break;
    case ToolPhase::kFunctionName:
    case ToolPhase::kParameterName:
      if (byte == '>') {
        if (cursor.pending.empty())
          throw std::invalid_argument("empty tool function or parameter name");
        cursor.pending.clear();
        cursor.tool_phase = cursor.tool_phase == ToolPhase::kFunctionName
                                ? ToolPhase::kBetweenParameters
                                : ToolPhase::kParameterBody;
      } else {
        // Only presence matters here; final parsing validates the full name.
        cursor.pending = "x";
      }
      break;
    case ToolPhase::kBetweenParameters:
      if (cursor.pending.empty() && whitespace)
        break;
      cursor.pending.push_back(byte);
      if (cursor.pending == function_end) {
        cursor.pending.clear();
        cursor.tool_phase = ToolPhase::kClose;
      } else if (cursor.pending == parameter_start) {
        cursor.pending.clear();
        cursor.tool_phase = ToolPhase::kParameterName;
      } else if (!function_end.starts_with(cursor.pending) &&
                 !parameter_start.starts_with(cursor.pending)) {
        throw std::invalid_argument(
            "invalid tool parameter or function delimiter");
      }
      break;
    case ToolPhase::kParameterBody:
      cursor.pending.push_back(byte);
      while (!parameter_end.starts_with(cursor.pending))
        cursor.pending.erase(0, 1);
      if (cursor.pending == parameter_end) {
        cursor.pending.clear();
        cursor.tool_phase = ToolPhase::kBetweenParameters;
      }
      break;
    case ToolPhase::kClose:
      if (cursor.pending.empty() && whitespace)
        break;
      cursor.pending.push_back(byte);
      if (!tool_end.starts_with(cursor.pending))
        throw std::invalid_argument("tool call requires its closing delimiter");
      if (cursor.pending == tool_end) {
        cursor.pending.clear();
        cursor.phase = Cursor::Phase::kAfterTool;
      }
      break;
  }
}

void SamplingConfig::Validate() const {
  if (!std::isfinite(temperature) || temperature < 0.0F) {
    throw std::invalid_argument(
        "sampling temperature must be finite and nonnegative");
  }
  if (top_k < 0) {
    throw std::invalid_argument("sampling top-k must be nonnegative");
  }
  if (!std::isfinite(top_p) || top_p <= 0.0F || top_p > 1.0F) {
    throw std::invalid_argument("sampling top-p must be in (0, 1]");
  }
  if (!std::isfinite(min_p) || min_p < 0.0F || min_p > 1.0F) {
    throw std::invalid_argument("sampling min-p must be in [0, 1]");
  }
  if (seed < -1) {
    throw std::invalid_argument("sampling seed must be -1 or nonnegative");
  }
  if (!std::isfinite(repeat_penalty) || repeat_penalty <= 0.0F) {
    throw std::invalid_argument(
        "sampling repeat penalty must be finite and positive");
  }
  if (!std::isfinite(frequency_penalty)) {
    throw std::invalid_argument("sampling frequency penalty must be finite");
  }
  if (!std::isfinite(presence_penalty)) {
    throw std::invalid_argument("sampling presence penalty must be finite");
  }
}

bool SamplingConfig::penalties_enabled() const noexcept {
  return repeat_penalty != 1.0F || frequency_penalty != 0.0F ||
         presence_penalty != 0.0F;
}

bool SamplingConfig::uses_random_sampling() const noexcept {
  return temperature > 0.0F && (top_k != 1 || min_keep > 1);
}

bool SamplingConfig::can_use_unmodified_argmax() const noexcept {
  return temperature == 0.0F && !penalties_enabled();
}

SamplingDistribution::SamplingDistribution(std::vector<Probability> entries)
    : entries_(std::move(entries)) {
  if (entries_.empty()) {
    throw std::invalid_argument("sampling distribution cannot be empty");
  }
  std::unordered_set<TokenId> tokens;
  double sum = 0.0;
  for (const auto& entry : entries_) {
    if (!std::isfinite(entry.value) || entry.value < 0.0 ||
        !tokens.insert(entry.token).second) {
      throw std::invalid_argument("sampling distribution is malformed");
    }
    sum += entry.value;
  }
  if (!(sum > 0.0) || !std::isfinite(sum)) {
    throw std::invalid_argument("sampling distribution has no probability");
  }
  for (auto& entry : entries_) {
    entry.value /= sum;
  }
  std::erase_if(entries_, [](const auto& entry) { return entry.value == 0; });
  std::ranges::sort(entries_,
                    [](const Probability& left, const Probability& right) {
                      if (left.value == right.value) {
                        return left.token < right.token;
                      }
                      return left.value > right.value;
                    });
}

SamplingDistribution::SamplingDistribution(std::vector<Probability> entries,
                                           double total)
    : entries_(std::move(entries)) {
  if (!(total > 0) || !std::isfinite(total))
    throw std::runtime_error("logit softmax normalization failed");
  for (auto& entry : entries_)
    entry.value /= total;
  std::erase_if(entries_, [](const auto& entry) { return entry.value == 0; });
}

std::span<const Probability> SamplingDistribution::entries() const noexcept {
  return entries_;
}

TokenId SamplingDistribution::best_token() const {
  if (entries_.empty()) {
    throw std::logic_error("sampling distribution is empty");
  }
  return std::ranges::max_element(entries_,
                                  [](const auto& a, const auto& b) {
                                    return IsBetterProbability(b, a);
                                  })
      ->token;
}

double SamplingDistribution::probability(TokenId token) const noexcept {
  for (const auto& entry : entries_) {
    if (entry.token == token) {
      return entry.value;
    }
  }
  return 0.0;
}

TokenId SamplingDistribution::Sample(std::uint64_t* rng_state) const {
  if (entries_.empty()) {
    throw std::logic_error("sampling distribution is empty");
  }
  if (entries_.size() == 1) {
    return entries_.front().token;
  }
  double sample = gufo::sampling::Uniform(rng_state);
  for (const auto& entry : entries_) {
    sample -= entry.value;
    if (sample < 0.0) {
      return entry.token;
    }
  }
  return entries_.back().token;
}

TokenId SamplingDistribution::SampleResidual(
    std::span<const TokenId> candidate_ids,
    std::span<const float> candidate_probabilities,
    std::uint64_t* rng_state) const {
  if (candidate_ids.size() != candidate_probabilities.size()) {
    throw std::invalid_argument("sparse probability row is malformed");
  }
  std::unordered_map<TokenId, double> draft;
  for (std::size_t index = 0; index < candidate_ids.size(); ++index) {
    const float probability_value = candidate_probabilities[index];
    if (!std::isfinite(probability_value) || probability_value < 0.0F) {
      throw std::invalid_argument(
          "sparse probability row contains an invalid value");
    }
    draft[candidate_ids[index]] += static_cast<double>(probability_value);
  }

  std::vector<Probability> residual;
  residual.reserve(entries_.size());
  double sum = 0.0;
  for (const auto& entry : entries_) {
    const auto found = draft.find(entry.token);
    const double q = found == draft.end() ? 0.0 : found->second;
    const double value = std::max(entry.value - q, 0.0);
    if (value > 0.0) {
      residual.push_back({.token = entry.token, .value = value});
      sum += value;
    }
  }
  if (!(sum > 0.0) || !std::isfinite(sum)) {
    return Sample(rng_state);
  }
  return SamplingDistribution(std::move(residual), sum).Sample(rng_state);
}

static SamplingDistribution DistributionWithPenalties(
    std::span<const float> logits, const SamplingConfig& config,
    std::span<const TokenPenalty> penalties) {
  config.Validate();
  if (logits.empty()) {
    throw std::invalid_argument("cannot sample an empty logit distribution");
  }
  if (logits.size() >
      static_cast<std::size_t>(std::numeric_limits<TokenId>::max())) {
    throw std::invalid_argument("logit distribution exceeds token ID range");
  }

  std::vector<Candidate> candidates;
  candidates.reserve(logits.size());
  for (std::size_t index = 0; index < logits.size(); ++index) {
    if (std::isfinite(logits[index])) {
      candidates.push_back({
          .token = static_cast<TokenId>(index),
          .logit = static_cast<double>(logits[index]),
      });
    }
  }
  if (candidates.empty()) {
    throw std::runtime_error("logit distribution contains no finite values");
  }

  ApplyPenalties(&candidates, config, penalties);
  ApplyTopK(&candidates, config);
  if (config.temperature == 0.0F) {
    return SamplingDistribution(
        {{.token = candidates.front().token, .value = 1.0}});
  }
  ApplyTopP(&candidates, config);
  ApplyMinP(&candidates, config);
  const auto probabilities =
      SoftmaxWeights(candidates, static_cast<double>(config.temperature));
  std::vector<Probability> entries;
  entries.reserve(candidates.size());
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    entries.push_back(
        {.token = candidates[index].token, .value = probabilities[index]});
  }
  return SamplingDistribution(std::move(entries));
}

SamplingDistribution BuildDistribution(
    std::span<const float> logits, const SamplingConfig& config,
    std::span<const TokenId> prompt_tokens,
    std::span<const TokenId> generated_tokens) {
  std::unordered_map<TokenId, TokenPenalty> counts;
  if (config.frequency_penalty != 0 || config.presence_penalty != 0) {
    for (const auto token : generated_tokens) {
      auto& p = counts[token];
      p.token = token;
      ++p.generated_count;
    }
  }
  if (config.repeat_penalty != 1 && config.repeat_last_n != 0) {
    const auto tail = generated_tokens.last(
        std::min(generated_tokens.size(), config.repeat_last_n));
    const auto prompt_tail = prompt_tokens.last(
        std::min(prompt_tokens.size(), config.repeat_last_n - tail.size()));
    for (auto sequence : {prompt_tail, tail}) {
      for (const auto token : sequence) {
        auto& p = counts[token];
        p.token = token;
        p.repeated = 1;
      }
    }
  }
  std::vector<TokenPenalty> penalties;
  for (const auto& [token, penalty] : counts)
    penalties.push_back(penalty);
  std::ranges::sort(penalties, {}, &TokenPenalty::token);
  return DistributionWithPenalties(logits, config, penalties);
}

SamplerState::SamplerState(SamplingConfig config,
                           std::span<const TokenId> initial_history,
                           std::shared_ptr<const TokenConstraint> constraint)
    : config_(config),
      history_(initial_history.begin(), initial_history.end()),
      rng_state_(InitialRngState(config.seed)) {
  config_.Validate();
  SetConstraint(std::move(constraint));
  TrimHistory();
  RebuildPenaltyCounts();
}

SamplerState::SamplerState(const SamplerState& other)
    : config_(other.config_),
      history_(other.history_),
      penalty_counts_(other.penalty_counts_),
      rng_state_(other.rng_state_),
      pending_sample_(other.pending_sample_),
      constraint_(other.constraint_),
      constraint_cursor_(other.constraint_cursor_
                             ? std::make_unique<TokenConstraint::Cursor>(
                                   *other.constraint_cursor_)
                             : nullptr) {}

SamplerState& SamplerState::operator=(const SamplerState& other) {
  if (this == &other) {
    return *this;
  }
  auto copied = SamplerState(other);
  *this = std::move(copied);
  return *this;
}

const SamplingConfig& SamplerState::config() const noexcept {
  return config_;
}

std::span<const TokenId> SamplerState::history() const noexcept {
  return history_;
}

std::uint64_t SamplerState::rng_state() const noexcept {
  return rng_state_;
}

std::uint64_t* SamplerState::mutable_rng_state() noexcept {
  return &rng_state_;
}

void SamplerState::SetRngState(std::uint64_t state) noexcept {
  rng_state_ = state;
  pending_sample_.reset();
}

void SamplerState::CopyDrawStateFrom(const SamplerState& other) noexcept {
  rng_state_ = other.rng_state_;
  pending_sample_ = other.pending_sample_;
}

void SamplerState::SetConstraint(
    std::shared_ptr<const TokenConstraint> constraint) {
  if (constraint && config_.temperature != 0)
    throw std::invalid_argument(
        "JSON constrained decoding requires temperature 0");
  auto cursor =
      constraint
          ? std::make_unique<TokenConstraint::Cursor>(constraint->Start())
          : nullptr;
  constraint_ = std::move(constraint);
  constraint_cursor_ = std::move(cursor);
  pending_sample_.reset();
}

bool SamplerState::has_constraint() const noexcept {
  return constraint_ != nullptr;
}

bool SamplerState::constraint_complete() const {
  return constraint_cursor_ && constraint_->Complete(*constraint_cursor_);
}

void SamplerState::ResetHistory(std::span<const TokenId> tokens) {
  if (constraint_)
    constraint_cursor_ =
        std::make_unique<TokenConstraint::Cursor>(constraint_->Start());
  pending_sample_.reset();
  penalty_counts_.clear();
  history_.assign(tokens.begin(), tokens.end());
  TrimHistory();
  RebuildPenaltyCounts();
}

void SamplerState::Accept(TokenId token) {
  Accept(std::span<const TokenId>(&token, 1));
}

void SamplerState::Accept(std::span<const TokenId> tokens) {
  std::unique_ptr<TokenConstraint::Cursor> advanced;
  if (constraint_) {
    advanced = std::make_unique<TokenConstraint::Cursor>(*constraint_cursor_);
    for (const auto token : tokens)
      constraint_->Accept(*advanced, token);
  }
  if (config_.frequency_penalty != 0 || config_.presence_penalty != 0) {
    for (const auto token : tokens) {
      auto found = std::ranges::lower_bound(penalty_counts_, token, {},
                                            &TokenPenalty::token);
      if (found == penalty_counts_.end() || found->token != token)
        found = penalty_counts_.insert(found, TokenPenalty{.token = token});
      if (found->generated_count == std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("generated token count overflow");
      ++found->generated_count;
    }
  }
  if (config_.repeat_last_n != 0) {
    history_.insert(history_.end(), tokens.begin(), tokens.end());
    TrimHistory();
  }
  RebuildPenaltyCounts();
  if (advanced)
    constraint_cursor_ = std::move(advanced);
}

SamplingDistribution SamplerState::Distribution(
    std::span<const float> logits) const {
  config_.Validate();
  if (logits.empty() || logits.size() > std::numeric_limits<TokenId>::max())
    throw std::invalid_argument("invalid sampling vocabulary size");
  if (config_.temperature == 0)
    return SamplingDistribution({{SampleGreedy(logits), 1.0}}, 1.0);
  const bool needs_floor = config_.min_p > 0 && config_.min_keep > 1;
  if (config_.top_k == 0 && config_.top_p == 1 && !needs_floor)
    return LinearDistribution(logits);
  auto working = *this;
  working.PrepareSelected(logits);
  auto& candidates = working.candidate_scratch_;
  const double maximum = candidates.front().value;
  double total = 0;
  for (auto& candidate : candidates) {
    candidate.value =
        std::exp((candidate.value - maximum) / config_.temperature);
    total += candidate.value;
  }
  return SamplingDistribution(std::move(candidates), total);
}

SamplingDistribution SamplerState::Distribution(
    std::span<const float> logits, std::span<const TokenId> token_ids) const {
  if (logits.size() != token_ids.size())
    throw std::invalid_argument("compact logits and token IDs differ in size");
  if (constraint_)
    return SamplingDistribution(
        {{SampleConstrainedGreedy(logits, token_ids), 1.0}}, 1.0);
  std::vector<TokenPenalty> penalties;
  penalties.reserve(token_ids.size());
  for (std::size_t i = 0; i < token_ids.size(); ++i) {
    const auto found = std::ranges::lower_bound(penalty_counts_, token_ids[i],
                                                {}, &TokenPenalty::token);
    if (found != penalty_counts_.end() && found->token == token_ids[i]) {
      auto penalty = *found;
      penalty.token = static_cast<TokenId>(i);
      penalties.push_back(penalty);
    }
  }
  return DistributionWithPenalties(logits, config_, penalties);
}

void SamplerState::DeferSample(TokenId token) {
  if (pending_sample_) {
    throw std::logic_error("a sampled token is already pending");
  }
  pending_sample_ = token;
}

TokenId SamplerState::Sample(std::span<const float> logits) {
  config_.Validate();
  if (logits.empty()) {
    throw std::invalid_argument("cannot sample an empty logit distribution");
  }
  if (logits.size() >
      static_cast<std::size_t>(std::numeric_limits<TokenId>::max())) {
    throw std::invalid_argument("logit distribution exceeds token ID range");
  }
  if (pending_sample_) {
    if (*pending_sample_ >= logits.size()) {
      throw std::invalid_argument("pending sample exceeds vocabulary");
    }
    const auto token = *pending_sample_;
    if (constraint_ && (!std::isfinite(logits[token]) ||
                        !constraint_->Allows(*constraint_cursor_, token))) {
      throw std::invalid_argument("pending sample violates JSON constraint");
    }
    pending_sample_.reset();
    return token;
  }
  if (config_.temperature == 0.0F) {
    return SampleGreedy(logits);
  }
  return Distribution(logits).Sample(&rng_state_);
}

TokenId SamplerState::SampleResidual(
    std::span<const float> target_logits,
    std::span<const TokenId> candidate_ids,
    std::span<const float> candidate_probabilities) {
  return Distribution(target_logits)
      .SampleResidual(candidate_ids, candidate_probabilities, &rng_state_);
}

double SamplerState::Uniform() {
  return gufo::sampling::Uniform(&rng_state_);
}

void SamplerState::TrimHistory() {
  if (config_.repeat_last_n == 0) {
    history_.clear();
    return;
  }
  if (history_.size() > config_.repeat_last_n) {
    history_.erase(
        history_.begin(),
        history_.end() - static_cast<std::ptrdiff_t>(config_.repeat_last_n));
  }
}

void SamplerState::RebuildPenaltyCounts() {
  if (!config_.penalties_enabled())
    return;
  for (auto& penalty : penalty_counts_)
    penalty.repeated = 0;
  if (config_.repeat_penalty != 1) {
    for (const auto token : history_) {
      auto found = std::ranges::lower_bound(penalty_counts_, token, {},
                                            &TokenPenalty::token);
      if (found == penalty_counts_.end() || found->token != token)
        found = penalty_counts_.insert(found, TokenPenalty{.token = token});
      found->repeated = 1;
    }
  }
  std::erase_if(penalty_counts_, [](const auto& p) {
    return p.generated_count == 0 && p.repeated == 0;
  });
}

double SamplerState::AdjustedLogit(TokenId token, float logit) const noexcept {
  const auto found = std::ranges::lower_bound(penalty_counts_, token, {},
                                              &TokenPenalty::token);
  return found == penalty_counts_.end() || found->token != token
             ? static_cast<double>(logit)
             : Penalize(logit, config_, *found);
}

TokenId SamplerState::SampleConstrainedGreedy(
    std::span<const float> logits, std::span<const TokenId> token_ids) const {
  if (logits.empty() || logits.size() > std::numeric_limits<TokenId>::max())
    throw std::invalid_argument("invalid sampling vocabulary size");
  const auto token_at = [&](TokenId index) {
    return token_ids.empty() ? index : token_ids[index];
  };
  std::vector<Candidate> candidates;
  candidates.reserve(logits.size());
  for (std::size_t index = 0; index < logits.size(); ++index) {
    if (!std::isfinite(logits[index]))
      continue;
    const auto id = static_cast<TokenId>(index);
    const double adjusted = AdjustedLogit(token_at(id), logits[index]);
    if (!std::isfinite(adjusted))
      throw std::runtime_error(
          "sampling penalties produced a non-finite logit");
    candidates.push_back({id, adjusted});
  }
  if (candidates.empty())
    throw std::runtime_error("logit distribution contains no finite values");
  const auto worse = [&](const Candidate& left, const Candidate& right) {
    return left.logit == right.logit
               ? token_at(left.token) > token_at(right.token)
               : left.logit < right.logit;
  };
  // Heapify once; grammar checks only visit candidates ahead of the winner.
  std::make_heap(candidates.begin(), candidates.end(), worse);
  while (!candidates.empty()) {
    const auto index = candidates.front().token;
    if (constraint_->Allows(*constraint_cursor_, token_at(index)))
      return index;
    std::pop_heap(candidates.begin(), candidates.end(), worse);
    candidates.pop_back();
  }
  throw std::runtime_error("JSON constraint has no legal finite token");
}

TokenId SamplerState::SampleGreedy(std::span<const float> logits) const {
  if (constraint_)
    return SampleConstrainedGreedy(logits);
  if (penalty_counts_.empty()) {
    float best_logit = -std::numeric_limits<float>::infinity();
    TokenId best_token = 0;
    bool found = false;
    for (std::size_t index = 0; index < logits.size(); ++index) {
      const float logit = logits[index];
      if (logit > best_logit && std::isfinite(logit)) {
        best_logit = logit;
        best_token = static_cast<TokenId>(index);
        found = true;
      }
    }
    if (!found) {
      throw std::runtime_error("logit distribution contains no finite values");
    }
    return best_token;
  }

  double best_logit = -std::numeric_limits<double>::infinity();
  TokenId best_token = 0;
  bool found = false;
  for (std::size_t index = 0; index < logits.size(); ++index) {
    if (!std::isfinite(logits[index])) {
      continue;
    }
    const double adjusted =
        AdjustedLogit(static_cast<TokenId>(index), logits[index]);
    if (!std::isfinite(adjusted)) {
      throw std::runtime_error(
          "sampling penalties produced a non-finite logit");
    }
    if (!found || adjusted > best_logit) {
      best_logit = adjusted;
      best_token = static_cast<TokenId>(index);
      found = true;
    }
  }
  if (!found) {
    throw std::runtime_error("logit distribution contains no finite values");
  }
  return best_token;
}

SamplingDistribution SamplerState::LinearDistribution(
    std::span<const float> logits) const {
  std::vector<Probability> candidates;
  candidates.reserve(logits.size());
  double maximum = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < logits.size(); ++i) {
    if (!std::isfinite(logits[i]))
      continue;
    const double value =
        penalty_counts_.empty()
            ? static_cast<double>(logits[i])
            : AdjustedLogit(static_cast<TokenId>(i), logits[i]);
    if (!std::isfinite(value))
      throw std::runtime_error(
          "sampling penalties produced a non-finite logit");
    maximum = std::max(maximum, value);
    candidates.push_back({static_cast<TokenId>(i), value});
  }
  if (candidates.empty())
    throw std::runtime_error("logit distribution contains no finite values");
  const double threshold =
      config_.min_p > 0
          ? maximum + static_cast<double>(config_.temperature) *
                          std::log(static_cast<double>(config_.min_p))
          : -std::numeric_limits<double>::infinity();
  double total = 0;
  for (auto& candidate : candidates) {
    candidate.value =
        candidate.value >= threshold
            ? std::exp((candidate.value - maximum) / config_.temperature)
            : 0;
    total += candidate.value;
  }
  return SamplingDistribution(std::move(candidates), total);
}

void SamplerState::PrepareSelected(std::span<const float> logits) {
  candidate_scratch_.clear();
  const auto read_adjusted = [&](std::size_t index) {
    const double adjusted =
        AdjustedLogit(static_cast<TokenId>(index), logits[index]);
    if (!std::isfinite(adjusted)) {
      throw std::runtime_error(
          "sampling penalties produced a non-finite logit");
    }
    return adjusted;
  };
  const auto select_best = [&](std::size_t limit) {
    candidate_scratch_.clear();
    candidate_scratch_.reserve(limit);
    for (std::size_t index = 0; index < logits.size(); ++index) {
      if (!std::isfinite(logits[index])) {
        continue;
      }
      const Probability candidate{
          .token = static_cast<TokenId>(index),
          .value = read_adjusted(index),
      };
      if (candidate_scratch_.size() < limit) {
        candidate_scratch_.push_back(candidate);
        std::push_heap(candidate_scratch_.begin(), candidate_scratch_.end(),
                       IsBetterProbability);
      } else if (IsBetterProbability(candidate, candidate_scratch_.front())) {
        std::pop_heap(candidate_scratch_.begin(), candidate_scratch_.end(),
                      IsBetterProbability);
        candidate_scratch_.back() = candidate;
        std::push_heap(candidate_scratch_.begin(), candidate_scratch_.end(),
                       IsBetterProbability);
      }
    }
    std::ranges::sort(candidate_scratch_, IsBetterProbability);
  };
  const auto select_all = [&]() {
    candidate_scratch_.clear();
    candidate_scratch_.reserve(logits.size());
    for (std::size_t index = 0; index < logits.size(); ++index) {
      if (!std::isfinite(logits[index])) {
        continue;
      }
      candidate_scratch_.push_back({
          .token = static_cast<TokenId>(index),
          .value = read_adjusted(index),
      });
    }
    std::ranges::sort(candidate_scratch_, IsBetterProbability);
  };

  double full_softmax_sum = 0.0;
  bool has_full_softmax_sum = false;
  if (config_.top_k > 0) {
    const std::size_t keep = std::min(
        logits.size(),
        std::max<std::size_t>(static_cast<std::size_t>(config_.top_k),
                              std::max<std::size_t>(config_.min_keep, 1)));
    select_best(keep);
  } else if (config_.top_p < 1.0F && logits.size() > 1024) {
    double maximum = -std::numeric_limits<double>::infinity();
    bool found = false;
    for (std::size_t index = 0; index < logits.size(); ++index) {
      if (!std::isfinite(logits[index])) {
        continue;
      }
      maximum = std::max(maximum, read_adjusted(index));
      found = true;
    }
    if (!found) {
      throw std::runtime_error("logit distribution contains no finite values");
    }
    for (std::size_t index = 0; index < logits.size(); ++index) {
      if (std::isfinite(logits[index])) {
        full_softmax_sum +=
            std::exp((read_adjusted(index) - maximum) / config_.temperature);
      }
    }
    if (!(full_softmax_sum > 0.0) || !std::isfinite(full_softmax_sum)) {
      throw std::runtime_error("logit softmax normalization failed");
    }
    has_full_softmax_sum = true;

    constexpr std::size_t initial_top_p_candidates = 256;
    const std::size_t initial_keep = std::min(
        logits.size(), std::max(initial_top_p_candidates,
                                std::max<std::size_t>(config_.min_keep, 1)));
    select_best(initial_keep);

    double selected_mass = 0.0;
    for (const auto& candidate : candidate_scratch_) {
      selected_mass +=
          std::exp((candidate.value - maximum) / config_.temperature);
    }
    if (selected_mass < static_cast<double>(config_.top_p) * full_softmax_sum) {
      select_all();
    }
  } else {
    select_all();
  }
  if (candidate_scratch_.empty()) {
    throw std::runtime_error("logit distribution contains no finite values");
  }

  std::size_t keep = candidate_scratch_.size();
  const std::size_t minimum = MinimumKept(config_, candidate_scratch_.size());
  if (config_.top_p < 1.0F && keep > 1) {
    const double maximum = candidate_scratch_.front().value;
    double sum = full_softmax_sum;
    if (!has_full_softmax_sum) {
      sum = 0.0;
      for (const auto& candidate : candidate_scratch_) {
        sum += std::exp((candidate.value - maximum) / config_.temperature);
      }
    }
    const double target = static_cast<double>(config_.top_p) * sum;
    double cumulative = 0.0;
    std::size_t top_p_keep = 0;
    while (top_p_keep < keep) {
      cumulative += std::exp((candidate_scratch_[top_p_keep].value - maximum) /
                             config_.temperature);
      ++top_p_keep;
      if (top_p_keep >= minimum && cumulative >= target) {
        break;
      }
    }
    keep = top_p_keep;
  }
  if (config_.min_p > 0.0F && keep > 1) {
    const double threshold = candidate_scratch_.front().value +
                             static_cast<double>(config_.temperature) *
                                 std::log(static_cast<double>(config_.min_p));
    std::size_t min_p_keep = 0;
    while (min_p_keep < keep &&
           candidate_scratch_[min_p_keep].value >= threshold) {
      ++min_p_keep;
    }
    keep = std::min(keep, std::max(min_p_keep, minimum));
  }
  candidate_scratch_.resize(keep);
}

std::uint64_t NextRandom(std::uint64_t* state) {
  if (state == nullptr) {
    throw std::invalid_argument("sampling requires RNG state");
  }
  std::uint64_t value = *state;
  if (value == 0) {
    value = UINT64_C(0x9e3779b97f4a7c15);
  }
  value ^= value >> 12U;
  value ^= value << 25U;
  value ^= value >> 27U;
  *state = value;
  return value * UINT64_C(0x2545f4914f6cdd1d);
}

double Uniform(std::uint64_t* state) {
  const std::uint64_t value = NextRandom(state);
  return static_cast<double>(value >> 11U) * 0x1.0p-53;
}

TokenId SampleLogits(std::span<const float> logits, float temperature,
                     std::uint64_t* rng_state) {
  if (temperature > 0.0F && rng_state == nullptr) {
    throw std::invalid_argument("sampling requires RNG state");
  }
  SamplingConfig config;
  config.temperature = temperature;
  config.seed = 0;
  SamplerState sampler(config);
  if (rng_state != nullptr) {
    sampler.SetRngState(*rng_state);
  }
  const TokenId token = sampler.Sample(logits);
  if (rng_state != nullptr) {
    *rng_state = sampler.rng_state();
  }
  return token;
}

double TokenProbability(std::span<const float> logits, float temperature,
                        TokenId token) {
  if (token >= logits.size()) {
    throw std::out_of_range("sampled token exceeds target vocabulary");
  }
  SamplingConfig config;
  config.temperature = temperature;
  return BuildDistribution(logits, config).probability(token);
}

TokenId SampleResidual(std::span<const float> target_logits, float temperature,
                       std::span<const TokenId> candidate_ids,
                       std::span<const float> candidate_probabilities,
                       std::uint64_t* rng_state) {
  SamplingConfig config;
  config.temperature = temperature;
  return BuildDistribution(target_logits, config)
      .SampleResidual(candidate_ids, candidate_probabilities, rng_state);
}

}  // namespace gufo::sampling
