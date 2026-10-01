#include "src/cli/serve/text_generation_scheduler.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <semaphore>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "src/cli/serve/generation_metrics.hpp"

namespace {

using gufo::server::ChatRequest;
using gufo::server::TextDecodeSelection;
using gufo::server::TextDecodeStep;
using gufo::server::TextExecutionPlan;
using gufo::server::TextExecutionPlanKind;
using gufo::server::TextGenerationError;
using gufo::server::TextGenerationErrorCode;
using gufo::server::TextGenerationScheduler;
using gufo::server::TextModelRunner;
using gufo::server::TextPrefillPolicy;
using gufo::server::TextPrefillStep;
using gufo::server::TextRequestMetadata;
using gufo::server::TextRequestPhase;
using gufo::server::TextRunnerAdvance;
using gufo::server::TextRunnerCapabilities;
using gufo::server::TextRunnerDescriptor;
using gufo::server::TextRunnerPool;
using gufo::server::TextRunnerResourceClaim;
using gufo::server::TextRunnerState;
using gufo::server::TextRunnerToken;
using gufo::server::TextSchedulerPolicy;

constexpr auto kTestTimeout = std::chrono::seconds{5};

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

enum class EventKind : std::uint8_t {
  kPrefill,
  kAdvance,
};

struct Event {
  EventKind kind;
  TextRunnerToken label;
  std::size_t index;
  std::size_t count;
  double start_ms{0};
};

struct FakeClock {
  TextGenerationScheduler::Clock::time_point Now() const {
    return TextGenerationScheduler::Clock::time_point{
        std::chrono::milliseconds{milliseconds.load()}};
  }
  void Advance(std::int64_t ms) { milliseconds.fetch_add(ms); }
  std::atomic<std::int64_t> milliseconds{0};
};

struct FakeControl {
  void Log(Event event) {
    const std::lock_guard<std::mutex> lock(mutex);
    events.push_back(event);
    condition.notify_all();
  }

  void WaitForAdvance(TextRunnerToken label) {
    std::unique_lock<std::mutex> lock(mutex);
    const bool entered = condition.wait_for(lock, kTestTimeout, [&] {
      return advance_gate_entered && advance_gate_label == label;
    });
    Expect(entered, "timed out waiting for blocked decode advance");
  }

  void WaitForPrefill(TextRunnerToken label) {
    std::unique_lock<std::mutex> lock(mutex);
    const bool entered = condition.wait_for(lock, kTestTimeout, [&] {
      return prefill_gate_entered && prefill_gate_label == label;
    });
    Expect(entered, "timed out waiting for blocked prefill");
  }

  void ReleaseAdvance() {
    {
      const std::lock_guard<std::mutex> lock(mutex);
      release_advance = true;
    }
    condition.notify_all();
  }

  void ReleasePrefill() {
    {
      const std::lock_guard<std::mutex> lock(mutex);
      release_prefill = true;
    }
    condition.notify_all();
  }

  void RecordInvalidation() {
    invalidations.fetch_add(1, std::memory_order_relaxed);
    condition.notify_all();
  }

  void WaitForInvalidations(std::size_t count) {
    std::unique_lock<std::mutex> lock(mutex);
    const bool reached = condition.wait_for(lock, kTestTimeout, [&] {
      return invalidations.load(std::memory_order_relaxed) >= count;
    });
    Expect(reached, "timed out waiting for state invalidation");
  }

  [[nodiscard]] std::vector<Event> Events() const {
    const std::lock_guard<std::mutex> lock(mutex);
    return events;
  }

  [[nodiscard]] std::vector<std::vector<TextRunnerToken>> AdvanceBatches()
      const {
    const std::lock_guard<std::mutex> lock(mutex);
    return advance_batches;
  }

  mutable std::mutex mutex;
  std::condition_variable condition;
  std::vector<Event> events;
  std::vector<std::vector<TextRunnerToken>> advance_batches;
  std::optional<TextRunnerToken> block_advance_label;
  std::optional<TextRunnerToken> block_prefill_label;
  std::optional<TextRunnerToken> throw_advance_label;
  TextRunnerToken advance_gate_label{0};
  TextRunnerToken prefill_gate_label{0};
  bool advance_gate_entered{false};
  bool prefill_gate_entered{false};
  bool release_advance{false};
  bool release_prefill{false};
  std::atomic<std::size_t> invalidations{0};
  std::atomic<std::size_t> states_created{0};
  std::atomic<std::size_t> decode_calls{0};
  std::atomic<std::size_t> batch_preparations{0};
  std::atomic<std::int64_t> first_request_advance_ns{0};
  bool incremental_prefill{true};
  std::size_t prefill_capacity{std::numeric_limits<std::size_t>::max()};
  bool supports_batched_advance{false};
  bool final_token_advance_required{true};
  bool incremental_text_is_exact{false};
  bool multi_token_decode{false};
  bool batched_multi_token_decode{false};
  std::size_t actual_batch_width{4};
  bool prefix_reuse{true};
  bool json_constraints{false};
  bool preview_first_token{false};
  std::size_t prefix_checkpoint_alignment{0};
  std::function<void()> snapshot_callback;
  std::shared_ptr<FakeClock> clock;
  std::int64_t prefill_token_ms{0};
  std::int64_t advance_ms{0};
  std::int64_t restore_ms{0};
  std::function<void(TextRunnerToken)> after_advance;
};

class FakeState final : public TextRunnerState {
public:
  explicit FakeState(std::shared_ptr<FakeControl> control)
      : control_(std::move(control)) {}

  void Invalidate() noexcept override {
    control_->RecordInvalidation();
    label = 0;
    position = 0;
    decode_count = 0;
    frontier.reset();
  }

  std::shared_ptr<FakeControl> control_;
  TextRunnerToken label{0};
  std::size_t position{0};
  std::size_t decode_count{0};
  std::optional<TextRunnerToken> frontier;
};

FakeState& RequireFakeState(TextRunnerState& state) {
  auto* fake = dynamic_cast<FakeState*>(&state);
  if (fake == nullptr) {
    throw std::logic_error("unexpected scheduler fake state");
  }
  return *fake;
}

const FakeState& RequireFakeState(const TextRunnerState& state) {
  const auto* fake = dynamic_cast<const FakeState*>(&state);
  if (fake == nullptr) {
    throw std::logic_error("unexpected scheduler fake state");
  }
  return *fake;
}

class FakeRunner final : public TextModelRunner {
public:
  explicit FakeRunner(std::shared_ptr<FakeControl> control)
      : control_(std::move(control)) {}

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = "scheduler-fake",
        .state_abi = "scheduler-fake-v1",
        .max_context = 4096,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = control_->incremental_prefill,
                .snapshot = bool(control_->snapshot_callback),
                .fork = bool(control_->snapshot_callback),
                .final_token_advance_required =
                    control_->final_token_advance_required,
                .incremental_text_is_exact =
                    control_->incremental_text_is_exact,
                .multi_token_decode = control_->multi_token_decode,
                .batched_multi_token_decode =
                    control_->batched_multi_token_decode,
                .batched_multi_token_decode_max_width = 4,
                .prefix_reuse = control_->prefix_reuse,
                .json_constraints = control_->json_constraints,
                .prefix_checkpoint_alignment =
                    control_->prefix_checkpoint_alignment,
            },
        .persistence = std::nullopt,
    };
  }

  void SetPromptContext(
      TextRunnerState&,
      std::shared_ptr<const gufo::server::TextPromptContext> context)
      const override {
    // The fake accepts any context; identity matching happens in the cache.
    (void)context;
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    return {
        .resident_weights_bytes = 0,
        .state_capacity_bytes = 8 * 64,
        .per_request_state_bytes = 64,
        .temporary_scratch_bytes = 0,
        .retained_snapshot_capacity_bytes =
            control_->snapshot_callback ? 4096U : 0U,
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{{
        .kind = TextExecutionPlanKind::kSerial,
        .physical_width = 1,
    }};
    if (control_->supports_batched_advance) {
      plans.push_back({
          .kind = TextExecutionPlanKind::kBatched,
          .physical_width = 2,
      });
      plans.push_back({
          .kind = TextExecutionPlanKind::kBatched,
          .physical_width = 4,
      });
    }
    return plans;
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return {static_cast<TextRunnerToken>(text.size())};
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest&) const override {
    return std::nullopt;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    control_->decode_calls.fetch_add(1, std::memory_order_relaxed);
    std::string text;
    for (const TextRunnerToken token : tokens) {
      if (!text.empty()) {
        text.push_back(',');
      }
      text += std::to_string(token);
    }
    return text;
  }

  [[nodiscard]] std::shared_ptr<const gufo::sampling::TokenConstraint>
  CreateJsonConstraint(
      std::shared_ptr<const gufo::JsonConstraint> prototype,
      gufo::sampling::TokenConstraint::Options options = {}) const override {
    if (!control_->json_constraints)
      return TextModelRunner::CreateJsonConstraint(std::move(prototype),
                                                   options);
    using Constraint = gufo::sampling::TokenConstraint;
    return std::make_shared<const Constraint>(
        std::move(prototype),
        std::make_shared<const Constraint::Vocabulary>(Constraint::Vocabulary{
            {"bad"}, {"{"}, {"}"}, {"\"x\":"}, {"1"}, {"", true, true}}),
        options);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    control_->states_created.fetch_add(1, std::memory_order_relaxed);
    return std::make_unique<FakeState>(control_);
  }

  void PrepareBatchExecution(TextRunnerState& state) const override {
    (void)RequireFakeState(state);
    control_->batch_preparations.fetch_add(1, std::memory_order_relaxed);
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& fake = RequireFakeState(state);
    if (offset != fake.position || prompt.empty()) {
      throw std::logic_error("invalid scheduler fake prefill");
    }
    fake.label = prompt.front();
    const std::size_t consumed = std::min(
        {max_input_tokens, prompt.size() - offset, control_->prefill_capacity});

    {
      std::unique_lock<std::mutex> lock(control_->mutex);
      control_->events.push_back({
          .kind = EventKind::kPrefill,
          .label = fake.label,
          .index = fake.position,
          .count = consumed,
          .start_ms =
              control_->clock
                  ? static_cast<double>(control_->clock->milliseconds.load())
                  : 0.0,
      });
      if (control_->block_prefill_label == fake.label &&
          !control_->prefill_gate_entered) {
        control_->prefill_gate_entered = true;
        control_->prefill_gate_label = fake.label;
        control_->condition.notify_all();
        const bool released = control_->condition.wait_for(
            lock, kTestTimeout, [&] { return control_->release_prefill; });
        if (!released) {
          throw std::runtime_error("prefill gate timed out");
        }
      }
    }

    if (control_->clock)
      control_->clock->Advance(control_->prefill_token_ms * consumed);
    fake.position += consumed;
    const bool ready = fake.position == prompt.size();
    if (ready) {
      fake.frontier = fake.label * 100;
    }
    return {
        .consumed_tokens = consumed,
        .decode_ready = ready,
    };
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state,
      gufo::sampling::SamplerState& sampler) const override {
    auto& fake = RequireFakeState(state);
    if (sampler.has_constraint()) {
      const std::vector<float> logits{100, 8, 7, 6, 5, 99};
      const auto token = sampler.Sample(logits);
      if (token == 5)
        return {.stop = true, .piece = {}};
      fake.frontier = token;
      const std::vector<std::string> pieces{"bad", "{", "}", "\"x\":", "1"};
      return {.token = token, .piece = pieces[token]};
    }
    if (!fake.frontier.has_value()) {
      throw std::logic_error("scheduler fake has no frontier");
    }
    return {
        .stop = false,
        .token = *fake.frontier,
        .piece = std::to_string(*fake.frontier),
    };
  }

  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state,
      gufo::sampling::SamplerState& sampler) const override {
    if (!control_->preview_first_token)
      return std::nullopt;
    return SelectNext(state, sampler);
  }

  struct SnapshotState final : gufo::server::TextRunnerSnapshot {
    explicit SnapshotState(const FakeState& state)
        : label(state.label),
          position(state.position),
          decode_count(state.decode_count),
          frontier(state.frontier) {}
    TextRunnerToken label;
    std::size_t position, decode_count;
    std::optional<TextRunnerToken> frontier;
    std::size_t PayloadBytes() const noexcept override {
      return sizeof(SnapshotState);
    }
  };
  std::size_t SnapshotPayloadBytes(const TextRunnerState&) const override {
    return sizeof(SnapshotState);
  }
  std::unique_ptr<gufo::server::TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override {
    if (control_->snapshot_callback)
      control_->snapshot_callback();
    return std::make_unique<SnapshotState>(RequireFakeState(state));
  }
  void RestoreOrFork(
      TextRunnerState& state,
      const gufo::server::TextRunnerSnapshot& snapshot) const override {
    auto& restored = RequireFakeState(state);
    const auto& saved = dynamic_cast<const SnapshotState&>(snapshot);
    restored.label = saved.label;
    restored.position = saved.position;
    restored.decode_count = saved.decode_count;
    restored.frontier = saved.frontier;
    if (control_->clock)
      control_->clock->Advance(control_->restore_ms);
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    const auto started = std::chrono::steady_clock::now();
    auto& fake = RequireFakeState(state);
    if (!fake.frontier.has_value() || token != *fake.frontier) {
      throw std::logic_error("scheduler fake frontier mismatch");
    }

    {
      std::unique_lock<std::mutex> lock(control_->mutex);
      if (control_->block_advance_label == fake.label &&
          fake.decode_count == 0 && !control_->advance_gate_entered) {
        control_->advance_gate_entered = true;
        control_->advance_gate_label = fake.label;
        control_->condition.notify_all();
        const bool released = control_->condition.wait_for(
            lock, kTestTimeout, [&] { return control_->release_advance; });
        if (!released) {
          throw std::runtime_error("advance gate timed out");
        }
      }
      if (control_->throw_advance_label == fake.label) {
        throw std::runtime_error("injected scheduler runner failure");
      }
      control_->events.push_back({
          .kind = EventKind::kAdvance,
          .label = fake.label,
          .index = fake.decode_count,
          .count = 1,
          .start_ms =
              control_->clock
                  ? static_cast<double>(control_->clock->milliseconds.load())
                  : 0.0,
      });
    }

    ++fake.position;
    ++fake.decode_count;
    fake.frontier = token + 1;
    if (control_->clock)
      control_->clock->Advance(control_->advance_ms);
    if (control_->after_advance)
      control_->after_advance(fake.label);
    if (fake.label == 1)
      control_->first_request_advance_ns.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - started)
              .count(),
          std::memory_order_relaxed);
  }

  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      gufo::sampling::SamplerState& sampler) const override {
    if (sampler.has_constraint() || !control_->multi_token_decode) {
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    auto& fake = RequireFakeState(state);
    if (!fake.frontier.has_value()) {
      throw std::logic_error("scheduler fake has no multi-token frontier");
    }
    const std::size_t count = std::min<std::size_t>(max_tokens, 3);
    TextDecodeStep step;
    step.selections.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
      const TextRunnerToken token = *fake.frontier;
      step.selections.push_back({
          .stop = false,
          .token = token,
          .piece = std::to_string(token),
      });
      Advance(state, token);
    }
    step.draft_tokens = count + 1;
    step.draft_accepted_tokens = count;
    return step;
  }

  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatch(
      std::span<const gufo::server::TextRunnerDecode> decodes) const override {
    auto steps = TextModelRunner::DecodeBatch(decodes);
    const auto width = std::min(control_->actual_batch_width, decodes.size());
    for (auto& step : steps) {
      step.execution_plan = {.kind = width > 1 ? TextExecutionPlanKind::kBatched
                                               : TextExecutionPlanKind::kSerial,
                             .physical_width = width};
    }
    return steps;
  }

  void AdvanceBatch(
      std::span<const TextRunnerAdvance> advances) const override {
    std::vector<TextRunnerToken> labels;
    labels.reserve(advances.size());
    for (const auto& advance : advances) {
      labels.push_back(RequireFakeState(advance.state.get()).label);
    }
    {
      const std::lock_guard<std::mutex> lock(control_->mutex);
      control_->advance_batches.push_back(std::move(labels));
    }
    TextModelRunner::AdvanceBatch(advances);
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireFakeState(state).position;
  }

private:
  std::shared_ptr<FakeControl> control_;
};

std::unique_ptr<TextGenerationScheduler> MakeScheduler(
    const std::shared_ptr<FakeControl>& control, std::size_t capacity,
    TextPrefillPolicy prefill_policy = {},
    TextSchedulerPolicy scheduler_policy = {}) {
  auto runner = std::make_shared<FakeRunner>(control);
  auto pool = std::make_shared<TextRunnerPool>(std::move(runner), capacity);
  return std::make_unique<TextGenerationScheduler>(
      std::move(pool), prefill_policy, scheduler_policy,
      [clock = control->clock] {
        return clock ? clock->Now() : TextGenerationScheduler::Clock::now();
      });
}

std::size_t EventIndex(std::span<const Event> events, EventKind kind,
                       TextRunnerToken label, std::size_t occurrence = 0) {
  std::size_t seen = 0;
  for (std::size_t index = 0; index < events.size(); ++index) {
    if (events[index].kind == kind && events[index].label == label) {
      if (seen == occurrence) {
        return index;
      }
      ++seen;
    }
  }
  return events.size();
}

std::vector<TextRunnerToken> ExpectedTokens(TextRunnerToken label,
                                            std::size_t count) {
  std::vector<TextRunnerToken> tokens;
  tokens.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    tokens.push_back(label * 100 + static_cast<TextRunnerToken>(index));
  }
  return tokens;
}

TextRequestMetadata ClientMetadata(std::string client_id) {
  return {
      .client_id = std::move(client_id),
      .deadline = std::nullopt,
      .request_start = TextGenerationScheduler::Clock::now(),
  };
}

void TestIdlePrefillUsesBulkWorkUnit() {
  auto control = std::make_shared<FakeControl>();
  auto scheduler = MakeScheduler(control, 1, {.decode_active_tokens = 2});

  const auto result = scheduler->Submit({7, 70, 71, 72, 73}, 2, 0.0F).Wait();

  const auto events = control->Events();
  Expect(events.front().kind == EventKind::kPrefill &&
             events.front().label == 7 && events.front().count == 5,
         "decode-idle prefill consumes the complete prompt");
  Expect(result.prefill_chunks == 1 && result.prefill_tokens == 5,
         "decode-idle prefill metrics report one bulk work unit");
  Expect(result.active_decode_prefill_chunks == 0,
         "decode-idle prefill is not counted as active-decode work");
  Expect(result.requested_logical_concurrency == 1 &&
             result.physical_execution_width == 1 &&
             result.execution_plan == "serial-c1",
         "C=1 telemetry reports immediate serial dispatch");
}

void TestRunnerCanSkipUnusedFinalAdvance() {
  auto control = std::make_shared<FakeControl>();
  control->final_token_advance_required = false;
  auto scheduler = MakeScheduler(control, 1);

  const auto result = scheduler->Submit({7}, 2, 0.0F).Wait();
  Expect(result.tokens == ExpectedTokens(7, 2),
         "skipped final advance preserves emitted tokens");

  std::size_t advances = 0;
  for (const auto& event : control->Events()) {
    if (event.kind == EventKind::kAdvance && event.label == 7) {
      ++advances;
    }
  }
  Expect(advances == 1,
         "runner skips the unused frontier computation after the final token");
}

void TestRunnerCanReuseExactIncrementalText() {
  auto control = std::make_shared<FakeControl>();
  control->incremental_text_is_exact = true;
  auto scheduler = MakeScheduler(control, 1);

  const auto result = scheduler->Submit({7}, 2, 0.0F).Wait();
  Expect(result.text == "700701",
         "exact incremental pieces form the final response text");
  Expect(control->decode_calls.load(std::memory_order_relaxed) == 0,
         "exact incremental text avoids duplicate final decoding");
}

void TestMultiTokenDecodePublishesDraftMetricsAndDisablesPrefixReuse() {
  auto control = std::make_shared<FakeControl>();
  control->incremental_prefill = false;
  control->multi_token_decode = true;
  control->prefix_reuse = false;
  auto scheduler = MakeScheduler(control, 1);

  const auto first = scheduler->Submit({7}, 5, 0.0F).Wait();
  Expect(first.tokens == ExpectedTokens(7, 5),
         "multi-token decode preserves the generated trajectory");
  Expect(first.draft_tokens == 7 && first.draft_accepted_tokens == 5,
         "multi-token decode reports accumulated draft statistics");
  Expect(!first.cache_hit,
         "multi-token state starts without continuation reuse");
  control->WaitForInvalidations(1);

  const auto second = scheduler->Submit({7, 70}, 2, 0.0F).Wait();
  Expect(!second.cache_hit,
         "runner-disabled prefix reuse cannot retain speculative state");
}

void TestMultiTokenRunnerCanSwitchToBatchedExecution() {
  auto control = std::make_shared<FakeControl>();
  control->multi_token_decode = true;
  control->prefix_reuse = false;
  control->supports_batched_advance = true;
  control->block_prefill_label = 1;
  auto scheduler = MakeScheduler(control, 2);

  auto request_a = scheduler->Submit({1, 10}, 4, 0.0F);
  control->WaitForPrefill(1);
  auto request_b = scheduler->Submit({2, 20}, 4, 0.0F);
  control->ReleasePrefill();

  const auto result_a = request_a.Wait();
  const auto result_b = request_b.Wait();
  Expect(result_a.tokens == ExpectedTokens(1, 4) &&
             result_b.tokens == ExpectedTokens(2, 4),
         "batched speculative-capable requests preserve both trajectories");
  Expect(result_a.execution_plan == "batched-w2" &&
             result_b.execution_plan == "batched-w2",
         "speculative-capable requests can use the physical W2 plan");
  Expect(result_a.draft_tokens == 0 && result_b.draft_tokens == 0,
         "target batching bypasses per-request draft steps");
  Expect(control->batch_preparations.load(std::memory_order_relaxed) >= 2,
         "both resident states are prepared before target batching");
}

void TestStickyAdmissionPrefersResidentLineage() {
  auto control = std::make_shared<FakeControl>();
  control->snapshot_callback = [] {};
  control->prefix_reuse = true;
  control->block_prefill_label = 1;
  auto scheduler = MakeScheduler(control, 1);

  auto lineage = std::make_shared<gufo::server::TextPromptContext>();
  lineage->cache_identity = {0xAA};
  auto other = std::make_shared<gufo::server::TextPromptContext>();
  other->cache_identity = {0xBB};

  // The resident turn seeds slot 0 with lineage 0xAA.
  auto resident = scheduler->Submit(
      {1, 10}, 4, 0.0F, {}, false,
      TextRequestMetadata{.client_id = "a",
                          .request_start = TextGenerationScheduler::Clock::now(),
                          .prompt_context = lineage});
  control->WaitForPrefill(1);

  // Two queued turns while the slot is busy: an unrelated client and the
  // resident lineage's own follow-up. Sticky admission must hand the freed
  // slot back to the follow-up so it serves its live state in place.
  auto unrelated = scheduler->Submit(
      {2, 20}, 4, 0.0F, {}, false,
      TextRequestMetadata{.client_id = "b",
                          .request_start = TextGenerationScheduler::Clock::now(),
                          .prompt_context = other});
  auto follow_up = scheduler->Submit(
      {1, 10, 11}, 4, 0.0F, {}, false,
      TextRequestMetadata{.client_id = "a",
                          .request_start = TextGenerationScheduler::Clock::now(),
                          .prompt_context = lineage});
  control->ReleasePrefill();

  const auto result_resident = resident.Wait();
  const auto result_follow_up = follow_up.Wait();
  const auto result_unrelated = unrelated.Wait();
  Expect(!result_resident.tokens.empty() && !result_follow_up.tokens.empty() &&
             !result_unrelated.tokens.empty(),
         "all three queued turns complete");
  const auto prefill_follow_up =
      EventIndex(control->events, EventKind::kPrefill, 1, 1);
  const auto prefill_unrelated =
      EventIndex(control->events, EventKind::kPrefill, 2, 0);
  Expect(prefill_follow_up < prefill_unrelated,
         "sticky admission admits the resident lineage before the unrelated "
         "head");
}

void TestBatchFailureIsolation() {
  for (const bool speculative : {false, true}) {
    auto control = std::make_shared<FakeControl>();
    control->multi_token_decode = speculative;
    control->batched_multi_token_decode = speculative;
    control->supports_batched_advance = true;
    control->block_prefill_label = 1;
    control->throw_advance_label = 1;
    auto scheduler = MakeScheduler(control, 2);
    auto failed = scheduler->Submit({1}, 12, 0.0F);
    control->WaitForPrefill(1);
    auto healthy = scheduler->Submit({2}, 12, 0.0F);
    control->ReleasePrefill();
    bool threw = false;
    try {
      (void)failed.Wait();
    } catch (const std::exception&) {
      threw = true;
    }
    Expect(threw, "failed batch member reports its error");
    const auto result = healthy.Wait();
    Expect(result.tokens == ExpectedTokens(2, 12),
           "a failing batch member must not fail or corrupt a healthy peer");
  }
}

void TestModelOwnedBatchMetrics() {
  for (const std::size_t actual_width : {1U, 2U}) {
    auto control = std::make_shared<FakeControl>();
    control->multi_token_decode = true;
    control->batched_multi_token_decode = true;
    control->supports_batched_advance = true;
    control->actual_batch_width = actual_width;
    control->block_prefill_label = 1;
    auto scheduler = MakeScheduler(control, 2);
    auto first = scheduler->Submit({1}, 12, 0.0F);
    control->WaitForPrefill(1);
    auto second = scheduler->Submit({2}, 12, 0.0F);
    control->ReleasePrefill();
    for (const auto& result : {first.Wait(), second.Wait()}) {
      Expect(
          result.physical_execution_width == actual_width &&
              result.execution_plan ==
                  (actual_width == 1 ? "serial-fallback" : "batched-w2"),
          "scheduler reports the runner's actual subgroup or serial execution");
    }
  }
}

void TestMultiResidentPrefillUsesBoundedWorkUnits() {
  for (const auto capacity : {1U, 2U}) {
    auto control = std::make_shared<FakeControl>();
    control->prefill_capacity = 3;
    auto scheduler =
        MakeScheduler(control, capacity, {.decode_active_tokens = 2});
    const auto result =
        scheduler->Submit({7, 70, 71, 72, 73, 74, 75}, 2, 0.0F).Wait();
    std::vector<std::size_t> chunks;
    for (const auto& event : control->Events()) {
      if (event.kind == EventKind::kPrefill && event.label == 7)
        chunks.push_back(event.count);
    }
    Expect(chunks == std::vector<std::size_t>({3, 3, 1}),
           "idle prefill uses model geometry regardless of spare slots");
    Expect(
        result.prefill_chunks == 3 && result.active_decode_prefill_chunks == 0,
        "model-sized work units still yield for admission and cancellation");
  }
}

void TestDecodeActivePrefillIsBounded(bool multi_token, bool batched) {
  auto control = std::make_shared<FakeControl>();
  control->multi_token_decode = multi_token;
  control->batched_multi_token_decode = batched;
  control->supports_batched_advance = multi_token;
  control->block_advance_label = 1;
  auto scheduler = MakeScheduler(control, 2, {.decode_active_tokens = 2});

  const std::size_t output_tokens = multi_token ? 30 : 10;
  auto request_a = scheduler->Submit({1}, output_tokens, 0.0F);
  control->WaitForAdvance(1);
  auto request_b = scheduler->Submit({2, 20, 21, 22, 23, 24, 25}, 2, 0.0F);
  control->ReleaseAdvance();

  const auto result_a = request_a.Wait();
  const auto result_b = request_b.Wait();
  Expect(result_a.tokens == ExpectedTokens(1, output_tokens),
         "active decoder preserves its isolated trajectory");
  Expect(result_b.tokens == ExpectedTokens(2, 2),
         "chunked prefill preserves the new request trajectory");

  const auto events = control->Events();
  std::size_t previous_b_prefill = events.size();
  std::size_t b_prefill_chunks = 0;
  for (std::size_t index = 0; index < events.size(); ++index) {
    if (events[index].kind != EventKind::kPrefill || events[index].label != 2) {
      continue;
    }
    Expect(events[index].count <= 2,
           "active-decode prefill respects its token budget");
    if (previous_b_prefill != events.size()) {
      bool a_advanced = false;
      for (std::size_t between = previous_b_prefill + 1; between < index;
           ++between) {
        a_advanced =
            a_advanced || (events[between].kind == EventKind::kAdvance &&
                           events[between].label == 1);
      }
      Expect(a_advanced, "an active decoder advances between prefill chunks");
    }
    previous_b_prefill = index;
    ++b_prefill_chunks;
  }
  Expect(b_prefill_chunks == 4,
         "long active-decode prompt is split into bounded chunks");
  Expect(result_b.prefill_chunks == 4 && result_b.prefill_tokens == 7 &&
             result_b.active_decode_prefill_chunks == 4 &&
             result_b.max_prefill_chunk_tokens == 2 &&
             result_b.max_consecutive_active_prefill_chunks == 1,
         "chunk metrics capture the selected active-decode policy");
}

void TestPrefillYieldsToEveryDueDecoder(bool multi_token, bool batched) {
  auto control = std::make_shared<FakeControl>();
  control->multi_token_decode = multi_token;
  control->batched_multi_token_decode = batched;
  control->supports_batched_advance = multi_token;
  control->block_advance_label = 2;
  auto scheduler = MakeScheduler(control, 3, {.decode_active_tokens = 2});

  const std::size_t output_tokens = multi_token ? 30 : 10;
  auto request_a = scheduler->Submit({1}, output_tokens, 0.0F);
  auto request_b = scheduler->Submit({2}, output_tokens, 0.0F);
  control->WaitForAdvance(2);
  auto request_c = scheduler->Submit({3, 30, 31, 32, 33, 34, 35}, 2, 0.0F);
  control->ReleaseAdvance();

  const auto result_a = request_a.Wait();
  const auto result_b = request_b.Wait();
  const auto result_c = request_c.Wait();
  Expect(result_a.tokens == ExpectedTokens(1, output_tokens) &&
             result_b.tokens == ExpectedTokens(2, output_tokens) &&
             result_c.tokens == ExpectedTokens(3, 2),
         "all mixed prefill/decode trajectories remain isolated");

  const auto events = control->Events();
  std::size_t previous_c_prefill = events.size();
  for (std::size_t index = 0; index < events.size(); ++index) {
    if (events[index].kind != EventKind::kPrefill || events[index].label != 3) {
      continue;
    }
    if (previous_c_prefill != events.size()) {
      bool a_advanced = false;
      bool b_advanced = false;
      for (std::size_t between = previous_c_prefill + 1; between < index;
           ++between) {
        if (events[between].kind == EventKind::kAdvance) {
          a_advanced = a_advanced || events[between].label == 1;
          b_advanced = b_advanced || events[between].label == 2;
        }
      }
      Expect(a_advanced && b_advanced,
             "every due decoder advances before another prefill chunk");
    }
    previous_c_prefill = index;
  }
  Expect(result_c.max_consecutive_active_prefill_chunks == 1,
         "scheduler never runs consecutive chunks while decode is due");
}

void TestNonIncrementalRunnerFallsBackSafely() {
  auto control = std::make_shared<FakeControl>();
  control->incremental_prefill = false;
  control->block_advance_label = 1;
  auto scheduler = MakeScheduler(control, 2, {.decode_active_tokens = 2});

  auto request_a = scheduler->Submit({1}, 8, 0.0F);
  control->WaitForAdvance(1);
  auto request_b = scheduler->Submit({2, 20, 21, 22, 23}, 2, 0.0F);
  control->ReleaseAdvance();

  Expect(request_a.Wait().tokens == ExpectedTokens(1, 8),
         "fallback preserves the active decoder trajectory");
  const auto result_b = request_b.Wait();
  Expect(result_b.tokens == ExpectedTokens(2, 2),
         "fallback preserves the admitted request trajectory");
  Expect(
      !result_b.incremental_prefill_supported && result_b.prefill_chunks == 1 &&
          result_b.max_prefill_chunk_tokens == 5 &&
          result_b.prefill_fallback_reason == "incremental_prefill_unavailable",
      "non-incremental runners report their full-prefill fallback");
}

void TestPendingLimitsRejectBeforeStateAdmission() {
  auto control = std::make_shared<FakeControl>();
  control->block_advance_label = 1;
  auto scheduler = MakeScheduler(control, 1, {},
                                 {
                                     .max_pending_requests = 1,
                                     .max_pending_requests_per_client = 1,
                                 });

  auto active = scheduler->Submit({1}, 4, 0.0F, {}, false,
                                  ClientMetadata("active-client"));
  control->WaitForAdvance(1);
  auto queued = scheduler->Submit({2}, 1, 0.0F, {}, false,
                                  ClientMetadata("queued-client"));

  bool rejected = false;
  try {
    (void)scheduler->Submit({3}, 1, 0.0F, {}, false,
                            ClientMetadata("third-client"));
  } catch (const TextGenerationError& error) {
    rejected = error.code() == TextGenerationErrorCode::kQueueFull;
  }
  Expect(rejected, "full pending queue rejects before admission");
  Expect(control->states_created.load(std::memory_order_relaxed) == 1,
         "rejected request cannot allocate another runner state");

  control->ReleaseAdvance();
  Expect(active.Wait().tokens == ExpectedTokens(1, 4) &&
             queued.Wait().tokens == ExpectedTokens(2, 1),
         "accepted work survives a queue rejection");
}

void TestPendingClientsAreRoundRobinAndIndividuallyBounded() {
  auto control = std::make_shared<FakeControl>();
  control->block_advance_label = 1;
  auto scheduler = MakeScheduler(control, 1, {},
                                 {
                                     .max_pending_requests = 4,
                                     .max_pending_requests_per_client = 2,
                                 });

  auto active = scheduler->Submit({1}, 3, 0.0F, {}, false,
                                  ClientMetadata("active-client"));
  control->WaitForAdvance(1);
  auto client_a_first =
      scheduler->Submit({2}, 1, 0.0F, {}, false, ClientMetadata("client-a"));
  auto client_a_second =
      scheduler->Submit({3}, 1, 0.0F, {}, false, ClientMetadata("client-a"));
  auto client_b =
      scheduler->Submit({4}, 1, 0.0F, {}, false, ClientMetadata("client-b"));

  bool client_rejected = false;
  try {
    (void)scheduler->Submit({5}, 1, 0.0F, {}, false,
                            ClientMetadata("client-a"));
  } catch (const TextGenerationError& error) {
    client_rejected = error.code() == TextGenerationErrorCode::kClientQueueFull;
  }
  Expect(client_rejected, "one client cannot monopolize the pending queue");

  control->ReleaseAdvance();
  (void)active.Wait();
  (void)client_a_first.Wait();
  (void)client_a_second.Wait();
  (void)client_b.Wait();

  const auto events = control->Events();
  const std::size_t a_first = EventIndex(events, EventKind::kPrefill, 2);
  const std::size_t a_second = EventIndex(events, EventKind::kPrefill, 3);
  const std::size_t b_first = EventIndex(events, EventKind::kPrefill, 4);
  Expect(a_first < b_first && b_first < a_second,
         "pending clients rotate before one client receives another admission");
}

void TestExpiredQueuedRequestNeverConsumesState() {
  auto control = std::make_shared<FakeControl>();
  control->block_advance_label = 1;
  auto scheduler = MakeScheduler(control, 1);

  auto active = scheduler->Submit({1}, 3, 0.0F);
  control->WaitForAdvance(1);
  auto expired = scheduler->Submit(
      {2}, 1, 0.0F, {}, false,
      TextRequestMetadata{
          .client_id = "expired-client",
          .deadline = TextGenerationScheduler::Clock::now() -
                      std::chrono::milliseconds{1},
          .request_start = TextGenerationScheduler::Clock::now(),
      });
  control->ReleaseAdvance();

  bool deadline_reported = false;
  try {
    (void)expired.Wait();
  } catch (const TextGenerationError& error) {
    deadline_reported =
        error.code() == TextGenerationErrorCode::kDeadlineExceeded;
  }
  Expect(deadline_reported, "expired queued work reports a stable deadline");
  Expect(active.Wait().tokens == ExpectedTokens(1, 3),
         "expired queued work does not disturb active generation");
  Expect(control->states_created.load(std::memory_order_relaxed) == 1,
         "expired queued work never creates or acquires another state");
}

void TestSlowConsumerOutputIsBoundedAndReclaimed() {
  auto control = std::make_shared<FakeControl>();
  control->block_advance_label = 5;
  auto scheduler = MakeScheduler(control, 1, {},
                                 {
                                     .max_buffered_output_bytes_per_request = 4,
                                     .max_buffered_output_bytes_total = 4,
                                 });

  auto request = scheduler->Submit({5}, 5, 0.0F, {}, true);
  std::mutex callback_mutex;
  std::condition_variable callback_condition;
  bool callback_entered = false;
  bool release_callback = false;
  bool backpressure_reported = false;
  std::jthread consumer([&] {
    try {
      (void)request.Wait([&](std::string_view) {
        std::unique_lock<std::mutex> lock(callback_mutex);
        callback_entered = true;
        control->ReleaseAdvance();
        callback_condition.notify_all();
        callback_condition.wait(lock, [&] { return release_callback; });
        return true;
      });
    } catch (const TextGenerationError& error) {
      backpressure_reported =
          error.code() == TextGenerationErrorCode::kOutputBackpressure;
    }
  });

  {
    std::unique_lock<std::mutex> lock(callback_mutex);
    const bool entered = callback_condition.wait_for(
        lock, kTestTimeout, [&] { return callback_entered; });
    Expect(entered, "slow consumer receives its first output piece");
  }
  control->WaitForInvalidations(1);
  {
    const std::lock_guard<std::mutex> lock(callback_mutex);
    release_callback = true;
  }
  callback_condition.notify_all();
  consumer.join();

  Expect(backpressure_reported,
         "bounded output queue fails a persistently slow consumer");
  Expect(scheduler->buffered_output_bytes() == 0,
         "failed slow-consumer output is fully reclaimed");
  Expect(scheduler->max_buffered_output_bytes() <= 4,
         "server-wide buffered output never exceeds its declared limit");

  const auto replacement = scheduler->Submit({6}, 2, 0.0F).Wait();
  Expect(replacement.tokens == ExpectedTokens(6, 2),
         "state is reusable after output backpressure cancellation");
}

void TestGeneratedOutputLimitAppliesWithoutStreaming() {
  auto control = std::make_shared<FakeControl>();
  auto scheduler = MakeScheduler(control, 1, {},
                                 {
                                     .max_output_bytes_per_request = 4,
                                 });

  bool output_limit_reported = false;
  try {
    (void)scheduler->Submit({7}, 3, 0.0F).Wait();
  } catch (const TextGenerationError& error) {
    output_limit_reported =
        error.code() == TextGenerationErrorCode::kOutputLimit;
  }
  Expect(output_limit_reported,
         "non-streaming generation obeys its output byte limit");
}

void TestMidGenerationAdmissionAndIsolatedTrajectories() {
  auto control = std::make_shared<FakeControl>();
  control->block_advance_label = 1;
  auto scheduler = MakeScheduler(control, 2);

  auto request_a = scheduler->Submit({1, 10}, 4, 0.0F, {}, true);
  control->WaitForAdvance(1);
  auto request_b = scheduler->Submit({2, 20}, 2, 0.0F, {}, true);
  control->ReleaseAdvance();

  std::vector<std::string> pieces_a;
  const auto result_a = request_a.Wait([&](std::string_view piece) {
    pieces_a.emplace_back(piece);
    return true;
  });
  std::vector<std::string> pieces_b;
  const auto result_b = request_b.Wait([&](std::string_view piece) {
    pieces_b.emplace_back(piece);
    return true;
  });

  Expect(result_a.tokens == ExpectedTokens(1, 4),
         "request A follows its isolated greedy trajectory");
  Expect(result_b.tokens == ExpectedTokens(2, 2),
         "request B follows its isolated greedy trajectory");
  Expect(pieces_a.size() == 4 && pieces_b.size() == 2,
         "each client receives only its own token pieces");

  const auto events = control->Events();
  const std::size_t b_prefill = EventIndex(events, EventKind::kPrefill, 2);
  const std::size_t a_last_advance =
      EventIndex(events, EventKind::kAdvance, 1, 3);
  Expect(b_prefill < a_last_advance,
         "request B begins prefill before request A completes");
}

void TestMidGenerationRequestJoinsNextDecodeBatch() {
  auto control = std::make_shared<FakeControl>();
  control->supports_batched_advance = true;
  control->block_advance_label = 1;
  auto scheduler = MakeScheduler(control, 2);

  auto request_a = scheduler->Submit({1}, 8, 0.0F);
  control->WaitForAdvance(1);
  auto request_b = scheduler->Submit({2}, 4, 0.0F);
  control->ReleaseAdvance();

  const auto result_a = request_a.Wait();
  const auto result_b = request_b.Wait();
  Expect(result_a.tokens == ExpectedTokens(1, 8) &&
             result_b.tokens == ExpectedTokens(2, 4),
         "dynamic batching preserves both isolated trajectories");

  bool joined = false;
  for (const auto& labels : control->AdvanceBatches()) {
    joined = joined || labels == std::vector<TextRunnerToken>({1, 2}) ||
             labels == std::vector<TextRunnerToken>({2, 1});
  }
  Expect(joined,
         "request B joins request A at a decode boundary after prefill");
  Expect(result_a.physical_execution_width == 2 &&
             result_b.physical_execution_width == 2 &&
             result_a.execution_plan == "batched-w2" &&
             result_b.execution_plan == "batched-w2",
         "joined requests report the real W=2 execution plan");
}

void TestFifoReplacementAdmissionWithOneSlot() {
  auto control = std::make_shared<FakeControl>();
  control->block_advance_label = 1;
  auto scheduler = MakeScheduler(control, 1);

  auto request_a = scheduler->Submit({1, 10}, 2, 0.0F);
  control->WaitForAdvance(1);
  auto request_b = scheduler->Submit({2, 20}, 1, 0.0F);
  auto request_c = scheduler->Submit({3, 30}, 1, 0.0F);
  control->ReleaseAdvance();

  const auto result_a = request_a.Wait();
  const auto result_b = request_b.Wait();
  const auto result_c = request_c.Wait();
  Expect(!result_a.cancelled && !result_b.cancelled && !result_c.cancelled,
         "all FIFO requests complete");

  std::vector<TextRunnerToken> prefill_order;
  for (const auto& event : control->Events()) {
    if (event.kind == EventKind::kPrefill) {
      prefill_order.push_back(event.label);
    }
  }
  Expect(prefill_order == std::vector<TextRunnerToken>({1, 2, 3}),
         "replacement admission preserves FIFO order");
}

void TestQueuedAndPrefillCancellation() {
  {
    auto control = std::make_shared<FakeControl>();
    control->block_advance_label = 1;
    auto scheduler = MakeScheduler(control, 1);

    auto active = scheduler->Submit({1, 10}, 2, 0.0F);
    control->WaitForAdvance(1);
    auto queued = scheduler->Submit({2, 20}, 1, 0.0F);
    queued.Cancel();
    control->ReleaseAdvance();

    Expect(!active.Wait().cancelled, "active request completes normally");
    Expect(queued.Wait().cancelled, "queued request cancellation is reported");
    const auto events = control->Events();
    Expect(EventIndex(events, EventKind::kPrefill, 2) == events.size(),
           "cancelled queued request never acquires a model state");
  }

  {
    auto control = std::make_shared<FakeControl>();
    control->block_advance_label = 1;
    control->block_prefill_label = 4;
    auto scheduler = MakeScheduler(control, 2, {.decode_active_tokens = 2});

    auto active = scheduler->Submit({1}, 4, 0.0F);
    control->WaitForAdvance(1);
    auto request = scheduler->Submit({4, 40, 41, 42}, 2, 0.0F);
    control->ReleaseAdvance();
    control->WaitForPrefill(4);
    request.Cancel();
    control->ReleasePrefill();

    const auto result = request.Wait();
    Expect(!active.Wait().cancelled,
           "active decoder survives another request cancellation");
    Expect(result.cancelled, "active-decode prefill cancellation is reported");
    Expect(EventIndex(control->Events(), EventKind::kAdvance, 4) ==
               control->Events().size(),
           "cancelled prefill never advances decode");
  }
}

void TestDecodeCancellationAndStateReclamation() {
  auto control = std::make_shared<FakeControl>();
  control->block_advance_label = 5;
  auto scheduler = MakeScheduler(control, 1);

  auto cancelled = scheduler->Submit({5, 50}, 5, 0.0F, {}, true);
  std::size_t delivered_pieces = 0;
  const auto cancelled_result = cancelled.Wait([&](std::string_view) {
    ++delivered_pieces;
    control->ReleaseAdvance();
    return false;
  });
  Expect(cancelled_result.cancelled,
         "callback cancellation reaches the scheduler");
  Expect(delivered_pieces == 1,
         "no token is published after callback cancellation");

  auto replacement = scheduler->Submit({6, 60}, 2, 0.0F);
  const auto replacement_result = replacement.Wait();
  Expect(replacement_result.tokens == ExpectedTokens(6, 2),
         "cancelled state slot is immediately reusable");
  Expect(control->invalidations.load(std::memory_order_relaxed) >= 1,
         "decode cancellation invalidates partial model state");
}

void TestFourResidentRequestsMakeProgress() {
  auto control = std::make_shared<FakeControl>();
  auto scheduler = MakeScheduler(control, 4);

  std::vector<TextGenerationScheduler::Request> requests;
  for (TextRunnerToken label = 1; label <= 4; ++label) {
    requests.push_back(scheduler->Submit({label, label + 10}, 3, 0.0F));
  }
  for (std::size_t index = 0; index < requests.size(); ++index) {
    const auto result = requests[index].Wait();
    Expect(result.tokens ==
               ExpectedTokens(static_cast<TextRunnerToken>(index + 1), 3),
           "C=4 serial fallback preserves isolated output");
    Expect(result.requested_logical_concurrency == 4 &&
               result.physical_execution_width == 1 &&
               result.execution_plan == "serial-fallback",
           "C=4 telemetry exposes the physical serial fallback");
  }
}

void TestRunnerFailureInvalidatesAndDoesNotPoisonReplacement() {
  auto control = std::make_shared<FakeControl>();
  control->throw_advance_label = 9;
  auto scheduler = MakeScheduler(control, 1);

  bool failed = false;
  try {
    auto request = scheduler->Submit({9, 90}, 2, 0.0F);
    (void)request.Wait();
  } catch (const std::runtime_error& exception) {
    failed = std::string_view(exception.what()) ==
             "injected scheduler runner failure";
  }
  Expect(failed, "runner failure reaches the submitting client");

  control->throw_advance_label.reset();
  auto replacement = scheduler->Submit({8, 80}, 2, 0.0F);
  Expect(replacement.Wait().tokens == ExpectedTokens(8, 2),
         "replacement request succeeds after runner failure");
}

}  // namespace

void TestFirstTokenPrecedesSnapshotAndPreservesBudget() {
  for (const bool multi : {false, true}) {
    auto control = std::make_shared<FakeControl>();
    control->multi_token_decode = multi;
    control->preview_first_token = true;
    std::binary_semaphore captured(0), release(0);
    control->snapshot_callback = [&] {
      captured.release();
      release.acquire();
    };
    auto scheduler = MakeScheduler(control, 1);
    auto request = scheduler->Submit({1, 10}, 7, 0.0F, {}, true);
    std::counting_semaphore<16> first_token(0);
    auto result = std::async(std::launch::async, [&] {
      return request.Wait([&](std::string_view) {
        first_token.release();
        return true;
      });
    });
    Expect(captured.try_acquire_for(kTestTimeout), "snapshot capture reached");
    const bool published = first_token.try_acquire_for(std::chrono::seconds(1));
    release.release();
    Expect(published, "first token is delivered while snapshot capture blocks");
    Expect(result.get().tokens == ExpectedTokens(1, 7),
           "preview is emitted once and counts toward the original budget");
  }
  for (const std::size_t limit : {1U, 7U}) {
    auto control = std::make_shared<FakeControl>();
    control->multi_token_decode = true;
    control->preview_first_token = true;
    control->batched_multi_token_decode = true;
    control->supports_batched_advance = true;
    auto scheduler = MakeScheduler(control, 4);
    std::vector<TextGenerationScheduler::Request> requests;
    for (TextRunnerToken label = 1; label <= 4; ++label)
      requests.push_back(scheduler->Submit({label, 10}, limit, 0.0F));
    for (std::size_t index = 0; index < requests.size(); ++index)
      Expect(requests[index].Wait().tokens ==
                 ExpectedTokens(static_cast<TextRunnerToken>(index + 1), limit),
             "batched previews retain independent token limits");
  }
}

void TestSnapshotDoesNotBlockOtherRequests() {
  for (const bool multi : {false, true}) {
    auto control = std::make_shared<FakeControl>();
    control->multi_token_decode = multi;
    control->preview_first_token = true;
    control->supports_batched_advance = true;
    control->batched_multi_token_decode = multi;
    std::binary_semaphore entered(0), release(0);
    std::atomic<unsigned> captures{0};
    control->snapshot_callback = [&] {
      if (captures.fetch_add(1) == 0) {
        entered.release();
        release.acquire();
      }
    };
    auto scheduler = MakeScheduler(control, 2, {.decode_active_tokens = 2});
    auto first = scheduler->Submit({1, 10}, 7, 0.0F);
    const bool started = entered.try_acquire_for(kTestTimeout);
    auto second = scheduler->Submit({2, 20, 21, 22, 23, 24, 25}, 7, 0.0F);
    auto result = std::async(std::launch::async, [&] { return second.Wait(); });
    const bool independent =
        result.wait_for(kTestTimeout) == std::future_status::ready;
    release.release();
    Expect(started && independent, "one capture must not block other requests");
    const auto second_result = result.get();
    Expect(second_result.tokens == ExpectedTokens(2, 7),
           "other request stays independent");
    Expect(second_result.max_prefill_chunk_tokens == 2 &&
               second_result.prefill_chunks == 4,
           "prefill stays bounded while an already-published request captures");
    const auto first_result = first.Wait();
    Expect(first_result.tokens == ExpectedTokens(1, 7),
           "captured request resumes exactly");
    const double advance_ms =
        control->first_request_advance_ns.load(std::memory_order_relaxed) / 1e6;
    Expect(advance_ms > 0 && first_result.decode_ms >= advance_ms,
           "async snapshot time must not be subtracted from timed model work");
    auto cached = scheduler->Submit({1, 10}, 7, 0.0F).Wait();
    Expect(cached.cache_hit && cached.tokens == ExpectedTokens(1, 7),
           "asynchronous capture retains the immutable prompt frontier");
  }
}

void TestCapturesAtCapacityAllowQueuedProgress() {
  for (const bool multi : {false, true}) {
    for (const std::size_t capacity : {1U, 2U, 4U}) {
      auto control = std::make_shared<FakeControl>();
      control->multi_token_decode = multi;
      control->preview_first_token = true;
      control->supports_batched_advance = true;
      control->batched_multi_token_decode = multi;
      std::counting_semaphore<16> entered(0), release(0);
      std::atomic<std::size_t> captures{0};
      control->snapshot_callback = [&] {
        if (captures.fetch_add(1) < capacity) {
          entered.release();
          release.acquire();
        }
      };
      auto scheduler = MakeScheduler(control, capacity);
      std::vector<TextGenerationScheduler::Request> requests;
      for (std::size_t i = 0; i < capacity; ++i) {
        TextGenerationScheduler::RequestMetadata metadata;
        // Exercise captures during prefill as well as after first-token
        // publication, including the single-slot admission deadlock.
        metadata.cache_prefix_tokens = multi ? 1 : 0;
        requests.push_back(
            scheduler->Submit({static_cast<TextRunnerToken>(i + 1), 10}, 7,
                              0.0F, {}, false, metadata));
        Expect(entered.try_acquire_for(kTestTimeout),
               "every resident reaches snapshot capture");
      }
      for (std::size_t i = capacity; i < capacity * 2; ++i)
        requests.push_back(scheduler->Submit(
            {static_cast<TextRunnerToken>(i + 1), 10}, 7, 0.0F));
      release.release(static_cast<std::ptrdiff_t>(capacity));
      auto results = std::async(std::launch::async, [&] {
        for (std::size_t i = 0; i < requests.size(); ++i)
          Expect(requests[i].Wait().tokens ==
                     ExpectedTokens(static_cast<TextRunnerToken>(i + 1), 7),
                 "capture and queued requests retain independent output");
      });
      Expect(results.wait_for(kTestTimeout) == std::future_status::ready,
             "all slots capturing must not deadlock queued admission");
      results.get();
    }
  }
}

void TestShutdownCancelsRunnerAcquisition() {
  auto control = std::make_shared<FakeControl>();
  auto pool = std::make_shared<TextRunnerPool>(
      std::make_shared<FakeRunner>(control), 1);
  auto lease = pool->Acquire({1, 10});
  auto scheduler = std::make_unique<TextGenerationScheduler>(pool);
  std::binary_semaphore acquiring(0);
  std::atomic<bool> notified{false};
  auto request = scheduler->Submit({2, 10}, 7, 0.0F, [&] {
    if (!notified.exchange(true))
      acquiring.release();
    return false;
  });
  Expect(acquiring.try_acquire_for(kTestTimeout),
         "request reaches admission with external runner lease");
  auto stopped = std::async(std::launch::async, [&] { scheduler.reset(); });
  const bool completed =
      stopped.wait_for(kTestTimeout) == std::future_status::ready;
  lease = {};
  Expect(completed, "shutdown cancels acquisition of a leased runner");
  stopped.get();
}

void TestJsonMetadataAndConcurrentCursors() {
  for (const bool multi : {false, true}) {
    auto control = std::make_shared<FakeControl>();
    control->json_constraints = true;
    control->multi_token_decode = multi;
    control->batched_multi_token_decode = multi;
    control->supports_batched_advance = true;
    control->incremental_text_is_exact = true;
    control->preview_first_token = true;
    control->snapshot_callback = [] {};
    control->block_prefill_label = 1;
    auto scheduler = MakeScheduler(control, 4);
    const auto prototype = std::make_shared<const gufo::JsonConstraint>(
        gufo::JsonConstraint::Compile(R"({"const":{}})"));
    TextRequestMetadata empty;
    empty.json_constraint = scheduler->runner().CreateJsonConstraint(prototype);
    TextRequestMetadata value;
    value.json_constraint = scheduler->runner().CreateJsonConstraint(
        std::make_shared<const gufo::JsonConstraint>(
            gufo::JsonConstraint::Compile(R"({"const":{"x":1}})")));
    auto first = scheduler->Submit({1, 10}, 12, 0.0F, {}, true, empty);
    control->WaitForPrefill(1);
    auto second = scheduler->Submit({2, 20}, 12, 0.0F, {}, false, value);
    auto third = scheduler->Submit({3, 30}, 12, 0.0F, {}, false, empty);
    auto plain = scheduler->Submit({4, 40}, 4, 0.0F);
    control->ReleasePrefill();
    std::string streamed;
    const auto a = first.Wait([&](std::string_view piece) {
      streamed += piece;
      return true;
    });
    const auto b = second.Wait();
    const auto c = third.Wait();
    const auto d = plain.Wait();
    Expect(a.text == "{}" && streamed == a.text && b.text == "{\"x\":1}" &&
               c.text == "{}",
           "streamed and buffered metadata retain independent grammar cursors");
    Expect(a.finish_reason ==
                   gufo::server::TextGenerationBackend::FinishReason::kStop &&
               b.finish_reason == a.finish_reason &&
               c.finish_reason == a.finish_reason,
           "constrained requests stop only at completed grammar");
    Expect(a.draft_tokens == 0 && b.draft_tokens == 0 && c.draft_tokens == 0,
           "mixed batch constrained rows never use draft generation");
    Expect(d.tokens == ExpectedTokens(4, 4),
           "unconstrained trajectory remains unchanged");
    const auto cached =
        scheduler->Submit({1, 10}, 12, 0.0F, {}, false, empty).Wait();
    Expect(cached.cache_hit && cached.text == "{}" && prototype->Allows("{}"),
           "cache reuse starts a fresh grammar, including first-token preview");
  }
}

void TestJsonAdmissionRejectsBeforeInference() {
  auto control = std::make_shared<FakeControl>();
  control->json_constraints = true;
  auto scheduler = MakeScheduler(control, 1);
  TextRequestMetadata metadata;
  metadata.json_constraint = scheduler->runner().CreateJsonConstraint(
      std::make_shared<const gufo::JsonConstraint>(
          gufo::JsonConstraint::JsonObject()));
  bool rejected = false;
  try {
    (void)scheduler->Submit({1}, 4, {.temperature = 1, .top_k = 1}, {}, false,
                            metadata);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Expect(
      rejected && control->Events().empty(),
      "sampled JSON is rejected before queued inference even with top-k one");
  auto unsupported = std::make_shared<FakeControl>();
  auto other = MakeScheduler(unsupported, 1);
  rejected = false;
  try {
    (void)other->Submit({1}, 4, 0.0F, {}, false, metadata);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Expect(rejected && unsupported->Events().empty(),
         "unsupported runner rejects constraint before inference");
}

TextRequestMetadata TimedMetadata(const std::shared_ptr<FakeControl>& control,
                                  std::string client = "anonymous",
                                  std::optional<std::size_t> estimate = {}) {
  auto metadata = ClientMetadata(std::move(client));
  metadata.request_start = control->clock->Now();
  metadata.estimated_prefill_tokens = estimate;
  return metadata;
}

void ExpectMilliseconds(double actual, double expected,
                        std::string_view label) {
  Expect(std::abs(actual - expected) < 1e-6, label);
}

void TestCandidatePoliciesAreOptInAndValidated() {
  const TextSchedulerPolicy policy;
  const TextPrefillPolicy prefill;
  Expect(policy.decode_burst.count() == 0 &&
             policy.queue_timeout.count() == 0 &&
             policy.max_inflight_requests_per_client == 0 &&
             policy.max_inflight_prefill_tokens == 0 &&
             !policy.prefer_short_prefill && !prefill.adaptive_chunking &&
             prefill.decode_active_tokens == 512,
         "unqualified policies do not silently change production defaults");
  for (int invalid = 0; invalid != 7; ++invalid) {
    auto control = std::make_shared<FakeControl>();
    auto p = policy;
    auto f = prefill;
    if (invalid == 0)
      p.decode_burst = std::chrono::milliseconds{-1};
    if (invalid == 1)
      p.queue_timeout = std::chrono::milliseconds{-1};
    if (invalid == 2)
      p.max_decode_steps_per_burst = 0;
    if (invalid == 3)
      p.admission_aging = std::chrono::milliseconds{-1};
    if (invalid == 4)
      f.target_chunk_time = std::chrono::milliseconds{0};
    if (invalid == 5)
      f.min_chunk_tokens = 0;
    if (invalid == 6)
      f.min_chunk_tokens = 513;
    bool rejected = false;
    try {
      (void)MakeScheduler(control, 2, f, p);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    Expect(rejected, "invalid bounded latency policy is rejected");
  }
}

void TestSustainedColdPrefillAndWarmDecode() {
  auto control = std::make_shared<FakeControl>();
  control->clock = std::make_shared<FakeClock>();
  control->advance_ms = 20;
  control->prefill_token_ms = 10;
  control->block_advance_label = 1;
  auto scheduler =
      MakeScheduler(control, 2, {.decode_active_tokens = 2},
                    {.decode_burst = std::chrono::milliseconds{100}});
  auto warm =
      scheduler->Submit({1}, 100, 0.0F, {}, false, TimedMetadata(control));
  control->WaitForAdvance(1);
  auto first = scheduler->Submit(std::vector<TextRunnerToken>(8, 2), 2, 0.0F,
                                 {}, false, TimedMetadata(control, "first"));
  auto second = scheduler->Submit(std::vector<TextRunnerToken>(8, 3), 2, 0.0F,
                                  {}, false, TimedMetadata(control, "second"));
  std::optional<TextGenerationScheduler::Request> later;
  std::size_t warm_steps = 0;
  control->after_advance = [&](TextRunnerToken label) {
    if (label == 1 && ++warm_steps == 12)
      later.emplace(scheduler->Submit(std::vector<TextRunnerToken>(8, 4), 2,
                                      0.0F, {}, false,
                                      TimedMetadata(control, "later")));
  };
  control->ReleaseAdvance();
  const auto warm_result = warm.Wait();
  Expect(later.has_value(),
         "fixture includes a later arrival, not only one burst");
  for (const auto& result : {first.Wait(), second.Wait(), later->Wait()}) {
    Expect(result.prefill_chunks == 4 && result.prefill_tokens == 8 &&
               result.resident_requests_at_admission <= 2 &&
               result.physical_execution_width == 1,
           "sustained cold requests progress within two real resident slots");
  }
  const auto events = control->Events();
  for (TextRunnerToken label : {2, 3, 4}) {
    double previous = -1;
    for (const auto& event : events) {
      if (event.kind != EventKind::kPrefill || event.label != label)
        continue;
      if (previous >= 0)
        Expect(event.start_ms - previous <= 160,
               "100ms decode bursts still guarantee bounded prefill progress");
      previous = event.start_ms;
    }
    Expect(EventIndex(events, EventKind::kPrefill, label, 3) <
               EventIndex(events, EventKind::kAdvance, 1, 99),
           "long prefill cannot starve behind an active warm decoder");
  }
  Expect(EventIndex(events, EventKind::kAdvance, 1, 4) <
             EventIndex(events, EventKind::kPrefill, 2),
         "decode gets measured work bursts rather than one step per chunk");
  ExpectMilliseconds(warm_result.peer_prefill_ms, 240,
                     "warm decoder reports all peer prefill pauses");
  Expect(warm_result.resident_wait_ms >= warm_result.peer_prefill_ms &&
             warm_result.decode_wall_ms > warm_result.decode_ms &&
             warm_result.max_inter_token_ms <= 100,
         "wall decode includes peer work while active decode does not");
  Expect(gufo::server::DecodeWallTokensPerSecond(warm_result) <
             gufo::server::ActiveDecodeTokensPerSecond(warm_result),
         "active-work TPS cannot masquerade as user-observed throughput");
}

void TestCachedReplacementsCannotStarvePrefill() {
  auto control = std::make_shared<FakeControl>();
  control->clock = std::make_shared<FakeClock>();
  control->advance_ms = 20;
  control->prefill_token_ms = 5;
  control->block_advance_label = 1;
  auto scheduler =
      MakeScheduler(control, 2, {.decode_active_tokens = 2},
                    {.decode_burst = std::chrono::milliseconds{100}});
  auto warm =
      scheduler->Submit({1}, 1, 0.0F, {}, false, TimedMetadata(control));
  control->WaitForAdvance(1);
  auto cold = scheduler->Submit(std::vector<TextRunnerToken>(6, 2), 1, 0.0F, {},
                                false, TimedMetadata(control, "cold"));
  std::vector<TextGenerationScheduler::Request> replacements;
  std::vector<TextRunnerToken> prefix{1};
  std::size_t generated = 0;
  std::binary_semaphore arrived(0);
  control->after_advance = [&](TextRunnerToken label) {
    if (label != 1)
      return;
    prefix.push_back(100 + static_cast<TextRunnerToken>(generated++));
    if (generated < 30)
      replacements.push_back(scheduler->Submit(prefix, 1, 0.0F, {}, false,
                                               TimedMetadata(control, "warm")));
    else
      arrived.release();
  };
  control->ReleaseAdvance();
  (void)warm.Wait();
  Expect(arrived.try_acquire_for(kTestTimeout),
         "bounded warm replacement sequence completes");
  for (auto& request : replacements)
    Expect(request.Wait().cache_hit,
           "replacement fixture uses genuinely ready cache frontiers");
  Expect(cold.Wait().prefill_chunks == 3,
         "cold prefill progresses despite warm turnover");
  const auto events = control->Events();
  Expect(EventIndex(events, EventKind::kPrefill, 2) <
             EventIndex(events, EventKind::kAdvance, 1, 6),
         "new decode-ready admissions cannot repeatedly extend an exhausted "
         "burst");
  Expect(EventIndex(events, EventKind::kPrefill, 2, 2) <
             EventIndex(events, EventKind::kAdvance, 1, 20),
         "sustained warm replacement leaves a bounded prefill share");
}

void TestZeroCostDecodeBurstStillYields() {
  auto control = std::make_shared<FakeControl>();
  control->clock = std::make_shared<FakeClock>();
  control->block_advance_label = 1;
  auto scheduler =
      MakeScheduler(control, 2, {.decode_active_tokens = 2},
                    {.decode_burst = std::chrono::milliseconds{100},
                     .max_decode_steps_per_burst = 3});
  auto warm =
      scheduler->Submit({1}, 30, 0.0F, {}, false, TimedMetadata(control));
  control->WaitForAdvance(1);
  auto cold = scheduler->Submit({2, 20, 21, 22, 23, 24}, 1, 0.0F, {}, false,
                                TimedMetadata(control));
  control->ReleaseAdvance();
  (void)warm.Wait();
  Expect(cold.Wait().prefill_chunks == 3, "step cap preserves cold progress");
  const auto events = control->Events();
  const auto first = EventIndex(events, EventKind::kPrefill, 2);
  Expect(EventIndex(events, EventKind::kAdvance, 1, 2) < first &&
             first < EventIndex(events, EventKind::kAdvance, 1, 4),
         "a zero-cost runner yields at the bounded decode step count");
}

void TestMeasuredPrefillBudgetAndGeometryFallback() {
  for (const bool adaptive : {false, true}) {
    auto control = std::make_shared<FakeControl>();
    control->clock = std::make_shared<FakeClock>();
    control->prefill_capacity = 2048;
    control->prefill_token_ms = 1;
    control->advance_ms = 20;
    control->block_advance_label = 1;
    auto scheduler = MakeScheduler(
        control, 2, {.min_chunk_tokens = 16, .adaptive_chunking = adaptive},
        {.decode_burst = std::chrono::milliseconds{100}});
    auto warm =
        scheduler->Submit({1}, 256, 0.0F, {}, false, TimedMetadata(control));
    control->WaitForAdvance(1);
    auto cold = scheduler->Submit(std::vector<TextRunnerToken>(2048, 2), 1,
                                  0.0F, {}, false, TimedMetadata(control));
    control->ReleaseAdvance();
    (void)warm.Wait();
    const auto result = cold.Wait();
    Expect(result.prefill_tokens == 2048,
           "adaptive chunking consumes every token");
    Expect(result.max_prefill_chunk_tokens == (adaptive ? 100U : 512U),
           "measured prefill uses 100ms estimate only when explicitly enabled");
    ExpectMilliseconds(result.max_prefill_chunk_ms, adaptive ? 100 : 512,
                       "chunk wall metric records actual measured work");
  }
  auto control = std::make_shared<FakeControl>();
  control->clock = std::make_shared<FakeClock>();
  control->block_advance_label = 1;
  auto scheduler = MakeScheduler(control, 2, {.adaptive_chunking = true},
                                 {.max_decode_steps_per_burst = 2});
  auto warm =
      scheduler->Submit({1}, 10, 0.0F, {}, false, TimedMetadata(control));
  control->WaitForAdvance(1);
  auto cold = scheduler->Submit(std::vector<TextRunnerToken>(1024, 2), 1, 0.0F,
                                {}, false, TimedMetadata(control));
  control->ReleaseAdvance();
  (void)warm.Wait();
  Expect(cold.Wait().max_prefill_chunk_tokens == 512,
         "zero-time calibration falls back to existing 512-token geometry");
}

void TestAdaptivePrefillFloorAndMtpBurstOvershoot() {
  auto control = std::make_shared<FakeControl>();
  control->clock = std::make_shared<FakeClock>();
  control->prefill_token_ms = 50;
  control->advance_ms = 25;
  control->multi_token_decode = true;
  control->block_advance_label = 1;
  auto scheduler =
      MakeScheduler(control, 2,
                    {.decode_active_tokens = 8,
                     .min_chunk_tokens = 4,
                     .adaptive_chunking = true},
                    {.decode_burst = std::chrono::milliseconds{100}});
  auto warm =
      scheduler->Submit({1}, 90, 0.0F, {}, false, TimedMetadata(control));
  control->WaitForAdvance(1);
  auto cold = scheduler->Submit(std::vector<TextRunnerToken>(11, 2), 1, 0.0F,
                                {}, false, TimedMetadata(control));
  control->ReleaseAdvance();
  (void)warm.Wait();
  const auto result = cold.Wait();
  Expect(result.prefill_chunks == 3 && result.max_prefill_chunk_tokens == 4,
         "adaptive floor is honored, with a smaller final tail");
  ExpectMilliseconds(
      result.max_prefill_chunk_ms, 200,
      "floor overshoot is observable, not a false 100ms guarantee");
  const auto events = control->Events();
  const auto first = EventIndex(events, EventKind::kPrefill, 2);
  Expect(EventIndex(events, EventKind::kAdvance, 1, 5) < first &&
             first < EventIndex(events, EventKind::kAdvance, 1, 6),
         "MTP yields after indivisible steps cross 100ms, not one token at a "
         "time");
}

void TestCancellationDuringDecodeRetainsWorkAndWallTiming() {
  for (const bool multi : {false, true}) {
    auto control = std::make_shared<FakeControl>();
    control->clock = std::make_shared<FakeClock>();
    control->prefill_token_ms = 5;
    control->advance_ms = 10;
    control->multi_token_decode = multi;
    control->block_advance_label = 1;
    auto scheduler = MakeScheduler(control, 1);
    auto request =
        scheduler->Submit({1}, 9, 0.0F, {}, false, TimedMetadata(control));
    control->WaitForAdvance(1);
    request.Cancel();
    control->ReleaseAdvance();
    const auto result = request.Wait();
    Expect(result.cancelled, "decode cancellation stays a separate outcome");
    ExpectMilliseconds(result.decode_ms, multi ? 30 : 10,
                       "cancellation retains already executed model work");
    ExpectMilliseconds(result.total_generation_wall_ms, multi ? 35 : 15,
                       "cancelled total includes prefill and executed decode");
  }
}

void TestShortAdmissionHasBoundedBypassAndAging() {
  for (const bool aged : {false, true}) {
    auto control = std::make_shared<FakeControl>();
    control->clock = std::make_shared<FakeClock>();
    control->block_advance_label = 1;
    auto scheduler =
        MakeScheduler(control, 1, {},
                      {.prefer_short_prefill = true,
                       .max_admission_bypasses = 2,
                       .admission_aging = std::chrono::milliseconds{100}});
    auto active =
        scheduler->Submit({1}, 1, 0.0F, {}, false, TimedMetadata(control));
    control->WaitForAdvance(1);
    auto old = scheduler->Submit(std::vector<TextRunnerToken>(16, 2), 1, 0.0F,
                                 {}, false, TimedMetadata(control, "old", 16));
    if (aged)
      control->clock->Advance(100);
    std::vector<TextGenerationScheduler::Request> shorts;
    for (TextRunnerToken label : {3, 4, 5})
      shorts.push_back(
          scheduler->Submit({label}, 1, 0.0F, {}, false,
                            TimedMetadata(control, std::to_string(label), 1)));
    control->ReleaseAdvance();
    (void)active.Wait();
    (void)old.Wait();
    for (auto& request : shorts)
      (void)request.Wait();
    std::vector<TextRunnerToken> order;
    for (const auto& event : control->Events())
      if (event.kind == EventKind::kPrefill)
        order.push_back(event.label);
    Expect(
        order == (aged ? std::vector<TextRunnerToken>{1, 2, 3, 4, 5}
                       : std::vector<TextRunnerToken>{1, 3, 4, 2, 5}),
        "old work runs after two bypasses or immediately at its aging bound");
  }
}

void TestUnknownCacheEstimateIsNotColdHardRejection() {
  auto control = std::make_shared<FakeControl>();
  control->clock = std::make_shared<FakeClock>();
  control->block_advance_label = 1;
  auto scheduler =
      MakeScheduler(control, 1, {}, {.max_inflight_prefill_tokens = 4});
  auto active = scheduler->Submit({1}, 1, 0.0F, {}, false,
                                  TimedMetadata(control, "active", 1));
  control->WaitForAdvance(1);
  auto unknown =
      scheduler->Submit(std::vector<TextRunnerToken>(100, 2), 1, 0.0F, {},
                        false, TimedMetadata(control, "unknown"));
  auto known = scheduler->Submit({3, 3, 3}, 1, 0.0F, {}, false,
                                 TimedMetadata(control, "known", 3));
  bool rejected = false;
  try {
    (void)scheduler->Submit({4, 4}, 1, 0.0F, {}, false,
                            TimedMetadata(control, "overflow", 2));
  } catch (const TextGenerationError& error) {
    rejected = error.code() == TextGenerationErrorCode::kPrefillBudgetFull &&
               error.http_status() == 429 && error.retryable();
  }
  Expect(rejected,
         "known outstanding prefill estimate is bounded before admission");
  known.Cancel();
  control->ReleaseAdvance();
  (void)active.Wait();
  Expect(known.Wait().cancelled, "cancelled estimate releases reservation");
  const auto result = unknown.Wait();
  Expect(
      !result.estimated_prefill_tokens &&
          result.effective_prefill_tokens == 100,
      "unknown eligibility remains unknown until safe actual cache admission");
  const auto replacement = scheduler
                               ->Submit({5, 5, 5, 5}, 1, 0.0F, {}, false,
                                        TimedMetadata(control, "new", 4))
                               .Wait();
  Expect(replacement.prefill_tokens == 4,
         "completed and cancelled work releases tokens");
}

void TestInflightClientLimitCountsResidentAndPending() {
  auto control = std::make_shared<FakeControl>();
  control->clock = std::make_shared<FakeClock>();
  control->block_advance_label = 1;
  auto scheduler =
      MakeScheduler(control, 2, {}, {.max_inflight_requests_per_client = 2});
  auto first =
      scheduler->Submit({1}, 1, 0.0F, {}, false, TimedMetadata(control, "a"));
  control->WaitForAdvance(1);
  auto queued =
      scheduler->Submit({2}, 1, 0.0F, {}, false, TimedMetadata(control, "a"));
  bool rejected = false;
  try {
    (void)scheduler->Submit({3}, 1, 0.0F, {}, false,
                            TimedMetadata(control, "a"));
  } catch (const TextGenerationError& error) {
    rejected = error.code() == TextGenerationErrorCode::kClientInflightFull;
  }
  auto other =
      scheduler->Submit({4}, 1, 0.0F, {}, false, TimedMetadata(control, "b"));
  queued.Cancel();
  control->ReleaseAdvance();
  (void)first.Wait();
  const auto cancelled = queued.Wait();
  const auto result = other.Wait();
  Expect(rejected && cancelled.cancelled && result.completion_tokens == 1,
         "client inflight cap cannot be evaded by acquiring a resident slot");
  Expect(
      cancelled.resident_requests_at_submit == 1 &&
          cancelled.inflight_requests_at_submit == 2 &&
          cancelled.client_inflight_requests_at_submit == 2 &&
          result.inflight_requests_at_submit == 3 &&
          result.queue_depth_at_submit == 2 &&
          result.requested_logical_concurrency == 2 &&
          result.physical_execution_width == 1,
      "task backlog, resident slots and physical width are distinct metrics");
  Expect(result.request_id != cancelled.request_id && cancelled.request_id != 0,
         "requests have content-free correlation IDs");
  (void)scheduler->Submit({5}, 1, 0.0F, {}, false, TimedMetadata(control, "a"))
      .Wait();
}

void TestCancellationBeforeAdmissionRecordsElapsedWait() {
  auto control = std::make_shared<FakeControl>();
  control->clock = std::make_shared<FakeClock>();
  control->block_advance_label = 1;
  auto scheduler = MakeScheduler(control, 1);
  auto active =
      scheduler->Submit({1}, 1, 0.0F, {}, false, TimedMetadata(control));
  control->WaitForAdvance(1);
  auto cancelled =
      scheduler->Submit({2}, 1, 0.0F, {}, false, TimedMetadata(control));
  control->clock->Advance(75);
  cancelled.Cancel();
  control->ReleaseAdvance();
  (void)active.Wait();
  const auto result = cancelled.Wait();
  ExpectMilliseconds(result.queue_ms, 75,
                     "legacy queue field includes cancelled wait");
  ExpectMilliseconds(result.queue_admission_ms, 75,
                     "cancelled admission wait is not zero");
  ExpectMilliseconds(result.total_generation_wall_ms, 75,
                     "cancelled end-to-end time is retained");
  Expect(!result.first_token_emitted && result.ttft_ms == 0 &&
             result.resident_wait_ms == 0 && result.decode_wall_ms == 0 &&
             result.resident_requests_at_admission == 0,
         "cancellation without a token is not advertised as a zero-latency "
         "success");
}

void TestAdmissionExpiryIsRetryableAndDoesNotExpireResidents() {
  for (const auto timeout :
       {std::chrono::milliseconds{0}, std::chrono::milliseconds{60}}) {
    auto control = std::make_shared<FakeControl>();
    control->clock = std::make_shared<FakeClock>();
    control->block_advance_label = 1;
    auto scheduler = MakeScheduler(control, 1, {}, {.queue_timeout = timeout});
    auto active =
        scheduler->Submit({1}, 2, 0.0F, {}, false, TimedMetadata(control));
    control->WaitForAdvance(1);
    auto queued =
        scheduler->Submit({2}, 1, 0.0F, {}, false, TimedMetadata(control));
    control->clock->Advance(60);
    control->ReleaseAdvance();
    bool expired = false;
    try {
      (void)queued.Wait();
    } catch (const TextGenerationError& error) {
      expired = error.code() == TextGenerationErrorCode::kAdmissionTimeout &&
                error.http_status() == 429 && error.retryable();
    }
    Expect(expired == (timeout.count() != 0),
           "queue expiry is exact, retryable and opt-in");
    Expect(active.Wait().completion_tokens == 2,
           "queue timeout never kills admitted work");
    const auto events = control->Events();
    Expect((EventIndex(events, EventKind::kPrefill, 2) == events.size()) ==
               expired,
           "expired queue entries never run inference");
  }
}

void TestSerialAndBatchedDecodeTimingExcludesPeerPreparation() {
  for (const bool batched : {false, true}) {
    auto control = std::make_shared<FakeControl>();
    control->clock = std::make_shared<FakeClock>();
    control->advance_ms = 10;
    control->block_advance_label = 1;
    control->supports_batched_advance = batched;
    auto scheduler = MakeScheduler(control, 2);
    auto first =
        scheduler->Submit({1}, 10, 0.0F, {}, false, TimedMetadata(control));
    control->WaitForAdvance(1);
    auto second =
        scheduler->Submit({2}, 5, 0.0F, {}, false, TimedMetadata(control));
    control->ReleaseAdvance();
    const auto a = first.Wait();
    const auto b = second.Wait();
    if (!batched) {
      ExpectMilliseconds(a.decode_ms, 100,
                         "serial active work excludes peer decode");
      ExpectMilliseconds(b.decode_ms, 50,
                         "serial active work charges only own decode");
    }
    Expect(a.decode_wall_ms >= a.decode_ms && b.decode_wall_ms >= b.decode_ms,
           "each request's active work fits inside its resident decode wall");
  }
}

void TestCheckpointVectorAndCancellationMetricsPropagate() {
  for (const bool cancel : {false, true}) {
    auto control = std::make_shared<FakeControl>();
    control->clock = std::make_shared<FakeClock>();
    control->prefill_capacity = 2;
    control->prefix_checkpoint_alignment = 2;
    std::binary_semaphore captured(0), release(0);
    std::atomic<std::size_t> snapshots{0};
    control->snapshot_callback = [&] {
      if (snapshots.fetch_add(1) == 0) {
        captured.release();
        release.acquire();
      }
    };
    auto scheduler = MakeScheduler(control, 1);
    auto metadata = TimedMetadata(control);
    metadata.cache_boundaries = {
        {gufo::server::TextCacheBoundaryKind::kSystemEnd, 2},
        {gufo::server::TextCacheBoundaryKind::kFullPrompt, 4}};
    auto request =
        scheduler->Submit({1, 10, 11, 12}, 1, 0.0F, {}, false, metadata);
    const bool reached = captured.try_acquire_for(kTestTimeout);
    if (reached) {
      // Queue processing proves the scheduler has parked the frozen resident;
      // the capture worker's entry alone can race that transition.
      auto probe = scheduler->Submit(
          {9}, 1, 0.0F, [] { return true; }, false,
          TimedMetadata(control, "probe"));
      Expect(probe.Wait().cancelled, "probe synchronizes the parked snapshot");
    }
    if (cancel)
      request.Cancel();
    control->clock->Advance(30);
    release.release();
    Expect(reached, "scheduler vector metadata reaches exact prefix capture");
    const auto result = request.Wait();
    Expect(result.cancelled == cancel &&
               result.prefix_checkpoint_kind == "system_end" &&
               result.prefix_checkpoint_tokens == 2 &&
               result.prefix_checkpoint_bytes > 0 &&
               result.prefix_checkpoint_failures == 0,
           "checkpoint metrics propagate on normal and cancelled completion");
    ExpectMilliseconds(result.snapshot_wait_ms, 30,
                       "snapshot pause is not resident ready wait");
    ExpectMilliseconds(result.resident_wait_ms, 0,
                       "isolated capture has no peer wait");
  }
}

void TestAdmissionTimingExcludesCacheRestore() {
  auto control = std::make_shared<FakeControl>();
  control->clock = std::make_shared<FakeClock>();
  control->prefill_token_ms = 2;
  control->advance_ms = 10;
  control->restore_ms = 40;
  control->snapshot_callback = [] {};
  auto scheduler = MakeScheduler(control, 1);
  (void)scheduler->Submit({1, 10}, 1, 0.0F, {}, false, TimedMetadata(control))
      .Wait();
  const auto hit =
      scheduler->Submit({1, 10}, 1, 0.0F, {}, false, TimedMetadata(control))
          .Wait();
  Expect(
      hit.cache_hit && hit.effective_prefill_tokens == 0,
      "cache hit reports actual zero prefill rather than full prompt demand");
  ExpectMilliseconds(hit.queue_admission_ms, 0,
                     "cache restore is not queue residence");
  ExpectMilliseconds(hit.total_generation_wall_ms, 50,
                     "restore remains in end-to-end latency");
  ExpectMilliseconds(hit.ttft_ms, 40,
                     "first token includes cache restore latency");
  ExpectMilliseconds(hit.decode_wall_ms, 10,
                     "decode wall starts at ready frontier");
}

void TestGenerationMetricsExplicitlySeparateActiveAndWall() {
  TextGenerationScheduler::Result result;
  result.completion_tokens = 10;
  result.prefill_tokens = 20;
  result.prefill_ms = 20;
  result.decode_ms = 100;
  result.decode_wall_ms = 400;
  result.total_generation_wall_ms = 1000;
  result.queue_ms = result.queue_admission_ms = 200;
  result.resident_wait_ms = 280;
  result.peer_prefill_ms = 250;
  result.first_token_emitted = true;
  result.ttft_ms = 600;
  result.text = "MUST_NOT_LOG_PRIVATE_OUTPUT";
  ExpectMilliseconds(gufo::server::ActiveDecodeTokensPerSecond(result), 100,
                     "active TPS");
  ExpectMilliseconds(gufo::server::DecodeWallTokensPerSecond(result), 25,
                     "decode wall TPS");
  ExpectMilliseconds(gufo::server::GenerationWallTokensPerSecond(result), 10,
                     "end-to-end TPS");
  const auto log = gufo::server::GenerationLogDetails(result);
  Expect(log.find(" active_decode_tps=100.0") != std::string::npos &&
             log.find(" decode_wall_tps=25.0") != std::string::npos &&
             log.find(" generation_wall_tps=10.0") != std::string::npos &&
             log.find(" decode_tps=") == std::string::npos &&
             log.find(result.text) == std::string::npos,
         "logs label active TPS and contain no generated content");
  const auto timings = gufo::server::GenerationTimings(result).dump();
  Expect(timings.find("\"predicted_time_basis\":\"active_decode\"") !=
                 std::string::npos &&
             timings.find("\"decode_wall_ms\":400") != std::string::npos &&
             timings.find("\"queue_admission_ms\":200") != std::string::npos,
         "legacy active timings are labeled beside new honest wall metrics");
  result = {};
  Expect(gufo::server::DecodeWallTokensPerSecond(result) == 0 &&
             gufo::server::GenerationWallTokensPerSecond(result) == 0,
         "absent timing never invents throughput");
}

int main() {
  TestCandidatePoliciesAreOptInAndValidated();
  TestSustainedColdPrefillAndWarmDecode();
  TestZeroCostDecodeBurstStillYields();
  TestCachedReplacementsCannotStarvePrefill();
  TestMeasuredPrefillBudgetAndGeometryFallback();
  TestAdaptivePrefillFloorAndMtpBurstOvershoot();
  TestCancellationDuringDecodeRetainsWorkAndWallTiming();
  TestShortAdmissionHasBoundedBypassAndAging();
  TestStickyAdmissionPrefersResidentLineage();
  TestUnknownCacheEstimateIsNotColdHardRejection();
  TestInflightClientLimitCountsResidentAndPending();
  TestCancellationBeforeAdmissionRecordsElapsedWait();
  TestAdmissionExpiryIsRetryableAndDoesNotExpireResidents();
  TestGenerationMetricsExplicitlySeparateActiveAndWall();
  TestAdmissionTimingExcludesCacheRestore();
  TestCheckpointVectorAndCancellationMetricsPropagate();
  TestSerialAndBatchedDecodeTimingExcludesPeerPreparation();
  TestJsonMetadataAndConcurrentCursors();
  TestJsonAdmissionRejectsBeforeInference();
  TestCapturesAtCapacityAllowQueuedProgress();
  TestShutdownCancelsRunnerAcquisition();
  TestSnapshotDoesNotBlockOtherRequests();
  TestFirstTokenPrecedesSnapshotAndPreservesBudget();
  TestIdlePrefillUsesBulkWorkUnit();
  TestRunnerCanSkipUnusedFinalAdvance();
  TestRunnerCanReuseExactIncrementalText();
  TestMultiTokenDecodePublishesDraftMetricsAndDisablesPrefixReuse();
  TestMultiTokenRunnerCanSwitchToBatchedExecution();
  TestModelOwnedBatchMetrics();
  TestBatchFailureIsolation();
  TestMultiResidentPrefillUsesBoundedWorkUnits();
  for (const bool multi_token : {false, true}) {
    for (const bool batched : {false, true}) {
      if (batched && !multi_token)
        continue;
      TestDecodeActivePrefillIsBounded(multi_token, batched);
      TestPrefillYieldsToEveryDueDecoder(multi_token, batched);
    }
  }
  TestNonIncrementalRunnerFallsBackSafely();
  TestPendingLimitsRejectBeforeStateAdmission();
  TestPendingClientsAreRoundRobinAndIndividuallyBounded();
  TestExpiredQueuedRequestNeverConsumesState();
  TestSlowConsumerOutputIsBoundedAndReclaimed();
  TestGeneratedOutputLimitAppliesWithoutStreaming();
  TestMidGenerationAdmissionAndIsolatedTrajectories();
  TestMidGenerationRequestJoinsNextDecodeBatch();
  TestFifoReplacementAdmissionWithOneSlot();
  TestQueuedAndPrefillCancellation();
  TestDecodeCancellationAndStateReclamation();
  TestFourResidentRequestsMakeProgress();
  TestRunnerFailureInvalidatesAndDoesNotPoisonReplacement();
  std::cout << "All text generation scheduler tests passed\n";
  return 0;
}
