#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "src/core/sampling.hpp"
#include "tests/models/qwen27b/sampling_cases.hpp"

namespace {

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

void TestGreedySelectsFiniteArgmaxWithoutAdvancingRng() {
  const std::array<float, 4> logits = {
      -1.0F, 5.0F, std::numeric_limits<float>::quiet_NaN(), 2.0F};
  std::uint64_t rng_state = 1234;

  const auto token = gufo::sampling::SampleLogits(logits, 0.0F, &rng_state);

  Expect(token == 1, "greedy sampling selects the finite argmax");
  Expect(rng_state == 1234, "greedy sampling does not consume RNG state");
}

void TestDefaultConfigPreservesGreedyDecoding() {
  const std::array<float, 4> logits = {-1.0F, 5.0F, 3.0F, 2.0F};
  gufo::sampling::SamplerState sampler;

  const auto before = sampler.rng_state();
  const auto token = sampler.Sample(logits);

  Expect(token == 1, "default sampler is greedy");
  Expect(sampler.rng_state() == before,
         "default sampler does not consume RNG state");
}

void TestTemperatureSamplingIsDeterministicAndNonGreedy() {
  const std::array<float, 3> logits = {0.0F, 0.0F, 0.0F};
  bool saw_first = false;
  bool saw_non_first = false;

  for (std::uint64_t seed = 1; seed <= 128; ++seed) {
    auto left_state = seed;
    auto right_state = seed;
    const auto left = gufo::sampling::SampleLogits(logits, 1.0F, &left_state);
    const auto right = gufo::sampling::SampleLogits(logits, 1.0F, &right_state);
    Expect(left == right && left_state == right_state,
           "temperature sampling is reproducible from the same RNG state");
    saw_first = saw_first || left == 0;
    saw_non_first = saw_non_first || left != 0;
  }

  Expect(saw_first && saw_non_first,
         "temperature sampling does not collapse to greedy argmax");
}

void TestTopKFiltersTheCandidateSet() {
  const std::array<float, 5> logits = {5.0F, 4.0F, 3.0F, 2.0F, 1.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 1.0F;
  config.top_k = 2;

  const auto distribution = gufo::sampling::BuildDistribution(logits, config);

  Expect(distribution.entries().size() == 2,
         "top-k retains exactly k candidates");
  Expect(distribution.probability(0) > 0.0 && distribution.probability(1) > 0.0,
         "top-k retains the highest-logit candidates");
  Expect(distribution.probability(2) == 0.0,
         "top-k removes lower-logit candidates");
}

void TestTopPFiltersByCumulativeProbability() {
  const std::array<float, 3> logits = {3.0F, 2.0F, 1.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 1.0F;
  config.top_p = 0.7F;

  const auto distribution = gufo::sampling::BuildDistribution(logits, config);

  Expect(distribution.entries().size() == 2,
         "top-p retains the smallest cumulative-probability prefix");
  Expect(distribution.probability(2) == 0.0,
         "top-p removes the low-probability tail");
}

void TestMinPFiltersRelativeToTheBestToken() {
  const std::array<float, 3> logits = {3.0F, 2.4F, 1.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 1.0F;
  config.min_p = 0.5F;

  const auto distribution = gufo::sampling::BuildDistribution(logits, config);

  Expect(distribution.entries().size() == 2,
         "min-p retains candidates above the relative threshold");
  Expect(distribution.probability(2) == 0.0,
         "min-p removes candidates far below the best token");
}

void TestMinKeepProvidesAFilterFloor() {
  const std::array<float, 4> logits = {5.0F, 4.0F, 3.0F, 2.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 1.0F;
  config.top_k = 1;
  config.top_p = 0.1F;
  config.min_p = 0.99F;
  config.min_keep = 3;

  const auto distribution = gufo::sampling::BuildDistribution(logits, config);

  Expect(distribution.entries().size() == 3,
         "min-keep prevents enabled filters from shrinking below its floor");
}

void TestTemperatureBeforeProbabilityFilters() {
  // Independent closed-form expectations for softmax([0,-1,-2] / T).
  // Also exercise the bounded (>1024 vocabulary) candidate selector.
  for (const std::size_t size : {3U, 2048U}) {
    std::vector<float> logits(size, -1000.0F);
    logits[0] = 0.0F;
    logits[1] = -1.0F;
    logits[2] = -2.0F;
    for (const bool nucleus : {false, true}) {
      gufo::sampling::SamplingConfig config{.temperature = 0.5F, .seed = 42};
      if (nucleus)
        config.top_p = 0.8F;
      else
        config.min_p = 0.2F;
      gufo::sampling::SamplerState sampler(config);
      for (const auto distribution :
           {gufo::sampling::BuildDistribution(logits, config),
            sampler.Distribution(logits)}) {
        Expect(distribution.entries().size() == 1 &&
                   distribution.probability(0) == 1.0,
               "temperature sharpens probabilities before filtering");
      }
      for (unsigned draw = 0; draw < 32; ++draw)
        Expect(sampler.Sample(logits) == 0,
               "ordinary decode uses the tempered candidate set");

      config.temperature = 2.0F;
      config.top_p = nucleus ? 0.8F : 1.0F;
      config.min_p = nucleus ? 0.0F : 0.5F;
      const double first = 1.0 / (1.0 + std::exp(-0.5));
      for (const auto distribution :
           {gufo::sampling::BuildDistribution(logits, config),
            gufo::sampling::SamplerState(config).Distribution(logits)}) {
        Expect(
            distribution.entries().size() == 2 &&
                std::abs(distribution.probability(0) - first) < 1e-12 &&
                std::abs(distribution.probability(1) - (1.0 - first)) < 1e-12,
            "temperature flattens probabilities before filtering");
      }
    }
  }
}

void TestRepeatPenaltyUsesCommittedHistory() {
  const std::array<float, 3> logits = {2.0F, 1.0F, 0.0F};
  const std::array<gufo::sampling::TokenId, 1> history = {0};
  gufo::sampling::SamplingConfig config;
  config.repeat_penalty = 3.0F;

  const auto distribution =
      gufo::sampling::BuildDistribution(logits, config, history);

  Expect(distribution.best_token() == 1,
         "repeat penalty can demote a recently used token");
}

void TestFrequencyAndPresencePenaltiesUseCounts() {
  const std::array<float, 2> logits = {2.0F, 1.4F};
  const std::array<gufo::sampling::TokenId, 2> history = {0, 0};

  gufo::sampling::SamplingConfig frequency_config;
  frequency_config.frequency_penalty = 0.4F;
  const auto frequency_distribution =
      gufo::sampling::BuildDistribution(logits, frequency_config, {}, history);
  Expect(frequency_distribution.best_token() == 1,
         "frequency penalty scales with occurrence count");

  gufo::sampling::SamplingConfig presence_config;
  presence_config.presence_penalty = 0.7F;
  const auto presence_distribution =
      gufo::sampling::BuildDistribution(logits, presence_config, {}, history);
  Expect(presence_distribution.best_token() == 1,
         "presence penalty applies once when a token is present");
}

void TestRepeatWindowIsBounded() {
  const std::array<float, 2> logits = {2.0F, 2.1F};
  const std::array<gufo::sampling::TokenId, 2> history = {0, 1};
  gufo::sampling::SamplingConfig config;
  config.repeat_penalty = 2.0F;
  config.repeat_last_n = 1;

  const auto distribution =
      gufo::sampling::BuildDistribution(logits, config, history);

  Expect(distribution.best_token() == 0,
         "repeat-last-n excludes tokens outside the configured window");
}

void TestFastGreedyMatchesNormalizedPenaltyRoute() {
  const std::array<float, 5> logits = {3.0F, 2.9F, 2.8F, 2.7F, 2.6F};
  const std::array<gufo::sampling::TokenId, 4> history = {0, 0, 1, 3};
  gufo::sampling::SamplingConfig config;
  config.repeat_penalty = 1.5F;
  config.frequency_penalty = 0.2F;
  config.presence_penalty = 0.1F;
  gufo::sampling::SamplerState sampler(config, history);

  const auto expected =
      gufo::sampling::BuildDistribution(logits, config, history).best_token();
  const auto actual = sampler.Sample(logits);

  Expect(actual == expected,
         "linear greedy route matches the normalized penalty route");
}

void TestFastTopKSamplingNeverEscapesSelectedSet() {
  const std::array<float, 6> logits = {8.0F, 7.0F, 1.0F, 0.0F, -1.0F, -2.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 1.0F;
  config.top_k = 2;

  bool saw_first = false;
  bool saw_second = false;
  for (std::int64_t seed = 1; seed <= 256; ++seed) {
    config.seed = seed;
    gufo::sampling::SamplerState sampler(config);
    const auto token = sampler.Sample(logits);
    Expect(token <= 1, "top-k fast route retains only selected candidates");
    saw_first = saw_first || token == 0;
    saw_second = saw_second || token == 1;
  }
  Expect(saw_first && saw_second,
         "top-k fast route samples the retained distribution");
}

void TestFastMinPSamplingNeverEscapesRelativeThreshold() {
  const std::array<float, 4> logits = {3.0F, 2.5F, 1.0F, -4.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 1.0F;
  config.min_p = 0.5F;

  for (std::int64_t seed = 1; seed <= 256; ++seed) {
    config.seed = seed;
    gufo::sampling::SamplerState sampler(config);
    Expect(sampler.Sample(logits) <= 1,
           "min-p linear route rejects tokens below the relative threshold");
  }
}

void TestSeedAndStateAreRequestLocal() {
  const std::array<float, 3> logits = {0.0F, 0.0F, 0.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 1.0F;
  config.seed = 42;
  config.repeat_last_n = 2;
  gufo::sampling::SamplerState left(config);
  gufo::sampling::SamplerState right(config);

  for (int sample = 0; sample < 16; ++sample) {
    Expect(left.Sample(logits) == right.Sample(logits),
           "equal seeds produce equal token sequences");
  }
  left.Accept(1);
  left.Accept(2);
  left.Accept(0);
  Expect(left.history().size() == 2 && left.history()[0] == 2 &&
             left.history()[1] == 0,
         "sampler retains only its bounded committed history");
  Expect(right.history().empty(), "sampler histories are request-local");
}

void TestDistributionIsNormalized() {
  const std::array<float, 4> logits = {2.0F, 1.0F, 0.0F, -1.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 0.8F;
  const auto distribution = gufo::sampling::BuildDistribution(logits, config);
  double sum = 0.0;
  for (const auto& entry : distribution.entries()) {
    sum += entry.value;
  }
  Expect(std::abs(sum - 1.0) < 1e-12,
         "post-filter probabilities are normalized");
}

void TestSamplingFailsClosedOnInvalidInputs() {
  bool empty_rejected = false;
  try {
    std::uint64_t rng_state = 1;
    (void)gufo::sampling::SampleLogits({}, 1.0F, &rng_state);
  } catch (const std::invalid_argument&) {
    empty_rejected = true;
  }
  Expect(empty_rejected, "empty logits are rejected");

  bool missing_rng_rejected = false;
  try {
    const std::array<float, 2> logits = {0.0F, 1.0F};
    (void)gufo::sampling::SampleLogits(logits, 1.0F, nullptr);
  } catch (const std::invalid_argument&) {
    missing_rng_rejected = true;
  }
  Expect(missing_rng_rejected, "sampled decoding requires RNG state");

  bool non_finite_rejected = false;
  try {
    const std::array<float, 2> logits = {
        std::numeric_limits<float>::quiet_NaN(),
        -std::numeric_limits<float>::infinity()};
    std::uint64_t rng_state = 1;
    (void)gufo::sampling::SampleLogits(logits, 1.0F, &rng_state);
  } catch (const std::runtime_error&) {
    non_finite_rejected = true;
  }
  Expect(non_finite_rejected, "an all-non-finite distribution is rejected");

  bool invalid_config_rejected = false;
  try {
    gufo::sampling::SamplingConfig config;
    config.top_p = 0.0F;
    config.Validate();
  } catch (const std::invalid_argument&) {
    invalid_config_rejected = true;
  }
  Expect(invalid_config_rejected, "invalid sampling controls are rejected");
}

void TestZeroDrawAndNonFiniteCandidates() {
  // Inverse xorshift state whose next 53-bit uniform is exactly zero.
  constexpr std::uint64_t zero_draw_state = UINT64_C(0x98d76a164d99a710);
  const std::array<float, 3> cold_logits{-1000.0F, 0.0F, -1.0F};
  gufo::sampling::SamplerState sampler({.temperature = 1.0F, .seed = 0});
  sampler.SetRngState(zero_draw_state);
  Expect(sampler.Sample(cold_logits) == 1,
         "zero draw must not select underflowed zero probability");
  const std::array<float, 5> masked_logits{
      std::numeric_limits<float>::quiet_NaN(), 4.0F,
      std::numeric_limits<float>::infinity(), 0.0F,
      -std::numeric_limits<float>::infinity()};
  for (const auto seed : {0, 73, 808}) {
    gufo::sampling::SamplerState masked({.temperature = 1.0F, .seed = seed});
    auto rng = masked.rng_state();
    const auto expected =
        gufo::sampling::BuildDistribution(masked_logits, masked.config())
            .Sample(&rng);
    Expect(masked.Sample(masked_logits) == expected,
           "linear CPU sampling must ignore nonfinite candidates");
  }
}

void TestStrategyReplayAgainstReference() {
  const std::array<float, 5> logits{0.4F, 2.0F, -1.0F, 1.1F, -0.3F};
  for (const auto& test : gufo::test::QwenSamplingCases()) {
    for (const auto seed : {0, 1, 73, 808}) {
      auto config = test.config;
      config.seed = seed;
      std::vector<gufo::sampling::TokenId> history{1, 1, 2, 3};
      gufo::sampling::SamplerState sampler(config, history);
      std::vector<gufo::sampling::TokenId> generated;
      auto replay = sampler;
      for (int step = 0; step < 16; ++step) {
        const auto distribution = gufo::sampling::BuildDistribution(
            logits, config, history, generated);
        std::vector<gufo::sampling::Probability> ordered(
            distribution.entries().begin(), distribution.entries().end());
        if (config.top_k == 0 && config.top_p == 1.0F &&
            !(config.min_p > 0.0F && config.min_keep > 1)) {
          std::ranges::sort(ordered, {}, &gufo::sampling::Probability::token);
        }
        auto rng = sampler.rng_state();
        double threshold = gufo::sampling::Uniform(&rng);
        auto expected = distribution.best_token();
        if (config.temperature > 0.0F) {
          for (const auto& entry : ordered) {
            if (entry.value > 0.0 && threshold < entry.value) {
              expected = entry.token;
              break;
            }
            threshold -= entry.value;
          }
        }
        const auto token = sampler.Sample(logits);
        Expect(token == expected && token == replay.Sample(logits), test.name);
        generated.push_back(token);
        sampler.Accept(token);
        replay.Accept(token);
      }
    }
  }
}

void TestFilteredDistributionAgainstFullSort() {
  std::vector<float> logits(4099);
  const std::array<gufo::sampling::TokenId, 6> history{8, 8, 8, 11, 17, 29};
  for (bool concentrated : {false, true}) {
    for (std::size_t i = 0; i < logits.size(); ++i)
      logits[i] = concentrated ? -(float)((i * 73) % logits.size()) * 0.2F
                               : std::sin((float)i * 0.7F);
    for (const auto& test : gufo::test::QwenSamplingCases()) {
      gufo::sampling::SamplerState sampler(test.config, history);
      const auto rng = sampler.rng_state();
      const auto expected =
          gufo::sampling::BuildDistribution(logits, test.config, history);
      const auto actual = sampler.Distribution(logits);
      const auto positive = std::ranges::count_if(
          expected.entries(), [](const auto& e) { return e.value > 0; });
      Expect(actual.entries().size() == static_cast<std::size_t>(positive),
             test.name);
      for (const auto& entry : expected.entries())
        Expect(std::abs(actual.probability(entry.token) - entry.value) < 1e-12,
               test.name);
      Expect(sampler.rng_state() == rng,
             "materializing probabilities consumes no random draws");
    }
  }
}

void TestDeferredResidualReplay() {
  const std::array<float, 3> logits{100.0F, -100.0F, -100.0F};
  gufo::sampling::SamplerState sampler({.temperature = 1.0F, .seed = 7});
  // A retained residual must win over a fresh draw, even when that fresh
  // draw would almost certainly select a different token.
  sampler.DeferSample(2);
  const auto rng = sampler.rng_state();
  auto copied = sampler;
  gufo::sampling::SamplerState assigned;
  assigned = sampler;
  bool duplicate_rejected = false;
  try {
    sampler.DeferSample(1);
  } catch (const std::logic_error&) {
    duplicate_rejected = true;
  }
  Expect(duplicate_rejected, "pending residual cannot be overwritten");
  gufo::sampling::SamplerState published({.temperature = 1.0F});
  published.Accept(1);
  published.CopyDrawStateFrom(sampler);
  Expect(published.history().size() == 1 && published.history()[0] == 1,
         "publishing draw state leaves committed history unchanged");
  for (auto* state : {&sampler, &copied, &assigned, &published}) {
    Expect(state->Sample(logits) == 2 && state->rng_state() == rng,
           "copied residual is consumed without another RNG draw");
    Expect(state->Sample(logits) == 0 && state->rng_state() != rng,
           "residual is consumed only once");
  }
  sampler.DeferSample(2);
  sampler.ResetHistory({});
  Expect(sampler.Sample(logits) == 0, "new history discards pending residual");
  sampler.DeferSample(2);
  sampler.SetRngState(rng);
  Expect(sampler.Sample(logits) == 0, "RNG reset discards pending residual");
}

void TestCanonicalTargetDistribution() {
  std::vector<float> logits(4099, -1000.0F);
  logits[0] = 0;
  logits[17] = -1;
  logits[4098] = -2;
  for (const auto& test : gufo::test::QwenSamplingCases()) {
    for (float temperature : {0.5F, 0.99999994F, 2.0F}) {
      auto config = test.config;
      config.temperature = temperature;
      config.seed = 19;
      gufo::sampling::SamplerState sampler(config);
      for (unsigned step = 0; step < 16; ++step) {
        const auto distribution = sampler.Distribution(logits);
        auto rng = sampler.rng_state();
        const auto expected = distribution.Sample(&rng);
        Expect(sampler.Sample(logits) == expected && sampler.rng_state() == rng,
               "AR and verification share exact support, CDF and RNG draws");
        sampler.Accept(expected);
      }
    }
  }
}

void TestResponsePenaltyScope() {
  using gufo::sampling::SamplerState;
  using gufo::sampling::TokenId;
  const std::array<float, 3> logits{3, 2, 1};
  SamplerState sampler({.temperature = 1,
                        .seed = 7,
                        .repeat_last_n = 0,
                        .frequency_penalty = .5F,
                        .presence_penalty = .25F},
                       std::vector<TokenId>{0, 0, 0});
  Expect(sampler.Distribution(logits).best_token() == 0,
         "frequency/presence must not penalize prompt tokens");
  sampler.Accept(std::vector<TokenId>{0, 0});
  sampler.Accept(std::vector<TokenId>(80, 2));
  const std::array<float, 3> expected_logits{1.75F, 2, -39.25F};
  const auto expected =
      gufo::sampling::BuildDistribution(expected_logits, {.temperature = 1});
  for (const auto token : {0U, 1U, 2U})
    Expect(std::abs(sampler.Distribution(logits).probability(token) -
                    expected.probability(token)) < 1e-12,
           "full response counts must outlive the repetition window");
  auto rejected = sampler;
  rejected.Accept(1);
  Expect(sampler.Distribution(logits).best_token() == 1 &&
             rejected.Distribution(logits).best_token() == 0,
         "tentative penalty counts must not leak through speculative rollback");
  auto replay = sampler;
  Expect(replay.Sample(logits) == sampler.Sample(logits),
         "penalty copies preserve RNG replay");
  sampler.ResetHistory(std::vector<TokenId>{0, 0});
  Expect(sampler.Distribution(logits).best_token() == 0,
         "new request resets generated counts");
}

std::shared_ptr<const gufo::sampling::TokenConstraint> JsonTokenConstraint() {
  using Constraint = gufo::sampling::TokenConstraint;
  auto vocabulary = std::make_shared<const Constraint::Vocabulary>(
      Constraint::Vocabulary{{"bad"},
                             {"{"},
                             {"}"},
                             {"", true, true},
                             {"", true, true},
                             {"{"},
                             {""},
                             {"{}", false, true},
                             {"\"x\":"},
                             {"1"}});
  return std::make_shared<const Constraint>(
      std::make_shared<const gufo::JsonConstraint>(
          gufo::JsonConstraint::JsonObject()),
      std::move(vocabulary));
}

void TestJsonConstrainedSelectionAndStops() {
  using gufo::sampling::SamplerState;
  std::array<float, 11> logits{100, 5, 4, 99, 98, 5, 97, 96, 3, 2, 95};
  const auto original = logits;
  SamplerState sampler({.seed = 19}, {}, JsonTokenConstraint());
  const auto rng = sampler.rng_state();
  Expect(sampler.has_constraint() && !sampler.constraint_complete(),
         "new JSON cursor is incomplete");
  Expect(sampler.Sample(logits) == 1 && sampler.Sample(logits) == 1,
         "mask illegal, empty, special, early EOS/PAD and unknown IDs; tie is "
         "lowest ID");
  Expect(
      logits == original && sampler.rng_state() == rng &&
          sampler.history().empty(),
      "constrained selection does not mutate logits, grammar, history or RNG");
  sampler.Accept(1);
  Expect(!sampler.constraint_complete() && sampler.Sample(logits) == 2,
         "accept advances once and closing object becomes legal");
  sampler.Accept(2);
  Expect(sampler.constraint_complete() && sampler.Sample(logits) == 3,
         "EOS is available only for a complete JSON document");
  logits[4] = 101;
  Expect(sampler.Sample(logits) == 4, "PAD is available after completion");
  sampler.Accept(4);
  Expect(sampler.constraint_complete(), "accepted stop does not alter grammar");

  const std::array<gufo::sampling::TokenId, 1> prompt{1};
  SamplerState penalized({.repeat_penalty = 2}, prompt, JsonTokenConstraint());
  Expect(penalized.Sample(original) == 5,
         "constrained selection uses adjusted rather than raw logits");
  const std::array<float, 3> compact{5, 5, 100};
  const std::array<gufo::sampling::TokenId, 3> ids{5, 1, 0};
  SamplerState compact_sampler({}, {}, JsonTokenConstraint());
  Expect(compact_sampler.Distribution(compact, ids).best_token() == 1,
         "compact constrained logits map legal IDs and break ties by full "
         "token ID");
}

void TestJsonConstraintCopiesAndReset() {
  using gufo::sampling::SamplerState;
  const std::array<float, 10> logits{100, 5, 4, 99, 98, 5, 97, 96, 3, 2};
  const auto constraint = JsonTokenConstraint();
  SamplerState sampler({.seed = 7}, {}, constraint);
  sampler.Accept(1);
  sampler.DeferSample(2);
  auto preview = sampler;
  SamplerState assigned;
  assigned = sampler;
  for (auto* copy : {&preview, &assigned}) {
    Expect(copy->Sample(logits) == 2, "sampler copies retain deferred draw");
    copy->Accept(2);
    Expect(copy->constraint_complete(),
           "copies own independent grammar cursor");
  }
  Expect(!sampler.constraint_complete() && sampler.Sample(logits) == 2,
         "preview acceptance never advances the committed cursor");
  sampler.Accept(2);
  sampler.ResetHistory(std::vector<gufo::sampling::TokenId>{0, 2, 3});
  Expect(!sampler.constraint_complete() && sampler.Sample(logits) == 1,
         "reset restores grammar prototype and does not accept prompt tokens");
  SamplerState another({}, {}, constraint);
  Expect(another.Sample(logits) == 1,
         "shared immutable grammar prototype was never advanced");
  sampler.DeferSample(0);
  sampler.SetConstraint(constraint);
  Expect(sampler.Sample(logits) == 1,
         "attaching a grammar discards old pending draws");
  sampler.DeferSample(0);
  bool rejected = false;
  try {
    (void)sampler.Sample(logits);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Expect(rejected, "an invalid deferred token cannot bypass the grammar");
  sampler.ResetHistory({});
  rejected = false;
  try {
    sampler.Accept(std::vector<gufo::sampling::TokenId>{1, 0});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Expect(rejected && sampler.Sample(logits) == 1 && sampler.history().empty(),
         "invalid accepted sequence leaves cursor and token history unchanged");
}

void TestJsonConstraintFailsClosedAndPreservesBytes() {
  using gufo::sampling::SamplerState;
  for (const auto config :
       {gufo::sampling::SamplingConfig{.temperature = 1},
        gufo::sampling::SamplingConfig{.temperature = 1, .top_k = 1}}) {
    bool rejected = false;
    try {
      SamplerState sampler(config, {}, JsonTokenConstraint());
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    Expect(rejected, "all positive temperatures reject JSON constraints");
  }
  SamplerState sampler({}, {}, JsonTokenConstraint());
  for (const auto logits :
       {std::vector<float>(10, std::numeric_limits<float>::quiet_NaN()),
        std::vector<float>{100, -INFINITY, -INFINITY, 90, 90, -INFINITY, 90,
                           90}}) {
    bool rejected = false;
    try {
      (void)sampler.Sample(logits);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    Expect(rejected, "no finite legal candidate fails closed");
  }
  using Constraint = gufo::sampling::TokenConstraint;
  const auto utf8 = std::make_shared<const Constraint>(
      std::make_shared<const gufo::JsonConstraint>(
          gufo::JsonConstraint::Compile(R"({"type":"string"})")),
      std::make_shared<const Constraint::Vocabulary>(
          Constraint::Vocabulary{{"\""},
                                 {std::string(1, '\xc3')},
                                 {std::string(1, '\xa9')},
                                 {"", true, true},
                                 {std::string(1, '\xff')}}));
  SamplerState unicode({}, {}, utf8);
  unicode.Accept(0);
  unicode.Accept(1);
  const std::array<float, 5> logits{10, 9, 8, 100, 101};
  Expect(unicode.Sample(logits) == 2,
         "partial UTF-8 token bytes are not replaced");
  unicode.Accept(2);
  unicode.Accept(0);
  Expect(unicode.constraint_complete() && unicode.Sample(logits) == 3,
         "split UTF-8 completes as valid JSON before EOS");
}

void TestStructuredOutputPhases() {
  using Constraint = gufo::sampling::TokenConstraint;
  using gufo::sampling::SamplerState;
  auto vocabulary = std::make_shared<const Constraint::Vocabulary>(
      Constraint::Vocabulary{{"reason <tool_call>still thinking"},
                             {"</thi"},
                             {"nk>\n"},
                             {"{\"marker\":\"<tool_call></think>\"}"},
                             {"<tool_"},
                             {"call>"},
                             {"{\"name\":\"tool\",\"arguments\":{}}"},
                             {"</tool_"},
                             {"call>"},
                             {"", true, true},
                             {"</think>", false, true},
                             {"trailing"},
                             {"<tool_call>", false, true},
                             {"</tool_call>", false, true},
                             {"{}"},
                             {"</tool_call>trailing"}});
  auto prototype = std::make_shared<const gufo::JsonConstraint>(
      gufo::JsonConstraint::JsonObject());
  auto constraint = std::make_shared<const Constraint>(
      prototype, vocabulary,
      Constraint::Options{.starts_in_reasoning = true,
                          .allow_tool_calls = true});
  const auto expect_illegal = [&](const SamplerState& state,
                                  gufo::sampling::TokenId token) {
    auto copy = state;
    bool rejected = false;
    try {
      copy.Accept(token);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    Expect(rejected, "phase transition must reject illegal token");
  };
  SamplerState reason({}, {}, constraint);
  reason.Accept(0);
  Expect(!reason.constraint_complete(),
         "tool markers inside reasoning do not switch routes");
  expect_illegal(reason, 9);
  reason.Accept(1);
  auto preview = reason;
  SamplerState assigned;
  assigned = reason;
  preview.Accept(2);
  preview.Accept(3);
  Expect(preview.constraint_complete() && !reason.constraint_complete() &&
             !assigned.constraint_complete(),
         "preview and assignment deep-copy partial reasoning marker state");
  reason.Accept(2);
  expect_illegal(reason, 10);
  reason.Accept(3);
  Expect(reason.constraint_complete(),
         "JSON string markers remain verbatim JSON data");
  reason.Accept(9);
  reason.ResetHistory({});
  Expect(!reason.constraint_complete(),
         "reset restores initial reasoning phase");
  reason.Accept(10);
  reason.Accept(4);
  expect_illegal(reason, 9);
  reason.Accept(5);
  reason.Accept(6);
  expect_illegal(reason, 9);
  expect_illegal(reason, 15);
  reason.Accept(7);
  expect_illegal(reason, 9);
  reason.Accept(8);
  Expect(reason.constraint_complete(),
         "tool completion requires a whole closing marker");
  expect_illegal(reason, 11);
  reason.Accept(12);
  Expect(!reason.constraint_complete(), "another tool call reopens EOS gating");
  reason.Accept(6);
  reason.Accept(13);
  Expect(reason.constraint_complete(),
         "multiple closed tool calls can terminate");

  auto required = std::make_shared<const Constraint>(
      prototype, vocabulary,
      Constraint::Options{.allow_tool_calls = true, .require_tool_call = true});
  SamplerState tool({}, {}, required);
  expect_illegal(tool, 14);
  tool.Accept(12);
  tool.Accept(6);
  tool.Accept(13);
  Expect(tool.constraint_complete(),
         "required tool route forbids the JSON answer branch");
  SamplerState json({}, {},
                    std::make_shared<const Constraint>(prototype, vocabulary));
  expect_illegal(json, 12);
  expect_illegal(json, 10);
  json.Accept(14);
  Expect(json.constraint_complete(),
         "plain structured output does not enable reasoning or tools");
}

void TestToolArgumentMarkersAreNotFraming() {
  using Constraint = gufo::sampling::TokenConstraint;
  using gufo::sampling::SamplerState;
  auto vocabulary = std::make_shared<Constraint::Vocabulary>();
  for (unsigned byte = 0; byte < 256; ++byte)
    vocabulary->push_back({std::string(1, static_cast<char>(byte))});
  vocabulary->push_back({"", true, true});
  const auto constraint = std::make_shared<const Constraint>(
      std::make_shared<const gufo::JsonConstraint>(
          gufo::JsonConstraint::JsonObject()),
      std::move(vocabulary), Constraint::Options{.allow_tool_calls = true});
  for (
      const std::string& body : std::array<std::string, 2>{
          R"({"name":"tool","arguments":{"s":"</tool_call> and \"quoted\" <tool_call>"}})",
          std::string(
              "<function=tool><parameter=s>literal </tool_call> </function>"
              "</parameter></function>")}) {
    SamplerState sampler({}, {}, constraint);
    for (const unsigned char byte : std::string("<tool_call>") + body) {
      sampler.Accept(byte);
      Expect(
          !sampler.constraint_complete(),
          "closing markers inside JSON strings or XML parameters cannot stop");
    }
    auto copy = sampler;
    for (const unsigned char byte : std::string("</tool_call>"))
      copy.Accept(byte);
    Expect(copy.constraint_complete() && !sampler.constraint_complete(),
           "tool body lexical state is copied without sharing mutable framing");
    copy.Accept(256);
  }
}

}  // namespace

int main() {
  bool sub_float_resolution = false;
  for (std::uint64_t seed = 1; seed < 100; ++seed) {
    auto state = seed;
    auto reference = seed;
    const auto bits = gufo::sampling::NextRandom(&reference);
    const double draw = gufo::sampling::Uniform(&state);
    Expect(draw >= 0 && draw < 1 && state == reference,
           "uniform stays half-open and consumes one RNG word");
    Expect(draw == static_cast<double>(bits >> 11) * 0x1.0p-53,
           "uniform retains all 53 random mantissa bits");
    sub_float_resolution |=
        draw != static_cast<double>(static_cast<float>(draw));
  }
  Expect(sub_float_resolution, "uniform must exceed float resolution");
  TestToolArgumentMarkersAreNotFraming();
  TestStructuredOutputPhases();
  TestJsonConstrainedSelectionAndStops();
  TestJsonConstraintCopiesAndReset();
  TestJsonConstraintFailsClosedAndPreservesBytes();
  TestCanonicalTargetDistribution();
  TestResponsePenaltyScope();
  TestGreedySelectsFiniteArgmaxWithoutAdvancingRng();
  TestDefaultConfigPreservesGreedyDecoding();
  TestTemperatureSamplingIsDeterministicAndNonGreedy();
  TestTopKFiltersTheCandidateSet();
  TestTopPFiltersByCumulativeProbability();
  TestMinPFiltersRelativeToTheBestToken();
  TestMinKeepProvidesAFilterFloor();
  TestTemperatureBeforeProbabilityFilters();
  TestRepeatPenaltyUsesCommittedHistory();
  TestFrequencyAndPresencePenaltiesUseCounts();
  TestRepeatWindowIsBounded();
  TestFastGreedyMatchesNormalizedPenaltyRoute();
  TestFastTopKSamplingNeverEscapesSelectedSet();
  TestFastMinPSamplingNeverEscapesRelativeThreshold();
  TestSeedAndStateAreRequestLocal();
  TestDistributionIsNormalized();
  TestSamplingFailsClosedOnInvalidInputs();
  TestZeroDrawAndNonFiniteCandidates();
  TestStrategyReplayAgainstReference();
  TestFilteredDistributionAgainstFullSort();
  TestDeferredResidualReplay();
  std::cout << "All logit sampler tests passed.\n";
  return 0;
}
