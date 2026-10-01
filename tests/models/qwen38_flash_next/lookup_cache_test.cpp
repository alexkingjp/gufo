#include "src/models/qwen38_flash_next/lookup_cache.hpp"

#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>

namespace lookup = gufo::models::qwen38_flash_next::lookup;

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <std::size_t n>
std::span<const std::int32_t> Span(const std::int32_t (&values)[n]) {
  return std::span(values, n);
}

void CheckFollowerTableAgainstReference() {
  std::mt19937_64 rng{20260927};
  lookup::FollowerTable table;
  std::map<std::uint32_t, std::uint32_t> reference;
  std::uint32_t total = 0;
  for (int step = 0; step < 20000; ++step) {
    const auto token = static_cast<std::uint32_t>(rng() % 64);
    table.Bump(token);
    ++reference[token];
    ++total;
    const auto* best = table.Best();
    if (best != nullptr) {
      std::pair<std::uint32_t, std::uint32_t> expected_best{0, 0};
      for (const auto& [token_id, count] : reference) {
        if (count > expected_best.second)
          expected_best = {token_id, count};
      }
      Require(best->count == expected_best.second,
              "follower best count diverged from reference");
      Require(best->token == expected_best.first,
              "follower best token diverged from reference");
    }
    Require(table.total() == total, "follower total diverged");
  }
}

void CheckCatchUpIdempotence() {
  std::mt19937_64 rng{7};
  std::vector<std::int32_t> tokens;
  for (int i = 0; i < 5000; ++i)
    tokens.push_back(static_cast<std::int32_t>(rng() % 512));

  lookup::ContextNgramCache full;
  full.CatchUp(tokens);

  // Incremental catch-up in slices must converge to the same query behavior.
  lookup::ContextNgramCache incremental;
  for (std::size_t end : {std::size_t{1}, std::size_t{17}, std::size_t{256},
                          std::size_t{4999}, tokens.size()}) {
    incremental.CatchUp(std::span(tokens).first(end));
  }
  Require(incremental.built() == full.built(), "catch-up watermarks differ");
  Require(incremental.size() == full.size(), "catch-up table sizes differ");

  std::mt19937_64 query_rng{99};
  for (int q = 0; q < 2000; ++q) {
    const std::size_t length = 1 + query_rng() % 3;
    std::vector<std::int32_t> context;
    for (std::size_t i = 0; i < length; ++i)
      context.push_back(static_cast<std::int32_t>(query_rng() % 512));
    const auto a = full.ProposeOne(context);
    const auto b = incremental.ProposeOne(context);
    Require(a.has_value() == b.has_value(),
            "catch-up paths disagree on proposal presence");
    if (a.has_value())
      Require(*a == *b, "catch-up paths disagree on proposal token");
  }

  // Re-catching-up the same tokens changes nothing.
  const auto before = full.size();
  full.CatchUp(tokens);
  Require(full.size() == before, "repeated catch-up duplicated entries");
}

void CheckThresholdsAndLongestMatch() {
  // Corpus: context "9 9" always followed by 5 (three times) then 6 (once).
  const std::vector<std::int32_t> corpus{
      1, 2, 9, 9, 5, 1, 2, 9, 9, 5, 1, 2, 9, 9, 5, 1, 2, 9, 9, 6,
      3, 4, 8, 8, 7, 3, 4, 8, 8, 7, 3, 4, 8, 8, 7};
  lookup::ContextNgramCache cache;
  cache.CatchUp(corpus);

  // Context "9 9": followers 5x3, 6x1 → dominant 5 passes (3/4 >= 0.5,
  // count 3 >= 1); at context length 2 thresholds are count>=1, p>=0.5.
  const auto proposal =
      cache.ProposeOne(Span({9, 9}));
  Require(proposal.has_value(), "dominant follower was not proposed");
  Require(*proposal == 5, "wrong dominant follower proposed");

  // A 3-way split fails the 0.5 probability floor at every context length,
  // and single counts fail the unigram count floor of 2.
  const std::vector<std::int32_t> split{10, 11, 20, 10, 11, 21, 10, 11, 22};
  lookup::ContextNgramCache split_cache;
  split_cache.CatchUp(split);
  Require(!split_cache.ProposeOne(Span({10, 11})).has_value(),
          "three-way split proposed a follower below the probability floor");

  // Longest confident match wins over a shorter one: after "30 31 32" the
  // corpus continues with 40 every time, while bare "32" has mixed followers.
  std::vector<std::int32_t> long_corpus;
  for (int repeat = 0; repeat < 5; ++repeat) {
    long_corpus.insert(long_corpus.end(), {30, 31, 32, 40});
  }
  for (int repeat = 0; repeat < 10; ++repeat) {
    long_corpus.insert(long_corpus.end(), {50, 32, 41});
  }
  lookup::ContextNgramCache long_cache;
  long_cache.CatchUp(long_corpus);
  const auto long_proposal =
      long_cache.ProposeOne(Span({31, 32}));
  Require(long_proposal.has_value() && *long_proposal == 40,
          "longest context match did not win");

  // Unigram floor: a token that is 1/3 of the corpus never passes
  // count>=2 && p>=0.66 on its own; a dominant repeated token does.
  const std::vector<std::int32_t> unigram{60, 60, 60, 60, 61, 62};
  lookup::ContextNgramCache unigram_cache;
  unigram_cache.CatchUp(unigram);
  const auto unigram_proposal =
      unigram_cache.ProposeOne(std::span<const std::int32_t>());
  Require(unigram_proposal.has_value() && *unigram_proposal == 60,
          "unigram majority was not proposed");
}

void CheckMapGrowth() {
  // Force many rehashes: distinct keys over a large synthetic corpus, then
  // verify every context still resolves to its trained follower.
  std::vector<std::int32_t> corpus;
  for (std::int32_t i = 0; i < 20000; i += 2) {
    // Context (i) is always followed by i+1; both passes make the follower
    // count reach the bigram floor of two.
    corpus.push_back(i);
    corpus.push_back(i + 1);
    corpus.push_back(i);
    corpus.push_back(i + 1);
  }
  lookup::ContextNgramCache cache;
  cache.CatchUp(corpus);
  for (std::int32_t i = 0; i < 20000; i += 2) {
    const std::int32_t context[] = {i};
    const auto proposal = cache.ProposeOne(context);
    Require(proposal.has_value(), "key lost across rehash growth");
    Require(*proposal == static_cast<std::uint32_t>(i + 1),
            "follower corrupted across rehash growth");
  }

  cache.Reset();
  Require(cache.size() == 0 && cache.built() == 0, "reset left state behind");
  Require(!cache.ProposeOne(Span({0})).has_value(),
          "reset cache proposed a follower");
}

void CheckCollisionSafety() {
  // The trunk verifies every proposal, so hash collisions must never break
  // the caller: a proposal is at worst a rejected draft. The contract here
  // is only that lookups terminate and return SOME trained follower.
  std::mt19937_64 rng{31337};
  lookup::ContextNgramCache cache;
  std::vector<std::int32_t> corpus;
  for (int i = 0; i < 50000; ++i)
    corpus.push_back(static_cast<std::int32_t>(rng() & 0xffff));
  cache.CatchUp(corpus);
  for (int q = 0; q < 5000; ++q) {
    std::vector<std::int32_t> context;
    for (int i = 0; i < 3; ++i)
      context.push_back(static_cast<std::int32_t>(rng() & 0xffff));
    const auto proposal = cache.ProposeOne(context);
    if (proposal.has_value())
      Require(*proposal <= 0xffff, "proposal escaped the trained vocabulary");
  }
}

int main() {
  try {
    CheckFollowerTableAgainstReference();
    CheckCatchUpIdempotence();
    CheckThresholdsAndLongestMatch();
    CheckMapGrowth();
    CheckCollisionSafety();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return 1;
  }
  std::cout << "lookup_cache tests passed\n";
  return 0;
}
