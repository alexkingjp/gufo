#ifndef GUFO_CORE_JSON_CONSTRAINT_HPP_
#define GUFO_CORE_JSON_CONSTRAINT_HPP_

#include <cstddef>
#include <memory>
#include <string_view>

#include "src/core/json.hpp"

namespace gufo {

// Incremental, bounded JSON byte grammar. Copies share immutable compiled rules
// but own their decoding state; Allows never changes that state. Empty token
// pieces are disallowed (EOS must use Complete, not Allows).
class JsonConstraint {
public:
  struct Limits {
    std::size_t max_schema_bytes = 262144;
    std::size_t max_schema_nodes = 16384;
    std::size_t max_grammar_nodes = 65536;
    // Shared across compilation, finite-value equality, and all schema proofs.
    std::size_t max_compile_steps = 1048576;
    std::size_t max_states = 4096;
    std::size_t max_depth = 32;
    std::size_t max_properties = 256;
    std::size_t max_items = 256;
    std::size_t max_string_length = 4096;
    std::size_t max_number_bytes = 128;
    std::size_t max_whitespace = 8;
    std::size_t max_output_bytes = 1048576;
  };

  // Throws invalid_argument with a JSON Pointer on unsupported, malformed, or
  // unsatisfiable schemas (including schemas exceeding the safety limits).
  // Supports boolean schemas, type (also arrays of types), properties,
  // required, additionalProperties, items, min/maxItems, min/maxLength, enum,
  // const, anyOf, provably disjoint oneOf, and local $ref/$defs/definitions.
  // Annotations are accepted; other keywords and unsupported intersections are
  // rejected.
  [[nodiscard]] static JsonConstraint Compile(const json::Value& schema);
  [[nodiscard]] static JsonConstraint Compile(const json::Value& schema,
                                              const Limits& limits);
  [[nodiscard]] static JsonConstraint Compile(std::string_view schema_json);
  [[nodiscard]] static JsonConstraint Compile(std::string_view schema_json,
                                              const Limits& limits);
  [[nodiscard]] static JsonConstraint Compile(const char* schema_json);

  // Generic JSON syntax, restricted to an object at the document root.
  [[nodiscard]] static JsonConstraint JsonObject();
  [[nodiscard]] static JsonConstraint JsonObject(const Limits& limits);

  JsonConstraint(const JsonConstraint&);
  JsonConstraint& operator=(const JsonConstraint&);
  JsonConstraint(JsonConstraint&&) noexcept;
  JsonConstraint& operator=(JsonConstraint&&) noexcept;
  ~JsonConstraint();

  [[nodiscard]] JsonConstraint Clone() const;
  [[nodiscard]] bool Allows(std::string_view token_bytes) const;
  // Strong exception guarantee: invalid/empty pieces throw invalid_argument
  // without advancing the state. Token pieces may split UTF-8 and JSON escapes.
  void Accept(std::string_view token_bytes);
  [[nodiscard]] bool Complete() const;
  // Structural validation, independent of emission order/canonical spelling.
  // Applies the compiled schema to the existing AST, not to the cursor state.
  // Fails closed on validation work exhaustion (1M visits), excessive nesting,
  // or inexact parsed numeric literals used for const/enum/integer checks.
  [[nodiscard]] bool Validate(const json::Value& value) const;

  // Generation uses a valid subset of each supported schema: named properties
  // appear in declaration order, followed by undeclared required properties;
  // only these keys are emitted when any are declared/required. Other objects
  // admit arbitrary keys (JSON syntax does not require their uniqueness).
  // Const/enum values and property names use json::Value::dump spelling.
  // Integers use decimal integer spelling, without fractions or exponents.
  // Strings count Unicode scalar values, including surrogate-pair escapes as
  // one scalar; malformed UTF-8 and unpaired surrogates are never accepted.
  // Unbounded schema lengths/depths are bounded by Limits. Number constraints
  // are lexical, not binary64-range checks. Whitespace is bounded per boundary.
  // Const/enum numbers must survive the AST's binary64 round trip and have
  // magnitude <= 2^53-1. Schema/reference expansion is limited to 128 hops;
  // combined compiler/proof/validator traversal is capped at 256 frames.
  // Saturated ambiguity frontiers retain a deterministic viable subset; this
  // can exclude additional otherwise-valid spellings, but cannot allow a
  // schema-invalid value. Every accepted prefix has a completion within the
  // byte/depth limits. Individual bytes need not correspond to tokenizer
  // tokens.

private:
  struct Impl;
  explicit JsonConstraint(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo

#endif  // GUFO_CORE_JSON_CONSTRAINT_HPP_
