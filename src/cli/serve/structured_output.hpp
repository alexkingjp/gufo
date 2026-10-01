#ifndef GUFO_CLI_SERVE_STRUCTURED_OUTPUT_HPP_
#define GUFO_CLI_SERVE_STRUCTURED_OUTPUT_HPP_

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>

#include "src/core/utf8.hpp"

namespace gufo::server {

// Classifies constrained Qwen output without interpreting markers inside JSON.
// The sampler owns JSON validity; complete tool calls are validated by the API.
class StructuredOutputFilter {
public:
  using EmitCallback =
      std::function<bool(std::string_view text, bool is_reasoning)>;

  StructuredOutputFilter(bool initial_reasoning, bool allow_tools,
                         EmitCallback emit = {});

  // False means the callback disconnected. Invalid routing throws runtime_error
  // without publishing route bytes. Final flushes UTF-8 and unfinished
  // reasoning; the caller decides whether an unfinished route means truncation
  // or failure.
  bool Push(std::string_view bytes, bool final = false);

  [[nodiscard]] std::string_view reasoning() const noexcept;
  [[nodiscard]] std::string_view content() const noexcept;
  [[nodiscard]] std::string_view tool_text() const noexcept;
  [[nodiscard]] bool has_tool_route() const noexcept;
  // Classification only: true does not certify complete JSON or tool syntax.
  [[nodiscard]] bool route_complete() const noexcept;

private:
  enum class Phase : std::uint8_t { kReasoning, kRoute, kJson, kTool };

  bool Emit(std::string_view text, bool is_reasoning);
  bool Route(std::string_view text);

  EmitCallback emit_;
  core::Utf8Decoder decoder_;
  std::string reasoning_;
  std::string content_;
  std::string tool_text_;
  std::string pending_;
  std::string route_whitespace_;
  Phase phase_;
  bool allow_tools_;
  bool finalized_{false};
  bool disconnected_{false};
};

}  // namespace gufo::server

#endif  // GUFO_CLI_SERVE_STRUCTURED_OUTPUT_HPP_
