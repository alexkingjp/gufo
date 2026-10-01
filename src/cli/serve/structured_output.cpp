#include "src/cli/serve/structured_output.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace gufo::server {
namespace {

constexpr std::string_view kThinkEnd = "</think>";
constexpr std::string_view kToolStart = "<tool_call>";

bool IsJsonWhitespace(char value) {
  return value == ' ' || value == '\t' || value == '\r' || value == '\n';
}

bool IsJsonStart(char value) {
  return value == '{' || value == '[' || value == '"' || value == '-' ||
         (value >= '0' && value <= '9') || value == 't' || value == 'f' ||
         value == 'n';
}

std::size_t HeldThinkPrefix(std::string_view text) {
  for (std::size_t size = std::min(text.size(), kThinkEnd.size() - 1); size > 0;
       --size) {
    if (kThinkEnd.starts_with(text.substr(text.size() - size)))
      return size;
  }
  return 0;
}

}  // namespace

StructuredOutputFilter::StructuredOutputFilter(bool initial_reasoning,
                                               bool allow_tools,
                                               EmitCallback emit)
    : emit_(std::move(emit)),
      phase_(initial_reasoning ? Phase::kReasoning : Phase::kRoute),
      allow_tools_(allow_tools) {}

bool StructuredOutputFilter::Push(std::string_view bytes, bool final) {
  if (disconnected_)
    return false;
  if (finalized_) {
    if (bytes.empty() && final)
      return true;
    throw std::logic_error("structured output was already finalized");
  }
  finalized_ = final;
  const auto text = decoder_.Push(bytes, final);
  switch (phase_) {
    case Phase::kReasoning: {
      pending_.append(text);
      const auto end = pending_.find(kThinkEnd);
      if (end != std::string::npos) {
        if (!Emit(std::string_view(pending_).substr(0, end), true))
          return false;
        const auto remaining = pending_.substr(end + kThinkEnd.size());
        pending_.clear();
        phase_ = Phase::kRoute;
        return Route(remaining);
      }
      const auto held = final ? 0 : HeldThinkPrefix(pending_);
      const auto ready = pending_.size() - held;
      if (!Emit(std::string_view(pending_).substr(0, ready), true))
        return false;
      pending_.erase(0, ready);
      return true;
    }
    case Phase::kRoute:
      return Route(text);
    case Phase::kJson:
      return Emit(text, false);
    case Phase::kTool:
      tool_text_.append(text);
      return true;
  }
  return true;
}

bool StructuredOutputFilter::Emit(std::string_view text, bool is_reasoning) {
  if (text.empty())
    return true;
  (is_reasoning ? reasoning_ : content_).append(text);
  if (emit_ && !emit_(text, is_reasoning)) {
    disconnected_ = true;
    return false;
  }
  return true;
}

bool StructuredOutputFilter::Route(std::string_view text) {
  if (pending_.empty()) {
    std::size_t whitespace = 0;
    while (whitespace < text.size() && IsJsonWhitespace(text[whitespace]))
      ++whitespace;
    route_whitespace_.append(text.substr(0, whitespace));
    text.remove_prefix(whitespace);
    if (text.empty())
      return true;
    if (IsJsonStart(text.front())) {
      phase_ = Phase::kJson;
      route_whitespace_.append(text);
      const bool emitted = Emit(route_whitespace_, false);
      route_whitespace_.clear();
      return emitted;
    }
    if (!allow_tools_ || text.front() != '<')
      throw std::runtime_error("structured output has an invalid answer route");
  }

  // Hold every routing byte until the full delimiter identifies a tool turn.
  while (!text.empty() && pending_.size() < kToolStart.size()) {
    if (text.front() != kToolStart[pending_.size()])
      throw std::runtime_error("structured output has an invalid tool prefix");
    pending_.push_back(text.front());
    text.remove_prefix(1);
  }
  if (pending_.size() == kToolStart.size()) {
    phase_ = Phase::kTool;
    tool_text_ = std::move(pending_);
    pending_.clear();
    route_whitespace_.clear();
    tool_text_.append(text);
  }
  return true;
}

std::string_view StructuredOutputFilter::reasoning() const noexcept {
  return reasoning_;
}

std::string_view StructuredOutputFilter::content() const noexcept {
  return content_;
}

std::string_view StructuredOutputFilter::tool_text() const noexcept {
  return tool_text_;
}

bool StructuredOutputFilter::has_tool_route() const noexcept {
  return phase_ == Phase::kTool;
}

bool StructuredOutputFilter::route_complete() const noexcept {
  return phase_ == Phase::kJson || phase_ == Phase::kTool;
}

}  // namespace gufo::server
