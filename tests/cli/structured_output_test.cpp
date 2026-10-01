#include "src/cli/serve/structured_output.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/utf8.hpp"

namespace {

using gufo::server::StructuredOutputFilter;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

struct Output {
  std::string reasoning;
  std::string content;
  std::vector<bool> channels;

  bool Emit(std::string_view text, bool is_reasoning) {
    gufo::core::Utf8Decoder decoder;
    Expect(decoder.Push(text, true) == text,
           "each callback contains complete valid UTF-8");
    (is_reasoning ? reasoning : content).append(text);
    channels.push_back(is_reasoning);
    return true;
  }
};

void TestJsonMarkersAcrossEverySplit() {
  const std::string json =
      " \t\r\n{\"text\":\"<think></think><tool_call></tool_call>"
      "<tool_response><|im_end|><|endoftext|>\",\"utf8\":\""
      "\xF0\x9F\x98\x80\xE2\x82\xAC\",\"escape\":\"\\u003c/think>\"}\n";
  for (std::size_t split = 0; split <= json.size(); ++split) {
    Output output;
    StructuredOutputFilter filter(false, true, [&](auto text, bool reasoning) {
      return output.Emit(text, reasoning);
    });
    Expect(filter.Push(std::string_view(json).substr(0, split)),
           "first JSON piece succeeds");
    Expect(filter.Push(std::string_view(json).substr(split), true),
           "second JSON piece succeeds");
    Expect(filter.content() == json && output.content == json,
           "JSON marker strings, escapes, whitespace and UTF-8 stay exact");
    Expect(filter.reasoning().empty() && output.reasoning.empty(),
           "JSON markers never create reasoning");
    Expect(!filter.has_tool_route() && filter.tool_text().empty(),
           "JSON markers never create tool calls");
    Expect(filter.route_complete(), "JSON routing is committed");
  }
}

void TestReasoningAcrossEverySplit() {
  const std::string reasoning =
      "literal <think><tool_call></tool_call><tool_response> and </thinX> "
      "\xE2\x82\xAC";
  const std::string content = "\n\n{\"text\":\"</think><tool_call>\"}";
  const std::string raw = reasoning + "</think>" + content;
  for (std::size_t split = 0; split <= raw.size(); ++split) {
    Output output;
    StructuredOutputFilter filter(true, true,
                                  [&](auto text, bool is_reasoning) {
                                    return output.Emit(text, is_reasoning);
                                  });
    Expect(filter.Push(std::string_view(raw).substr(0, split)),
           "first reasoning piece succeeds");
    Expect(filter.Push(std::string_view(raw).substr(split), true),
           "reasoning-to-JSON transition succeeds");
    Expect(filter.reasoning() == reasoning && output.reasoning == reasoning,
           "only closing think delimiter ends reasoning");
    Expect(filter.content() == content && output.content == content,
           "reasoning and control delimiter never reach content");
    Expect(filter.route_complete() && !filter.has_tool_route(),
           "reasoning transitions to a JSON-only branch");
  }
}

void TestToolRouteByteByByte() {
  const std::string reasoning = "tool literal <tool_call> is still reasoning";
  const std::string tools =
      "<tool_call>\n<function=read><parameter=path>file</parameter>"
      "</function></tool_call>\n<tool_call>"
      "{\"name\":\"next\",\"arguments\":{\"text\":\"</think>\"}}"
      "</tool_call> \t\r\n";
  const std::string raw = reasoning + "</think> \t\r\n" + tools;
  Output output;
  StructuredOutputFilter filter(true, true, [&](auto text, bool is_reasoning) {
    return output.Emit(text, is_reasoning);
  });
  for (const char byte : raw)
    Expect(filter.Push(std::string_view(&byte, 1)),
           "bytewise tool input succeeds");
  Expect(filter.Push({}, true), "tool stream finalizes");
  Expect(filter.reasoning() == reasoning && output.reasoning == reasoning,
         "tool branch preserves only actual preceding reasoning");
  Expect(filter.tool_text() == tools,
         "tool buffer excludes routing whitespace and retains all call bytes");
  Expect(filter.content().empty() && output.content.empty(),
         "no tool bytes reach answer content");
  Expect(filter.has_tool_route() && filter.route_complete(),
         "full tool prefix commits tool route");
}

void TestToolRoutingAcrossEverySplit() {
  const std::string tools =
      "<tool_call>{\"name\":\"f\",\"arguments\":{}}"
      "</tool_call>";
  const std::string raw = " \r\n\t" + tools;
  for (std::size_t split = 0; split <= raw.size(); ++split) {
    Output output;
    StructuredOutputFilter filter(false, true, [&](auto text, bool reasoning) {
      return output.Emit(text, reasoning);
    });
    Expect(filter.Push(std::string_view(raw).substr(0, split)),
           "first tool route piece succeeds");
    Expect(filter.Push(std::string_view(raw).substr(split), true),
           "second tool route piece succeeds");
    Expect(filter.tool_text() == tools && filter.has_tool_route(),
           "all tool prefix splits classify identically");
    Expect(output.content.empty() && output.reasoning.empty(),
           "tool route emits no callback bytes");
  }
}

void TestJsonRootsAndNoAutoThinking() {
  constexpr std::array<std::string_view, 10> roots{
      "{}",   "[]",   "\"<think></think>\"", "-1", "0", "23", "true", "false",
      "null", "1.2e3"};
  for (const auto root : roots) {
    StructuredOutputFilter filter(false, false);
    Expect(filter.Push(root, true), "generic JSON root is classified");
    Expect(filter.content() == root && filter.route_complete(),
           "classification does not assume an object root");
  }
  StructuredOutputFilter filter(false, true);
  bool rejected = false;
  try {
    (void)filter.Push("<think>unexpected</think>{}", true);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  Expect(rejected && filter.reasoning().empty() && filter.content().empty(),
         "thinking-off never rediscovers a think block in generated bytes");
}

void TestUnfinishedPhasesDoNotLeak() {
  constexpr std::array<std::string_view, 8> reasoning_tails{
      "", "<", "</", "</t", "</th", "</thi", "</thin", "</think"};
  for (const auto tail : reasoning_tails) {
    Output output;
    StructuredOutputFilter filter(true, true, [&](auto text, bool reasoning) {
      return output.Emit(text, reasoning);
    });
    const std::string text = std::string("reasoning ") + std::string(tail);
    Expect(filter.Push(text), "unfinished reasoning is accepted");
    Expect(filter.Push({}, true), "final reasoning prefix is flushed");
    Expect(output.reasoning == text && filter.reasoning() == text,
           "partial closing marker stays reasoning at length limit");
    Expect(output.content.empty() && filter.content().empty() &&
               !filter.route_complete(),
           "reasoning-only truncation has no final answer");
  }
  constexpr std::array<std::string_view, 6> routes{
      "", " \t\r\n", "<", "<tool", " \t<tool_call", "<tool_call"};
  for (const auto route : routes) {
    Output output;
    StructuredOutputFilter filter(false, true, [&](auto text, bool reasoning) {
      return output.Emit(text, reasoning);
    });
    Expect(filter.Push(route, true), "partial routing waits for caller policy");
    Expect(!filter.route_complete() && !filter.has_tool_route(),
           "partial tool prefix is not a committed tool route");
    Expect(filter.content().empty() && filter.tool_text().empty() &&
               output.content.empty() && output.reasoning.empty(),
           "ambiguous route never falls back to answer text");
  }
  StructuredOutputFilter filter(true, true);
  Expect(filter.Push("reasoning</think>\n", true),
         "closing thought without answer may be truncated");
  Expect(!filter.route_complete() && filter.reasoning() == "reasoning" &&
             filter.content().empty(),
         "thought completion is not answer completion");
}

void TestInvalidRoutesAreNotPublished() {
  constexpr std::array<std::string_view, 8> invalid{
      "answer", "<tool_calX>", "<tool_call X>", "<tool_ call>",
      "\v{}",   "\f{}",        "\xC2\xA0{}",    "</think>{}"};
  for (const auto raw : invalid) {
    Output output;
    StructuredOutputFilter filter(false, true, [&](auto text, bool reasoning) {
      return output.Emit(text, reasoning);
    });
    bool rejected = false;
    try {
      (void)filter.Push(raw, true);
    } catch (const std::runtime_error& error) {
      rejected = true;
      const std::string_view message = error.what();
      Expect(message == "structured output has an invalid answer route" ||
                 message == "structured output has an invalid tool prefix",
             "route error uses a constant diagnostic, not raw model output");
    }
    Expect(rejected && output.content.empty() && output.reasoning.empty(),
           "invalid answer routing is rejected before publishing bytes");
  }
  StructuredOutputFilter filter(false, false);
  bool rejected = false;
  try {
    (void)filter.Push(" \n<tool_call>", true);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  Expect(rejected && !filter.has_tool_route() && filter.content().empty(),
         "tool-choice none cannot select a tool route");
}

void TestUtf8Flushing() {
  Output output;
  StructuredOutputFilter filter(true, true, [&](auto text, bool reasoning) {
    return output.Emit(text, reasoning);
  });
  Expect(filter.Push("\xF0\x9F"), "incomplete UTF-8 is held");
  Expect(output.reasoning.empty(), "UTF-8 prefix is not published early");
  Expect(filter.Push("\x98\x80</thi"), "scalar completion is published");
  Expect(output.reasoning == "\xF0\x9F\x98\x80",
         "completed reasoning scalar is exact");
  Expect(filter.Push("nk>\"\xE2"), "JSON scalar prefix is held");
  Expect(filter.Push("\x82\xAC\"", true), "JSON scalar is completed");
  Expect(output.content == "\"\xE2\x82\xAC\"",
         "JSON scalar remains valid UTF-8");

  StructuredOutputFilter truncated(true, false);
  Expect(truncated.Push("\xE2\x82", true), "truncated scalar is finalized");
  Expect(truncated.reasoning() == "\xEF\xBF\xBD" && truncated.content().empty(),
         "malformed reasoning uses existing replacement policy, not content");
}

void TestToolBranchNeverFallsBack() {
  Output output;
  StructuredOutputFilter filter(false, true, [&](auto text, bool reasoning) {
    return output.Emit(text, reasoning);
  });
  const std::string malformed =
      "<tool_call>unfinished</think>possibly secret text";
  Expect(filter.Push(malformed, true), "tool validation belongs to API parser");
  Expect(filter.has_tool_route() && filter.route_complete(),
         "route_complete certifies classification, not tool-call syntax");
  Expect(filter.tool_text() == malformed && output.content.empty() &&
             output.reasoning.empty(),
         "malformed or truncated tool output stays quarantined");
}

void TestCallbackCancellationAndFinalization() {
  std::size_t calls = 0;
  StructuredOutputFilter filter(true, true, [&](auto, bool reasoning) {
    ++calls;
    Expect(reasoning, "cancelled callback was reasoning, not content");
    return false;
  });
  Expect(!filter.Push("reasoning</think>{}"),
         "callback cancellation stops same-chunk phase transitions");
  Expect(filter.content().empty() && !filter.route_complete() && calls == 1,
         "no content leaks after reasoning callback disconnects");
  Expect(!filter.Push("{}", true) && calls == 1,
         "disconnected filter stays disconnected");

  StructuredOutputFilter finished(false, false);
  Expect(finished.Push("{}", true) && finished.Push({}, true),
         "empty repeated finalization is harmless");
  bool rejected = false;
  try {
    (void)finished.Push("more");
  } catch (const std::logic_error&) {
    rejected = true;
  }
  Expect(rejected && finished.content() == "{}",
         "input after finalization cannot change output");
}

}  // namespace

int main() {
  TestJsonMarkersAcrossEverySplit();
  TestReasoningAcrossEverySplit();
  TestToolRouteByteByByte();
  TestToolRoutingAcrossEverySplit();
  TestJsonRootsAndNoAutoThinking();
  TestUnfinishedPhasesDoNotLeak();
  TestInvalidRoutesAreNotPublished();
  TestUtf8Flushing();
  TestToolBranchNeverFallsBack();
  TestCallbackCancellationAndFinalization();
  std::cout << "structured_output_test passed\n";
}
