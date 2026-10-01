#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

// Keep parser diagnostics testable without exposing them as a serving API.
#include "src/cli/serve/openai_chat.cpp"

namespace {

using gufo::server::ChatRequest;
using gufo::server::ParsedChatRequest;
using gufo::server::ParsedGeneration;
using gufo::server::TextGenerationBackend;
using gufo::server::TextGenerationError;
using gufo::server::TextGenerationErrorCode;
using gufo::tokenization::ChatTool;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

ParsedGeneration Parse(std::string_view text,
                       const std::vector<ChatTool>& tools) {
  return gufo::server::ParseGeneration(
      text, TextGenerationBackend::InitialOutputState::kContent, tools,
      ChatRequest::ToolChoice::kAuto);
}

void ExpectRejected(const ParsedChatRequest& request,
                    const ParsedGeneration& generated,
                    std::string_view message) {
  bool rejected = false;
  try {
    gufo::server::ValidateGeneratedTools(request, generated);
  } catch (const TextGenerationError& error) {
    rejected = error.code() == TextGenerationErrorCode::kToolChoiceUnsatisfied;
  }
  Expect(rejected, message);
}

void TestCompleteBatchAccounting() {
  const std::vector<ChatTool> tools{
      {.name = "f",
       .parameters_json =
           R"({"type":"object","properties":{"text":{"type":"string"},"count":{"type":"integer"}}})"}};
  const std::string good =
      R"(<tool_call>{"name":"f","arguments":{"text":"ok","count":1}}</tool_call>)";
  const std::string dsml_invoke =
      "DSML\xef\xbd\x9c"
      "invoke";
  const std::string dsml_parameter =
      "DSML\xef\xbd\x9c"
      "parameter";
  const std::string dsml =
      "<tool_calls><" + dsml_invoke + " name=\"f\"><" + dsml_parameter +
      " name=\"text\" string=\"true\">ok</" + dsml_parameter + "></" +
      dsml_invoke + "></tool_calls>";
  const std::string invalid_dsml =
      "<tool_calls><" + dsml_invoke + " name=\"f\"><" + dsml_parameter +
      " name=\"count\" string=\"false\">bad</" + dsml_parameter + "></" +
      dsml_invoke + "></tool_calls>";
  const std::string literal =
      "<think>preserved</think> literal </tool_call> and <tool_call>";
  const std::string json_literals =
      "<tool_call>{\"name\":\"f\",\"arguments\":{\"text\":" +
      gufo::json::Value(literal).dump() + "}}</tool_call>";
  const std::string xml_literals = "<tool_call><function=f><parameter=text>" +
                                   literal +
                                   "</parameter></function></tool_call>";
  const std::string cross_protocol_literal =
      "<tool_call>{\"name\":\"f\",\"arguments\":{\"text\":" +
      gufo::json::Value(dsml).dump() + "}}</tool_call>";

  struct Case {
    std::string_view label;
    std::string text;
    std::size_t calls;
    bool invalid;
  };
  const std::vector<Case> cases{
      {"valid pair", good + good, 2, false},
      {"unknown second",
       good + R"(<tool_call>{"name":"unknown","arguments":{}}</tool_call>)", 1,
       true},
      {"malformed second", good + "<tool_call>{broken}</tool_call>", 1, true},
      {"bad typed second",
       good + "<tool_call><function=f><parameter=count>oops</parameter>"
              "</function></tool_call>",
       1, true},
      {"incomplete second", good + "<tool_call><function=f>", 1, true},
      {"trailing junk", good + "ignored junk", 1, true},
      {"invalid first", "<tool_call>{broken}</tool_call>" + good, 1, true},
      {"JSON marker literals", json_literals, 1, false},
      {"XML marker literals", xml_literals, 1, false},
      {"DSML call", dsml, 1, false},
      {"mixed syntaxes", good + dsml, 2, false},
      {"DSML missing envelope end",
       dsml.substr(0, dsml.size() - std::string_view("</tool_calls>").size()),
       1, true},
      {"DSML invalid typed second", dsml + invalid_dsml, 1, true},
      {"cross-protocol literal", cross_protocol_literal, 1, false},
      {"unclosed XML parameter cannot absorb next call",
       "<tool_call><function=f><parameter=text>unclosed"
       "<tool_call><function=f><parameter=text>ok</parameter>"
       "<parameter=count>42</parameter></function></tool_call>",
       1, true},
  };
  ParsedChatRequest request;
  request.chat.tools = tools;
  for (const auto& test : cases) {
    const auto parsed = Parse(test.text, tools);
    Expect(parsed.tool_calls.size() == test.calls, test.label);
    Expect(parsed.invalid_tool_calls == test.invalid, test.label);
    if (test.invalid) {
      ExpectRejected(request, parsed, test.label);
    } else {
      gufo::server::ValidateGeneratedTools(request, parsed);
    }
  }
  for (const auto& text : {json_literals, xml_literals}) {
    const auto parsed = Parse(text, tools);
    Expect(parsed.reasoning_content.empty(),
           "tool argument markers never become reasoning");
    const auto& argument = parsed.tool_calls.front().arguments.front();
    Expect(argument.is_string && argument.value == literal,
           "tool argument marker bytes stay literal");
  }
  const auto cross_protocol = Parse(cross_protocol_literal, tools);
  Expect(cross_protocol.tool_calls.front().arguments.front().value == dsml,
         "embedded DSML remains one string argument, not a second call");

  ExpectRejected(request,
                 Parse(good + "<tool_call>{broken}</tool_call>", tools),
                 "invalid auto batch is rejected before emission");
  request.parallel_tool_calls = false;
  ExpectRejected(request, Parse(good + good, tools),
                 "parallel=false rejects a complete two-call batch");
  gufo::server::ValidateGeneratedTools(request, Parse(good, tools));
}

}  // namespace

int main() {
  TestCompleteBatchAccounting();
  std::cout << "tool batch validation tests passed\n";
}
