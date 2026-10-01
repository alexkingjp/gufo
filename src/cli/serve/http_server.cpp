#include "src/cli/serve/http_server.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <unordered_set>
#include <utility>

#include "src/cli/serve/asr_service.hpp"
#include "src/cli/serve/audio_asr_api.hpp"
#include "src/cli/serve/audio_tts_api.hpp"
#include "src/cli/serve/audio_websocket.hpp"
#include "src/cli/serve/image_api.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/openai_chat.hpp"
#include "src/cli/serve/sampling_request.hpp"
#include "src/cli/serve/tts_service.hpp"
#include "src/cli/serve/video_api.hpp"
#include "src/cli/serve/video_jobs.hpp"
#include "src/cli/serve/websocket.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/json.hpp"
#include "src/core/utf8.hpp"
#include "src/models/qwen/chat_template.hpp"

namespace gufo::server {
namespace {

// ---------------------------------------------------------------------------
// Socket I/O helpers
// ---------------------------------------------------------------------------

bool ReadUntil(std::string& out, int fd, std::string_view delim) {
  char buf[4096];
  while (out.find(delim) == std::string::npos) {
    const ssize_t n = ::read(fd, buf, sizeof(buf));
    if (n <= 0)
      return false;
    out.append(buf, static_cast<std::size_t>(n));
    if (out.size() > (static_cast<std::size_t>(16) * 1024 * 1024))
      return false;
  }
  return true;
}

bool ReadN(std::string& out, int fd, std::size_t n) {
  out.reserve(n);
  std::size_t got = 0;
  char buf[4096];
  while (got < n) {
    const std::size_t want = std::min(sizeof(buf), n - got);
    const ssize_t r = ::read(fd, buf, want);
    if (r <= 0)
      return false;
    out.append(buf, static_cast<std::size_t>(r));
    got += static_cast<std::size_t>(r);
  }
  return true;
}

bool SendAll(int fd, std::string_view data) {
  std::size_t sent = 0;
  while (sent < data.size()) {
#ifdef MSG_NOSIGNAL
    const int flags = MSG_NOSIGNAL;
#else
    const int flags = 0;
#endif
    const ssize_t n = ::send(fd, data.data() + sent, data.size() - sent, flags);
    if (n <= 0)
      return false;
    sent += static_cast<std::size_t>(n);
  }
  return true;
}

bool SendChunk(int fd, std::string_view data) {
  if (data.empty())
    return true;
  char header[2 * sizeof(std::size_t) + 2];
  const auto length =
      std::to_chars(header, header + sizeof(header) - 2, data.size(), 16);
  *length.ptr = '\r';
  *(length.ptr + 1) = '\n';
  return SendAll(fd, std::string_view(header, length.ptr + 2 - header)) &&
         SendAll(fd, data) && SendAll(fd, "\r\n");
}

bool IsPeerDisconnected(int fd) noexcept {
  pollfd descriptor{
      .fd = fd,
      .events = POLLIN | POLLERR | POLLHUP,
      .revents = 0,
  };
#ifdef POLLRDHUP
  descriptor.events |= POLLRDHUP;
#endif
  const int ready = ::poll(&descriptor, 1, 0);
  if (ready <= 0)
    return false;
  if ((descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0)
    return true;
#ifdef POLLRDHUP
  if ((descriptor.revents & POLLRDHUP) != 0)
    return true;
#endif
  if ((descriptor.revents & POLLIN) == 0)
    return false;
  char byte;
  const auto count = ::recv(fd, &byte, 1, MSG_PEEK | MSG_DONTWAIT);
  return count == 0 || (count < 0 && errno != EAGAIN && errno != EWOULDBLOCK &&
                        errno != EINTR);
}

// ---------------------------------------------------------------------------
// Request / response helpers
// ---------------------------------------------------------------------------

std::string ToLower(std::string_view s) {
  std::string out(s);
  for (auto& c : out) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return out;
}

std::string UrlDecode(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (std::size_t i = 0; i < s.size(); ++i) {
    const char c = s[i];
    if (c == '+') {
      out += ' ';
    } else if (c == '%' && i + 2 < s.size()) {
      auto hexval = [](char h) {
        if (h >= '0' && h <= '9')
          return h - '0';
        if (h >= 'a' && h <= 'f')
          return h - 'a' + 10;
        if (h >= 'A' && h <= 'F')
          return h - 'A' + 10;
        return -1;
      };
      const int high = hexval(s[i + 1]);
      const int low = hexval(s[i + 2]);
      if (high >= 0 && low >= 0) {
        out += static_cast<char>((high << 4) | low);
        i += 2;
      } else {
        out += c;
      }
    } else {
      out += c;
    }
  }
  return out;
}

std::optional<std::size_t> ParseContentLength(const HttpRequest& request) {
  std::optional<std::size_t> length;
  for (const auto& [name, value] : request.headers) {
    const auto lowered = ToLower(name);
    // Chunked transfer is not implemented. Never interpret its encoded bytes
    // as an empty or partial inference request.
    if (lowered == "transfer-encoding") {
      return std::nullopt;
    }
    if (lowered != "content-length") {
      continue;
    }
    std::size_t parsed = 0;
    const auto [end, error] =
        std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (error != std::errc{} || end != value.data() + value.size() ||
        (length.has_value() && *length != parsed)) {
      return std::nullopt;
    }
    length = parsed;
  }
  return length.value_or(0);
}

std::string CredentialHash(std::string_view value) {
  return crypto::Sha256Hex(std::span(
      reinterpret_cast<const std::uint8_t*>(value.data()), value.size()));
}

bool IsAuthorized(const HttpRequest& request, const std::string& key_hash) {
  if (key_hash.empty()) {
    return true;
  }
  std::string_view credential;
  bool found = false;
  for (const auto& [name, value] : request.headers) {
    if (ToLower(name) == "authorization") {
      if (found) {
        return false;
      }
      found = true;
      credential = value;
    }
  }
  const auto space = credential.find(' ');
  if (space == std::string_view::npos ||
      ToLower(credential.substr(0, space)) != "bearer") {
    return false;
  }
  credential.remove_prefix(space + 1);
  while (credential.starts_with(' ')) {
    credential.remove_prefix(1);
  }
  // Comparing digests does not disclose matching prefixes of the API key.
  return CredentialHash(credential) == key_hash;
}

bool HasHeader(const HttpResponse& response, std::string_view name) {
  const std::string lowered = ToLower(name);
  return std::ranges::any_of(response.headers, [&](const auto& header) {
    return ToLower(header.first) == lowered;
  });
}

std::string BuildResponseHead(const HttpResponse& resp,
                              std::optional<std::size_t> content_length) {
  std::string out;
  out.reserve(256);
  out += "HTTP/1.1 ";
  out += std::to_string(resp.status);
  out += ' ';
  out += resp.reason;
  out += "\r\n";
  if (resp.status != 101 && !HasHeader(resp, "content-type")) {
    out += "Content-Type: application/json\r\n";
  }
  if (content_length.has_value()) {
    out += "Content-Length: " + std::to_string(*content_length) + "\r\n";
  }
  out += "Access-Control-Allow-Origin: *\r\n";
  out += "Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n";
  out += "Access-Control-Allow-Headers: Content-Type, Authorization, Range\r\n";
  for (const auto& [name, value] : resp.headers) {
    out += name;
    out += ": ";
    out += value;
    out += "\r\n";
  }
  if (!HasHeader(resp, "connection"))
    out += "Connection: close\r\n";
  out += "\r\n";
  return out;
}

std::string BuildResponse(const HttpResponse& resp) {
  std::string out = BuildResponseHead(resp, resp.body.size());
  out += resp.body;
  return out;
}

// ---------------------------------------------------------------------------
// Response constructors
// ---------------------------------------------------------------------------

long long Now() {
  return static_cast<long long>(std::time(nullptr));
}

std::string RandomId() {
  static constexpr char kChars[] = "abcdefghijklmnopqrstuvwxyz0123456789";
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<std::size_t> dist(0, 35);
  std::string out;
  out.reserve(12);
  for (int i = 0; i < 12; ++i) {
    out += kChars[dist(gen)];
  }
  return out;
}

HttpResponse Ok(const json::Value& v) {
  return {.status = 200, .reason = "OK", .body = v.dump()};
}

json::Value UsageJson(const TextGenerationBackend::Result& result) {
  json::Value usage = json::Value::object();
  usage["prompt_tokens"] = result.prompt_tokens;
  usage["completion_tokens"] = result.completion_tokens;
  usage["total_tokens"] = result.prompt_tokens + result.completion_tokens;
  json::Value prompt_details = json::Value::object();
  prompt_details["cached_tokens"] = result.cached_prompt_tokens;
  usage["prompt_tokens_details"] = std::move(prompt_details);

  const double prompt_per_second = PrefillTokensPerSecond(result);
  const double predicted_per_second =
      (result.decode_ms > 0.0 && result.completion_tokens > 0)
          ? (static_cast<double>(result.completion_tokens) /
             (result.decode_ms / 1000.0))
          : 0.0;

  usage["cached_tokens"] = result.cached_prompt_tokens;
  usage["prompt_tokens_per_second"] = prompt_per_second;
  usage["completion_tokens_per_second"] = predicted_per_second;
  usage["draft_tokens"] = result.draft_tokens;
  usage["draft_tokens_accepted"] = result.draft_accepted_tokens;
  return usage;
}

HttpResponse WithTiming(HttpResponse response,
                        const TextGenerationBackend::Result& result) {
  RecordServerMetrics(result);

  std::ostringstream value;
  value << std::fixed << std::setprecision(3) << "ttft;dur=" << result.ttft_ms
        << ", inter_token;dur=" << result.mean_inter_token_ms
        << ", max_inter_token;dur=" << result.max_inter_token_ms;
  response.headers.emplace_back("Server-Timing", value.str());

  response.log_details = GenerationLogDetails(result);
  return response;
}

HttpResponse Err(int status, const char* reason, const char* message,
                 const char* type, const char* code) {
  json::Value e = json::Value::object();
  json::Value obj = json::Value::object();
  obj["message"] = message;
  obj["type"] = type;
  obj["code"] = code;
  e["error"] = std::move(obj);
  return {.status = status, .reason = reason, .body = e.dump()};
}

HttpResponse NotImplemented(const HttpRequest&, TextGenerationBackend&) {
  return Err(501, "Not Implemented",
             "endpoint not implemented on this text model", "server_error",
             "not_implemented");
}

// Parse a role string into a ChatRole.
tokenization::ChatRole RoleFrom(const std::string& r) {
  if (r == "system")
    return tokenization::ChatRole::kSystem;
  if (r == "developer")
    return tokenization::ChatRole::kDeveloper;
  if (r == "assistant")
    return tokenization::ChatRole::kAssistant;
  if (r == "tool")
    return tokenization::ChatRole::kTool;
  return tokenization::ChatRole::kUser;
}

bool ReadTextContent(const json::Value* content, std::string* out) {
  if (content == nullptr)
    return false;
  if (content->is_string()) {
    *out = content->get_str();
  } else if (content->is_array()) {
    for (const auto& part : content->items()) {
      const auto* text = part.find("text");
      const auto type = part.member_str("type");
      if (!part.is_object() || text == nullptr || !text->is_string() ||
          (type != "text" && type != "input_text" && type != "output_text")) {
        return false;
      }
      *out += text->str();
    }
  } else {
    return false;
  }
  return true;
}

bool ReadResponsesReasoning(const json::Value& item, std::string* reasoning) {
  if (item.contains("role") || item.contains("tool_calls"))
    return false;
  if (const auto* id = item.find("id");
      id && (!id->is_string() || id->str().empty()))
    return false;
  if (const auto* status = item.find("status");
      status && (!status->is_string() || (status->str() != "completed" &&
                                          status->str() != "incomplete")))
    return false;
  if (const auto* encrypted = item.find("encrypted_content");
      encrypted && !encrypted->is_null())
    return false;
  // Gufo emits raw reasoning, not summaries or opaque encrypted state. Do not
  // substitute either for the model's original assistant thought.
  if (const auto* summary = item.find("summary");
      summary && (!summary->is_array() || !summary->empty()))
    return false;
  const auto* content = item.find("content");
  if (!content || !content->is_array() || content->empty())
    return false;
  for (const auto& part : content->items()) {
    const auto* text = part.find("text");
    if (!part.is_object() || part.member_str("type") != "reasoning_text" ||
        !text || !text->is_string())
      return false;
    reasoning->append(text->str());
  }
  return !reasoning->empty();
}

bool ReadTextMessages(const json::Value* input,
                      std::vector<tokenization::ChatMessage>* messages,
                      bool responses = false) {
  if (input == nullptr || !input->is_array() || input->empty())
    return false;
  std::optional<std::string> reasoning;
  const auto flush_reasoning = [&] {
    if (reasoning.has_value()) {
      // A budget-limited response can contain reasoning without a visible
      // answer. Retain that assistant turn instead of attaching it to a user.
      messages->emplace_back(tokenization::ChatRole::kAssistant, std::string{},
                             std::string{}, std::move(*reasoning));
      reasoning.reset();
    }
  };
  for (const auto& item : input->items()) {
    if (responses && item.member_str("type") == "reasoning") {
      flush_reasoning();
      reasoning.emplace();
      if (!ReadResponsesReasoning(item, &*reasoning))
        return false;
      continue;
    }
    const auto role = item.member_str("role");
    const auto* type = item.find("type");
    if (!item.is_object() ||
        (role != "user" && role != "assistant" && role != "system" &&
         role != "developer") ||
        (type && (!type->is_string() || type->str() != "message")) ||
        item.contains("tool_calls") ||
        (responses && item.contains("reasoning_content"))) {
      return false;
    }
    tokenization::ChatMessage message;
    message.role = RoleFrom(role);
    if (!ReadTextContent(item.find("content"), &message.content))
      return false;
    if (message.role != tokenization::ChatRole::kAssistant)
      flush_reasoning();
    if (reasoning.has_value()) {
      message.thought = std::move(*reasoning);
      reasoning.reset();
    }
    messages->push_back(std::move(message));
  }
  flush_reasoning();
  return true;
}

HttpResponse InvalidCompatibilityRequest(std::string_view message) {
  return Err(400, "Bad Request", std::string(message).c_str(),
             "invalid_request_error", "invalid_request");
}

bool ResponsesFunctionName(std::string_view name) {
  return !name.empty() && name.size() <= 64 &&
         std::ranges::all_of(name, [](unsigned char c) {
           return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
         });
}

std::optional<HttpResponse> NormalizeResponsesTools(const json::Value& body,
                                                    json::Value* chat) {
  if (const auto* tools = body.find("tools")) {
    if (!tools->is_array())
      return InvalidCompatibilityRequest("'tools' must be an array");
    (*chat)["tools"] = json::Value::array();
    std::unordered_set<std::string> names;
    for (const auto& tool : tools->items()) {
      if (!tool.is_object() || tool.member_str("type") != "function")
        return InvalidCompatibilityRequest("only function tools are supported");
      const bool nested = tool.contains("function");
      const auto* function = nested ? tool.find("function") : &tool;
      if (!function->is_object())
        return InvalidCompatibilityRequest("function tools must be objects");
      if (nested) {
        for (const auto& [key, value] : tool.members()) {
          if (key != "type" && key != "function")
            return InvalidCompatibilityRequest("ambiguous function tool shape");
        }
      }
      for (const auto& [key, value] : function->members()) {
        if (key != "name" && key != "description" && key != "parameters" &&
            key != "strict" && (nested || key != "type"))
          return InvalidCompatibilityRequest(
              "unsupported function tool field '" + key + "'");
      }
      const auto name = function->member_str("name");
      if (!ResponsesFunctionName(name) || !names.insert(name).second)
        return InvalidCompatibilityRequest(
            "function names must be unique, 1-64 letters, digits, '_', '-' or "
            "'.'");
      auto normalized = json::Value::object();
      normalized["type"] = "function";
      normalized["function"]["name"] = name;
      if (const auto* description = function->find("description")) {
        if (!description->is_string() && !description->is_null())
          return InvalidCompatibilityRequest(
              "tool description must be a string");
        if (!description->is_null())
          normalized["function"]["description"] = *description;
      }
      if (const auto* strict = function->find("strict")) {
        if (!strict->is_bool() && !strict->is_null())
          return InvalidCompatibilityRequest("tool strict must be a boolean");
        if (!strict->is_null())
          normalized["function"]["strict"] = *strict;
      }
      const auto* parameters = function->find("parameters");
      if (parameters && !parameters->is_null()) {
        if (!parameters->is_object())
          return InvalidCompatibilityRequest(
              "tool parameters must be an object schema");
        if (const auto* type = parameters->find("type");
            type && (!type->is_string() || type->str() != "object"))
          return InvalidCompatibilityRequest(
              "tool parameters must describe an object");
        if (const auto* properties = parameters->find("properties");
            properties && !properties->is_object())
          return InvalidCompatibilityRequest(
              "tool parameter properties must be an object");
        if (const auto* required = parameters->find("required")) {
          if (!required->is_array() ||
              std::ranges::any_of(required->items(), [](const auto& value) {
                return !value.is_string();
              }))
            return InvalidCompatibilityRequest(
                "tool required must be an array of names");
        }
        normalized["function"]["parameters"] = *parameters;
      } else {
        // An omitted schema declares a no-argument function, not arbitrary
        // arguments. Use the same explicit definition for both prompt and API.
        normalized["function"]["parameters"] = json::parse(
            R"({"type":"object","properties":{},"additionalProperties":false})");
      }
      (*chat)["tools"].push_back(std::move(normalized));
    }
  }
  if (const auto* choice = body.find("tool_choice")) {
    if (choice->is_object() && choice->contains("name")) {
      if (choice->member_str("type") != "function" || choice->size() != 2 ||
          !ResponsesFunctionName(choice->member_str("name")))
        return InvalidCompatibilityRequest(
            "named tool_choice requires type and function name");
      (*chat)["tool_choice"]["type"] = "function";
      (*chat)["tool_choice"]["function"]["name"] = *choice->find("name");
    } else {
      if (choice->is_object()) {
        const auto* function = choice->find("function");
        if (choice->size() != 2 || !function || !function->is_object() ||
            function->size() != 1)
          return InvalidCompatibilityRequest("invalid named tool_choice");
      }
      (*chat)["tool_choice"] = *choice;
    }
  }
  if (const auto* parallel = body.find("parallel_tool_calls")) {
    if (!parallel->is_bool())
      return InvalidCompatibilityRequest(
          "parallel_tool_calls must be a boolean");
    (*chat)["parallel_tool_calls"] = *parallel;
  }
  return std::nullopt;
}

bool ResponsesHistoryMetadata(const json::Value& item) {
  if (const auto* id = item.find("id");
      id && (!id->is_string() || id->str().empty()))
    return false;
  if (const auto* status = item.find("status");
      status && (!status->is_string() || (status->str() != "completed" &&
                                          status->str() != "incomplete")))
    return false;
  return true;
}

std::optional<HttpResponse> NormalizeResponsesInput(const json::Value& body,
                                                    json::Value* messages) {
  *messages = json::Value::array();
  const auto message = [](const std::string& role, const std::string& text) {
    auto value = json::Value::object();
    value["role"] = role;
    value["content"] = text;
    return value;
  };
  if (const auto* instructions = body.find("instructions");
      instructions && !instructions->is_null()) {
    if (!instructions->is_string())
      return InvalidCompatibilityRequest(
          "'instructions' must be a string or null");
    messages->push_back(message("system", instructions->str()));
  }
  const auto* input = body.find("input");
  if (input && input->is_string() && !input->str().empty()) {
    messages->push_back(message("user", input->str()));
    return std::nullopt;
  }
  if (!input || !input->is_array() || input->empty())
    return InvalidCompatibilityRequest(
        "'input' must be nonempty text or an item array");

  struct PendingCall {
    std::string id;
    std::string name;
    std::optional<std::string> output;
  };
  std::vector<PendingCall> pending;
  std::unordered_set<std::string> call_ids;
  std::optional<json::Value> assistant;
  bool assistant_has_message = false;
  bool receiving_outputs = false;
  const auto flush_assistant = [&] {
    if (assistant) {
      messages->push_back(std::move(*assistant));
      assistant.reset();
    }
    assistant_has_message = false;
  };
  const auto ensure_assistant = [&]() -> json::Value& {
    if (!assistant)
      assistant = message("assistant", "");
    return *assistant;
  };
  for (const auto& item : input->items()) {
    if (!item.is_object() || !ResponsesHistoryMetadata(item))
      return InvalidCompatibilityRequest(
          "input items require valid objects, IDs and terminal status");
    const auto type = item.member_str("type", "message");
    if (const auto* value = item.find("type"); value && !value->is_string())
      return InvalidCompatibilityRequest("input item type must be a string");
    if (type == "function_call_output") {
      if (item.contains("role") || item.contains("tool_calls") ||
          item.contains("reasoning_content"))
        return InvalidCompatibilityRequest(
            "function_call_output cannot have message fields");
      const auto id = item.member_str("call_id");
      auto found = std::ranges::find(pending, id, &PendingCall::id);
      if (id.empty() || found == pending.end() || found->output.has_value())
        return InvalidCompatibilityRequest(
            "function_call_output requires an unmatched call_id from input");
      std::string text;
      if (!ReadTextContent(item.find("output"), &text))
        return InvalidCompatibilityRequest(
            "function output must be a string or text content-part array; "
            "images and files are unsupported");
      flush_assistant();
      receiving_outputs = true;
      found->output = std::move(text);
      if (std::ranges::all_of(pending, [](const auto& call) {
            return call.output.has_value();
          })) {
        // Qwen renders tool results positionally. Correlate by call_id first,
        // then restore call order even when parallel results arrive reversed.
        for (auto& call : pending) {
          auto value = message("tool", *call.output);
          value["tool_call_id"] = call.id;
          value["name"] = call.name;
          messages->push_back(std::move(value));
        }
        pending.clear();
        receiving_outputs = false;
      }
      continue;
    }
    if (receiving_outputs)
      return InvalidCompatibilityRequest(
          "all pending function calls need outputs before the next input item");
    if (type == "function_call") {
      if (item.contains("role") || item.contains("tool_calls") ||
          item.contains("reasoning_content") || item.contains("content"))
        return InvalidCompatibilityRequest(
            "function_call cannot have message fields");
      const auto id = item.member_str("call_id");
      const auto name = item.member_str("name");
      const auto* arguments = item.find("arguments");
      if (id.empty() || !ResponsesFunctionName(name) ||
          !call_ids.insert(id).second || !arguments || !arguments->is_string())
        return InvalidCompatibilityRequest(
            "function_call requires a unique call_id, name and JSON arguments "
            "string");
      try {
        if (!json::parse(arguments->str()).is_object())
          return InvalidCompatibilityRequest(
              "function arguments must encode a JSON object");
      } catch (const std::exception&) {
        return InvalidCompatibilityRequest(
            "function arguments must encode a complete JSON object; use '{}' "
            "for no arguments");
      }
      auto call = json::Value::object();
      call["id"] = id;
      call["type"] = "function";
      call["function"]["name"] = name;
      call["function"]["arguments"] = *arguments;
      ensure_assistant()["tool_calls"].push_back(std::move(call));
      pending.push_back({id, name, std::nullopt});
      continue;
    }
    if (type == "reasoning") {
      if (!pending.empty())
        return InvalidCompatibilityRequest(
            "function calls need outputs before another reasoning turn");
      std::string reasoning;
      if (!ReadResponsesReasoning(item, &reasoning))
        return InvalidCompatibilityRequest(
            "reasoning items must contain nonempty raw reasoning_text");
      flush_assistant();
      ensure_assistant()["reasoning_content"] = std::move(reasoning);
      continue;
    }
    const auto role = item.member_str("role");
    if (type != "message" ||
        (role != "user" && role != "assistant" && role != "system" &&
         role != "developer") ||
        item.contains("tool_calls") || item.contains("reasoning_content"))
      return InvalidCompatibilityRequest(
          "input supports text messages, reasoning, function_call and "
          "function_call_output items only");
    std::string text;
    if (!ReadTextContent(item.find("content"), &text))
      return InvalidCompatibilityRequest(
          "message content must be a string or text content-part array");
    if (role == "assistant") {
      if (assistant_has_message) {
        if (!pending.empty())
          return InvalidCompatibilityRequest(
              "function calls need outputs before another assistant turn");
        flush_assistant();
      }
      ensure_assistant()["content"] = std::move(text);
      assistant_has_message = true;
    } else {
      if (!pending.empty())
        return InvalidCompatibilityRequest(
            "function calls need outputs before another message");
      flush_assistant();
      messages->push_back(message(role, text));
    }
  }
  if (!pending.empty())
    return InvalidCompatibilityRequest(
        "every input function_call requires a matching function_call_output");
  flush_assistant();
  return std::nullopt;
}

bool SameJson(const json::Value& left, const json::Value& right) {
  if (left.type() != right.type())
    return false;
  if (left.is_object()) {
    if (left.size() != right.size())
      return false;
    for (const auto& [name, value] : left.members()) {
      const auto* other = right.find(name);
      if (!other || !SameJson(value, *other))
        return false;
    }
    return true;
  }
  if (left.is_array()) {
    if (left.size() != right.size())
      return false;
    for (std::size_t index = 0; index < left.size(); ++index) {
      if (!SameJson(left.items()[index], right.items()[index]))
        return false;
    }
    return true;
  }
  return left.dump() == right.dump();
}

std::optional<HttpResponse> NormalizeResponsesFormat(const json::Value& body,
                                                     json::Value* chat) {
  const auto* text = body.find("text");
  if (!text)
    return std::nullopt;
  if (!text->is_object())
    return InvalidCompatibilityRequest("'text' must be an object");
  for (const auto& [name, value] : text->members()) {
    if (name != "format")
      return InvalidCompatibilityRequest("unsupported text control '" + name +
                                         "'");
  }
  const auto* format = text->find("format");
  if (!format)
    return std::nullopt;
  if (!format->is_object())
    return InvalidCompatibilityRequest("'text.format' must be an object");
  const auto type = format->member_str("type");
  for (const auto& [name, value] : format->members()) {
    if (name != "type" &&
        (type != "json_schema" || (name != "name" && name != "schema" &&
                                   name != "strict" && name != "description")))
      return InvalidCompatibilityRequest("unsupported text.format field '" +
                                         name + "'");
  }
  auto normalized = *format;
  if (type == "json_schema") {
    normalized = json::Value::object();
    normalized["type"] = "json_schema";
    normalized["json_schema"] = json::Value::object();
    for (const auto& [name, value] : format->members()) {
      if (name != "type")
        normalized["json_schema"][name] = value;
    }
  } else if (type != "json_object" && type != "text") {
    return InvalidCompatibilityRequest("unsupported text.format type");
  }
  if (const auto* conventional = body.find("response_format");
      conventional && !SameJson(*conventional, normalized))
    return InvalidCompatibilityRequest(
        "text.format and response_format disagree");
  if (type == "text" && format->size() == 1 &&
      !body.contains("response_format"))
    return std::nullopt;
  (*chat)["response_format"] = std::move(normalized);
  return std::nullopt;
}

// Validate compatibility options before dispatch so a client never gets an
// answer to a different request. Only Responses supports streaming/reasoning.
std::optional<HttpResponse> ReadCompatibilityOptions(
    const json::Value& body, TextGenerationBackend& backend,
    std::string_view token_field, std::size_t* max_tokens,
    sampling::SamplingConfig* sampling_config, bool responses = false) {
  if (!body.is_object())
    return InvalidCompatibilityRequest("request body must be an object");
  if (const auto* model = body.find("model"); model != nullptr) {
    if (!model->is_string())
      return InvalidCompatibilityRequest("'model' must be a string");
    if (model->str() != backend.model_id())
      return Err(404, "Not Found", "requested model is not loaded",
                 "invalid_request_error", "model_not_found");
  }
  for (const std::string field : {"stream", "echo", "store", "background"}) {
    if (const auto* value = body.find(field); value != nullptr) {
      if (responses && field == "stream") {
        if (!value->is_bool())
          return InvalidCompatibilityRequest("'stream' must be a boolean");
      } else if (!value->is_bool() || value->as_bool()) {
        return InvalidCompatibilityRequest("'" + field + "' must be false");
      }
    }
  }
  for (const std::string field : {"n", "best_of"}) {
    if (const auto* value = body.find(field);
        value != nullptr &&
        (!value->is_number() || value->as_double() != 1.0)) {
      return InvalidCompatibilityRequest("only '" + field + "=1' is supported");
    }
  }
  for (const std::string field : {"stream_options",
                                  "stop",
                                  "stop_sequences",
                                  "logprobs",
                                  "top_logprobs",
                                  "suffix",
                                  "tools",
                                  "tool_choice",
                                  "parallel_tool_calls",
                                  "response_format",
                                  "text",
                                  "reasoning",
                                  "reasoning_effort",
                                  "thinking",
                                  "chat_template_kwargs",
                                  "previous_response_id",
                                  "conversation",
                                  "include",
                                  "truncation",
                                  "modalities",
                                  "audio"}) {
    if (responses && (field == "reasoning" || field == "reasoning_effort" ||
                      field == "thinking" || field == "chat_template_kwargs" ||
                      field == "tools" || field == "tool_choice" ||
                      field == "parallel_tool_calls" || field == "text" ||
                      field == "response_format" || field == "include"))
      continue;
    if (body.contains(field)) {
      return InvalidCompatibilityRequest("request field '" + field +
                                         "' is not supported on this endpoint");
    }
  }
  for (const std::string field : {"max_tokens", "max_completion_tokens",
                                  "max_output_tokens", "n_predict"}) {
    if (field != token_field && body.contains(field)) {
      return InvalidCompatibilityRequest("use '" + std::string(token_field) +
                                         "' on this endpoint");
    }
  }
  const auto defaults = backend.sampling_defaults();
  *max_tokens = defaults.max_tokens;
  if (const auto error = detail::ReadSamplingInteger(
          body, token_field, std::size_t{1},
          std::size_t{std::numeric_limits<std::uint32_t>::max()}, max_tokens)) {
    return InvalidCompatibilityRequest(error->message);
  }
  if (const auto error =
          ParseSamplingConfig(body, defaults.sampling, sampling_config)) {
    return Err(400, "Bad Request", error->message.c_str(),
               "invalid_request_error", error->code.c_str());
  }
  return std::nullopt;
}

// ---------------------------------------------------------------------------
// Endpoint handlers
// ---------------------------------------------------------------------------

HttpResponse ListModels(TextGenerationBackend* backend,
                        const VideoJobService* video_jobs,
                        const TtsService* tts, const AsrService* asr,
                        const ImageService* images) {
  json::Value resp = json::Value::object();
  resp["object"] = "list";
  json::Value data = json::Value::array();
  if (backend != nullptr) {
    json::Value model = json::Value::object();
    model["id"] = backend->model_id();
    model["object"] = "model";
    model["created"] = Now();
    model["owned_by"] = "gufo";
    data.push_back(std::move(model));
  }
  if (video_jobs != nullptr && video_jobs->ready()) {
    json::Value root_model = json::Value::object();
    root_model["id"] = "minimax-h3";
    root_model["object"] = "model";
    root_model["created"] = Now();
    root_model["owned_by"] = "operator-supplied-minimax";
    root_model["capability"] = "video";
    data.push_back(std::move(root_model));
    for (const std::string_view preset :
         {"minimax-h3-exact", "minimax-h3-fast", "minimax-h3-aggressive",
          "minimax-h3-dev", "minimax-h3-fullres"}) {
      json::Value model = json::Value::object();
      model["id"] = std::string(preset);
      model["object"] = "model";
      model["created"] = Now();
      model["owned_by"] = "operator-supplied-minimax";
      model["root"] = "minimax-h3";
      model["capability"] = "video";
      data.push_back(std::move(model));
    }
  }
  if (tts != nullptr && tts->ready()) {
    json::Value model = json::Value::object();
    model["id"] = tts->model_id();
    model["object"] = "model";
    model["created"] = Now();
    model["owned_by"] = "operator-supplied-qwen";
    model["capability"] = "audio_tts";
    data.push_back(std::move(model));
  }
  if (asr != nullptr && asr->ready()) {
    json::Value model = json::Value::object();
    model["id"] = asr->model_id();
    model["object"] = "model";
    model["created"] = Now();
    model["owned_by"] = "operator-supplied-qwen";
    model["capability"] = "audio_asr";
    data.push_back(std::move(model));
  }
  if (images != nullptr) {
    auto model = json::Value::object();
    model["id"] = images->model_id();
    model["object"] = "model";
    model["created"] = Now();
    model["owned_by"] = "operator-supplied-qwen";
    model["capability"] = "image";
    data.push_back(std::move(model));
  }
  resp["data"] = std::move(data);
  return Ok(resp);
}

HttpResponse OpenAiCompletions(const HttpRequest& req,
                               TextGenerationBackend& b) try {
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (const std::exception& e) {
    return Err(400, "Bad Request", e.what(), "invalid_request_error",
               "parse_error");
  }

  std::size_t max_tokens = 0;
  sampling::SamplingConfig sampling_config;
  if (auto error = ReadCompatibilityOptions(body, b, "max_tokens", &max_tokens,
                                            &sampling_config)) {
    return std::move(*error);
  }

  const auto* input = body.find("prompt");
  if (input == nullptr || !input->is_string()) {
    return InvalidCompatibilityRequest("'prompt' must be a single string");
  }
  const std::string prompt = input->str();
  if (prompt.empty()) {
    return Err(400, "Bad Request", "'prompt' is required",
               "invalid_request_error", "missing_prompt");
  }

  const auto res = b.complete(prompt, max_tokens, sampling_config,
                              req.is_cancelled, {}, req.client_id);

  json::Value resp = json::Value::object();
  resp["id"] = "cmpl-" + RandomId();
  resp["object"] = "text_completion";
  resp["created"] = Now();
  resp["model"] = b.model_id();
  json::Value choices = json::Value::array();
  json::Value c = json::Value::object();
  c["text"] = core::Utf8Decoder{}.Push(res.text, true);
  c["index"] = 0;
  c["logprobs"] = json::Value();
  c["finish_reason"] =
      res.finish_reason == TextGenerationBackend::FinishReason::kLength
          ? "length"
          : "stop";
  choices.push_back(std::move(c));
  resp["choices"] = std::move(choices);
  resp["usage"] = UsageJson(res);
  resp["timings"] = GenerationTimings(res);
  return WithTiming(Ok(resp), res);
} catch (const std::length_error& error) {
  return Err(400, "Bad Request", error.what(), "invalid_request_error",
             "context_length_exceeded");
} catch (const std::invalid_argument& error) {
  return Err(400, "Bad Request", error.what(), "invalid_request_error",
             "invalid_prompt");
}

json::Value ResponsesUsage(const json::Value& chat_usage) {
  auto usage = json::Value::object();
  usage["input_tokens"] = chat_usage.member_size("prompt_tokens");
  usage["output_tokens"] = chat_usage.member_size("completion_tokens");
  usage["total_tokens"] = chat_usage.member_size("total_tokens");
  if (const auto* details = chat_usage.find("prompt_tokens_details"))
    usage["input_tokens_details"] = *details;
  // The backend counts all generated tokens, but not a separate reasoning
  // subtotal. Omit that unmeasured breakdown rather than reporting a false
  // zero.
  return usage;
}

json::Value ResponsesPart(std::string_view text, bool reasoning) {
  auto part = json::Value::object();
  part["type"] = reasoning ? "reasoning_text" : "output_text";
  part["text"] = std::string(text);
  if (!reasoning) {
    part["annotations"] = json::Value::array();
    part["logprobs"] = json::Value::array();
  }
  return part;
}

json::Value ResponsesItem(const std::string& id, std::string_view text,
                          bool reasoning, const char* status,
                          bool include_content = true) {
  auto item = json::Value::object();
  item["id"] = id;
  item["type"] = reasoning ? "reasoning" : "message";
  item["status"] = status;
  if (reasoning)
    item["summary"] = json::Value::array();
  else
    item["role"] = "assistant";
  item["content"] = json::Value::array();
  if (include_content)
    item["content"].push_back(ResponsesPart(text, reasoning));
  return item;
}

json::Value ResponsesFunctionItem(const std::string& id,
                                  const std::string& call_id,
                                  const std::string& name,
                                  const std::string& arguments,
                                  const char* status) {
  auto item = json::Value::object();
  item["id"] = id;
  item["type"] = "function_call";
  item["status"] = status;
  item["call_id"] = call_id;
  item["name"] = name;
  item["arguments"] = arguments;
  return item;
}

json::Value ResponsesEnvelope(const json::Value& body, const json::Value& chat,
                              const std::string& model, std::size_t max_tokens,
                              const sampling::SamplingConfig& sampling) {
  auto response = json::Value::object();
  response["id"] = "resp_" + RandomId();
  response["object"] = "response";
  response["created_at"] = Now();
  response["status"] = "in_progress";
  response["error"] = json::Value();
  response["incomplete_details"] = json::Value();
  response["model"] = model;
  response["output"] = json::Value::array();
  response["usage"] = json::Value();
  response["tools"] = json::Value::array();
  if (const auto* tools = chat.find("tools")) {
    for (const auto& tool : tools->items()) {
      auto flat = *tool.find("function");
      flat["type"] = "function";
      response["tools"].push_back(std::move(flat));
    }
  }
  response["tool_choice"] = response["tools"].empty() ? "none" : "auto";
  if (const auto* choice = chat.find("tool_choice")) {
    if (choice->is_object()) {
      response["tool_choice"] = json::Value::object();
      response["tool_choice"]["type"] = "function";
      response["tool_choice"]["name"] =
          choice->find("function")->member_str("name");
    } else {
      response["tool_choice"] = *choice;
    }
  }
  response["parallel_tool_calls"] = !body.contains("parallel_tool_calls") ||
                                    body.find("parallel_tool_calls")->as_bool();
  if (const auto* format = chat.find("response_format")) {
    response["text"]["format"] = *format;
    if (format->member_str("type") == "json_schema") {
      response["text"]["format"] = *format->find("json_schema");
      response["text"]["format"]["type"] = "json_schema";
    }
  } else {
    response["text"]["format"]["type"] = "text";
  }
  for (const auto* field : {"metadata", "reasoning"}) {
    if (const auto* value = body.find(field))
      response[field] = *value;
  }
  response["store"] = false;
  response["background"] = false;
  response["max_output_tokens"] = max_tokens;
  response["temperature"] = static_cast<double>(sampling.temperature);
  response["top_p"] = static_cast<double>(sampling.top_p);
  if (const auto* instructions = body.find("instructions"))
    response["instructions"] = *instructions;
  return response;
}

// Chat owns request validation, admission, UTF-8/reasoning parsing and metrics.
// This adapter consumes its complete SSE frames, never arbitrary network bytes.
class ResponsesStream {
public:
  ResponsesStream(json::Value response, const HttpResponse::BodyWriter& writer)
      : response_(std::move(response)), writer_(writer) {}

  bool Push(std::string_view frame) {
    if (!connected_)
      return false;
    if (!started_) {
      started_ = true;
      if (!ResponseEvent("response.created") ||
          !ResponseEvent("response.in_progress"))
        return false;
    }
    if (frame == "data: [DONE]\n\n") {
      if (!terminal_) {
        auto error = json::Value::object();
        error["code"] = "generation_failed";
        error["message"] =
            "generation ended without usage or a terminal choice";
        return Fail(error);
      }
      return connected_;
    }
    const auto event = json::parse(frame.substr(6));
    if (const auto* error = event.find("error"))
      return Fail(*error);
    if (terminal_)
      return connected_;
    if (const auto* usage = event.find("usage")) {
      if (finish_reason_.empty())
        throw std::logic_error("chat usage arrived before its terminal choice");
      const bool limited = finish_reason_ == "length";
      if (!active_ && output_.empty() && !Open(false))
        return false;
      if (!Close(limited ? "incomplete" : "completed") ||
          !CloseFunctions(limited ? "incomplete" : "completed"))
        return false;
      response_["output"] = Output();
      response_["usage"] = ResponsesUsage(*usage);
      if (const auto* timings = event.find("timings"))
        response_["timings"] = *timings;
      response_["status"] = limited ? "incomplete" : "completed";
      if (limited)
        response_["incomplete_details"]["reason"] = "max_output_tokens";
      else
        response_["completed_at"] = Now();
      terminal_ = true;
      return ResponseEvent(limited ? "response.incomplete"
                                   : "response.completed");
    }
    if (const auto* choices = event.find("choices")) {
      for (const auto& choice : choices->items()) {
        if (const auto* delta = choice.find("delta")) {
          if (!Text(delta->member_str("reasoning_content"), true) ||
              !Text(delta->member_str("content"), false))
            return false;
          if (const auto* tools = delta->find("tool_calls")) {
            for (const auto& tool : tools->items()) {
              if (!Function(tool))
                return false;
            }
          }
        }
        const auto finish = choice.member_str("finish_reason");
        if (!finish.empty())
          finish_reason_ = finish;
      }
    }
    return connected_;
  }

  [[nodiscard]] bool terminal() const { return terminal_; }

private:
  bool Emit(const char* type, json::Value event) {
    if (!connected_)
      return false;
    event["type"] = type;
    event["sequence_number"] = sequence_++;
    connected_ = writer_("event: " + std::string(type) +
                         "\ndata: " + event.dump() + "\n\n");
    return connected_;
  }

  bool ResponseEvent(const char* type) {
    auto event = json::Value::object();
    event["response"] = response_;
    return Emit(type, std::move(event));
  }

  json::Value Output() const {
    auto output = json::Value::array();
    for (const auto& item : output_)
      output.push_back(item);
    return output;
  }

  json::Value ItemEvent() const {
    auto event = json::Value::object();
    event["output_index"] = active_index_;
    return event;
  }

  json::Value ContentEvent() const {
    auto event = ItemEvent();
    event["item_id"] = item_id_;
    event["content_index"] = 0;
    return event;
  }

  bool Open(bool reasoning) {
    active_ = true;
    reasoning_ = reasoning;
    item_id_ = (reasoning ? "rs_" : "msg_") + RandomId();
    text_.clear();
    active_index_ = output_.size();
    output_.push_back(
        ResponsesItem(item_id_, {}, reasoning_, "in_progress", false));
    auto event = ItemEvent();
    event["item"] = output_.back();
    if (!Emit("response.output_item.added", std::move(event)))
      return false;
    event = ContentEvent();
    event["part"] = ResponsesPart({}, reasoning_);
    return Emit("response.content_part.added", std::move(event));
  }

  bool Text(const std::string& text, bool reasoning) {
    if (text.empty())
      return true;
    if (active_ && reasoning != reasoning_ && !Close("completed"))
      return false;
    if (!active_ && !Open(reasoning))
      return false;
    text_ += text;
    auto event = ContentEvent();
    event["delta"] = text;
    if (!reasoning_)
      event["logprobs"] = json::Value::array();
    return Emit(reasoning_ ? "response.reasoning_text.delta"
                           : "response.output_text.delta",
                std::move(event));
  }

  bool Close(const char* status) {
    if (!active_)
      return true;
    auto event = ContentEvent();
    event["text"] = text_;
    if (!reasoning_)
      event["logprobs"] = json::Value::array();
    if (!Emit(reasoning_ ? "response.reasoning_text.done"
                         : "response.output_text.done",
              std::move(event)))
      return false;
    event = ContentEvent();
    event["part"] = ResponsesPart(text_, reasoning_);
    if (!Emit("response.content_part.done", std::move(event)))
      return false;
    event = ItemEvent();
    auto item = ResponsesItem(item_id_, text_, reasoning_, status);
    event["item"] = item;
    if (!Emit("response.output_item.done", std::move(event)))
      return false;
    output_[active_index_] = std::move(item);
    active_ = false;
    return true;
  }

  bool Function(const json::Value& delta) {
    if (!Close("completed"))
      return false;
    const auto index = delta.member_size("index");
    auto found = functions_.find(index);
    const auto* function = delta.find("function");
    if (found == functions_.end()) {
      if (!response_.find("parallel_tool_calls")->as_bool() &&
          !functions_.empty())
        throw std::runtime_error(
            "multiple calls violate parallel_tool_calls=false");
      const auto call_id = delta.member_str("id");
      const auto name = function ? function->member_str("name") : "";
      if (call_id.empty() || name.empty())
        throw std::logic_error("chat tool delta is missing call identity");
      const auto output_index = output_.size();
      auto item = ResponsesFunctionItem("fc_" + RandomId(), call_id, name, "",
                                        "in_progress");
      auto event = json::Value::object();
      event["output_index"] = output_index;
      event["item"] = item;
      output_.push_back(std::move(item));
      found = functions_.emplace(index, output_index).first;
      if (!Emit("response.output_item.added", std::move(event)))
        return false;
    }
    auto& item = output_[found->second];
    if ((delta.contains("id") &&
         delta.member_str("id") != item.member_str("call_id")) ||
        (function && function->contains("name") &&
         function->member_str("name") != item.member_str("name")))
      throw std::logic_error("chat tool delta changed call identity");
    const auto* arguments = function ? function->find("arguments") : nullptr;
    if (arguments) {
      if (!arguments->is_string())
        throw std::logic_error("chat function arguments delta is not a string");
      item["arguments"] = item.member_str("arguments") + arguments->str();
      if (!arguments->str().empty()) {
        auto event = json::Value::object();
        event["output_index"] = found->second;
        event["item_id"] = item.member_str("id");
        event["delta"] = *arguments;
        return Emit("response.function_call_arguments.delta", std::move(event));
      }
    }
    return true;
  }

  bool CloseFunctions(const char* status) {
    // Calls may be interleaved by Chat index. Keep Responses output indices in
    // first-appearance order and finalize only after the entire batch succeeds.
    for (auto& item : output_) {
      if (item.member_str("type") != "function_call" ||
          item.member_str("status") != "in_progress")
        continue;
      const auto index = static_cast<std::size_t>(&item - output_.data());
      auto event = json::Value::object();
      event["output_index"] = index;
      event["item_id"] = item.member_str("id");
      event["name"] = item.member_str("name");
      event["arguments"] = item.member_str("arguments");
      if (!Emit("response.function_call_arguments.done", std::move(event)))
        return false;
      item["status"] = status;
      event = json::Value::object();
      event["output_index"] = index;
      event["item"] = item;
      if (!Emit("response.output_item.done", std::move(event)))
        return false;
    }
    return true;
  }

  bool Fail(const json::Value& error) {
    if (terminal_)
      return connected_;
    if (!Close("incomplete") || !CloseFunctions("incomplete"))
      return false;
    auto event = error;
    event["param"] = json::Value();
    if (!Emit("error", std::move(event)))
      return false;
    response_["status"] = "failed";
    response_["error"] = error;
    response_["output"] = Output();
    terminal_ = true;
    return ResponseEvent("response.failed");
  }

  json::Value response_;
  const HttpResponse::BodyWriter& writer_;
  std::vector<json::Value> output_;
  std::map<std::size_t, std::size_t> functions_;
  std::size_t active_index_{0};
  std::string item_id_;
  std::string text_;
  std::string finish_reason_;
  std::size_t sequence_{0};
  bool connected_{true};
  bool started_{false};
  bool active_{false};
  bool reasoning_{false};
  bool terminal_{false};
};

HttpResponse OpenAiResponses(const HttpRequest& req,
                             TextGenerationBackend& b) try {
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (const std::exception& e) {
    return Err(400, "Bad Request", e.what(), "invalid_request_error",
               "parse_error");
  }

  std::size_t max_tokens = 0;
  sampling::SamplingConfig sampling_config;
  if (auto error = ReadCompatibilityOptions(
          body, b, "max_output_tokens", &max_tokens, &sampling_config, true)) {
    return std::move(*error);
  }

  if (const auto* include = body.find("include")) {
    if (!include->is_array() ||
        std::ranges::any_of(include->items(), [](const auto& item) {
          return !item.is_string() ||
                 item.str() != "reasoning.encrypted_content";
        }))
      return InvalidCompatibilityRequest(
          "only include:['reasoning.encrypted_content'] is supported as an "
          "advisory request; Gufo returns raw reasoning");
  }
  if (const auto* metadata = body.find("metadata");
      metadata && !metadata->is_null()) {
    if (!metadata->is_object() || metadata->size() > 16)
      return InvalidCompatibilityRequest(
          "metadata must contain at most 16 string pairs");
    for (const auto& [key, value] : metadata->members()) {
      if (key.size() > 64 || !value.is_string() || value.str().size() > 512)
        return InvalidCompatibilityRequest(
            "metadata keys/values exceed string limits");
    }
  }
  auto chat_body = json::Value::object();
  for (const auto& [name, value] : body.members()) {
    if (name != "input" && name != "instructions" &&
        name != "max_output_tokens" && name != "reasoning" && name != "tools" &&
        name != "tool_choice" && name != "parallel_tool_calls" &&
        name != "text")
      chat_body[name] = value;
  }
  if (auto error = NormalizeResponsesTools(body, &chat_body))
    return std::move(*error);
  if (auto error = NormalizeResponsesFormat(body, &chat_body))
    return std::move(*error);
  if (auto error = NormalizeResponsesInput(body, &chat_body["messages"]))
    return std::move(*error);
  if (const auto* reasoning = body.find("reasoning")) {
    if (!reasoning->is_object())
      return InvalidCompatibilityRequest("'reasoning' must be an object");
    for (const auto& [name, value] : reasoning->members()) {
      if (name != "effort")
        return InvalidCompatibilityRequest(
            "only 'reasoning.effort' is supported");
      if (const auto* top = body.find("reasoning_effort");
          top && top->dump() != value.dump())
        return InvalidCompatibilityRequest(
            "reasoning.effort and reasoning_effort disagree");
      chat_body["reasoning_effort"] = value;
    }
  }
  chat_body["model"] = b.model_id();
  chat_body["max_completion_tokens"] = max_tokens;
  const bool stream = body.find("stream") && body.find("stream")->as_bool();
  if (stream)
    chat_body["stream_options"]["include_usage"] = true;
  auto chat_request = req;
  chat_request.body = chat_body.dump();
  auto response = HandleOpenAiChat(chat_request, b);
  if (response.status != 200)
    return response;
  auto envelope = ResponsesEnvelope(body, chat_body, b.model_id(), max_tokens,
                                    sampling_config);
  if (stream) {
    auto source = std::move(response.streaming_body);
    auto source_log = response.stream_log;
    // A protocol failure is delivered as a complete Responses event stream,
    // not an unterminated HTTP chunked body. Retain its diagnostic as text.
    response.stream_log = std::make_shared<HttpResponse::StreamLog>();
    response.streaming_body =
        [source = std::move(source), envelope = std::move(envelope), source_log,
         log = response.stream_log](const HttpResponse::BodyWriter& writer) {
          ResponsesStream adapter(envelope, writer);
          source([&](std::string_view frame) { return adapter.Push(frame); });
          if (source_log) {
            log->details = source_log->details;
            if (!source_log->error_code.empty()) {
              log->details += " error_code=" + source_log->error_code;
              if (!adapter.terminal())
                log->error_code = source_log->error_code;
            }
          }
        };
    return response;
  }
  const auto chat = json::parse(response.body);
  const auto& choice = chat.find("choices")->items().front();
  const auto& message = *choice.find("message");
  const bool limited = choice.member_str("finish_reason") == "length";
  const char* status = limited ? "incomplete" : "completed";
  const auto reasoning = message.member_str("reasoning_content");
  const auto text = message.member_str("content");
  const auto* calls = message.find("tool_calls");
  const bool has_calls = calls && !calls->empty();
  if (has_calls && calls->size() > 1 &&
      !envelope.find("parallel_tool_calls")->as_bool())
    return Err(502, "Bad Gateway",
               "model generated parallel calls when disabled", "server_error",
               "tool_choice_unsatisfied");
  if (!reasoning.empty())
    envelope["output"].push_back(
        ResponsesItem("rs_" + RandomId(), reasoning, true,
                      text.empty() && !has_calls ? status : "completed"));
  if (!text.empty() || (reasoning.empty() && !has_calls))
    envelope["output"].push_back(ResponsesItem(
        "msg_" + RandomId(), text, false, has_calls ? "completed" : status));
  if (has_calls) {
    for (const auto& call : calls->items()) {
      const auto& function = *call.find("function");
      envelope["output"].push_back(
          ResponsesFunctionItem("fc_" + RandomId(), call.member_str("id"),
                                function.member_str("name"),
                                function.member_str("arguments"), status));
    }
  }
  envelope["status"] = status;
  if (limited)
    envelope["incomplete_details"]["reason"] = "max_output_tokens";
  else
    envelope["completed_at"] = Now();
  envelope["usage"] = ResponsesUsage(*chat.find("usage"));
  envelope["timings"] = *chat.find("timings");
  response.body = envelope.dump();
  return response;
} catch (const std::length_error& error) {
  return Err(400, "Bad Request", error.what(), "invalid_request_error",
             "context_length_exceeded");
} catch (const std::invalid_argument& error) {
  return Err(400, "Bad Request", error.what(), "invalid_request_error",
             "invalid_prompt");
}

HttpResponse AnthropicMessages(const HttpRequest& req,
                               TextGenerationBackend& b) try {
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (const std::exception& e) {
    return Err(400, "Bad Request", e.what(), "invalid_request_error",
               "parse_error");
  }

  std::size_t max_tokens = 0;
  sampling::SamplingConfig sampling_config;
  if (auto error = ReadCompatibilityOptions(body, b, "max_tokens", &max_tokens,
                                            &sampling_config)) {
    return std::move(*error);
  }

  std::vector<tokenization::ChatMessage> messages;
  if (const auto* system = body.find("system")) {
    std::string text;
    if (!ReadTextContent(system, &text)) {
      return InvalidCompatibilityRequest("'system' must contain only text");
    }
    messages.push_back(
        {tokenization::ChatRole::kSystem, std::move(text), "", ""});
  }
  if (!ReadTextMessages(body.find("messages"), &messages)) {
    return InvalidCompatibilityRequest(
        "'messages' must contain text messages; use /v1/chat/completions "
        "for images and tools");
  }

  ChatRequest chat{std::move(messages)};
  chat.client_id = req.client_id;
  const auto res = b.chat(chat, max_tokens, sampling_config, req.is_cancelled);

  json::Value resp = json::Value::object();
  resp["id"] = "msg_" + RandomId();
  resp["type"] = "message";
  resp["role"] = "assistant";
  resp["model"] = b.model_id();
  json::Value content = json::Value::array();
  json::Value txt = json::Value::object();
  txt["type"] = "text";
  txt["text"] = core::Utf8Decoder{}.Push(res.text, true);
  content.push_back(std::move(txt));
  resp["content"] = std::move(content);
  resp["stop_reason"] =
      res.finish_reason == TextGenerationBackend::FinishReason::kLength
          ? "max_tokens"
          : "end_turn";
  resp["stop_sequence"] = json::Value();
  json::Value usage = json::Value::object();
  usage["input_tokens"] = res.prompt_tokens;
  usage["output_tokens"] = res.completion_tokens;
  usage["cache_creation_input_tokens"] = 0;
  usage["cache_read_input_tokens"] = res.cached_prompt_tokens;
  resp["usage"] = std::move(usage);
  resp["timings"] = GenerationTimings(res);
  return WithTiming(Ok(resp), res);
} catch (const std::length_error& error) {
  return Err(400, "Bad Request", error.what(), "invalid_request_error",
             "context_length_exceeded");
} catch (const std::invalid_argument& error) {
  return Err(400, "Bad Request", error.what(), "invalid_request_error",
             "invalid_prompt");
}

HttpResponse LlamaCompletion(const HttpRequest& req,
                             TextGenerationBackend& b) try {
  json::Value body;
  try {
    body = json::parse(req.body);
  } catch (const std::exception& e) {
    return Err(400, "Bad Request", e.what(), "invalid_request_error",
               "parse_error");
  }

  std::size_t max_tokens = 0;
  sampling::SamplingConfig sampling_config;
  if (auto error = ReadCompatibilityOptions(body, b, "n_predict", &max_tokens,
                                            &sampling_config)) {
    return std::move(*error);
  }

  const auto* input = body.find("prompt");
  if (input == nullptr || !input->is_string() || input->str().empty()) {
    return InvalidCompatibilityRequest("'prompt' must be a nonempty string");
  }
  const std::string prompt = input->str();

  const auto res = b.complete(prompt, max_tokens, sampling_config,
                              req.is_cancelled, {}, req.client_id);

  json::Value resp = json::Value::object();
  resp["content"] = core::Utf8Decoder{}.Push(res.text, true);
  resp["stop"] = true;
  const bool limited =
      res.finish_reason == TextGenerationBackend::FinishReason::kLength;
  resp["stopped_eos"] = !limited && !res.cancelled;
  resp["stopped_length"] = limited;
  resp["stopped_word"] = false;
  resp["stopped_limit"] = limited;
  resp["stopping_word"] = "";
  resp["tokens_predicted"] = res.completion_tokens;
  resp["tokens_evaluated"] = res.prompt_tokens;
  resp["tokens_cached"] = res.cached_prompt_tokens;
  resp["timings"] = GenerationTimings(res);
  resp["usage"] = UsageJson(res);
  return WithTiming(Ok(resp), res);
} catch (const std::length_error& error) {
  return Err(400, "Bad Request", error.what(), "invalid_request_error",
             "context_length_exceeded");
} catch (const std::invalid_argument& error) {
  return Err(400, "Bad Request", error.what(), "invalid_request_error",
             "invalid_prompt");
}

HttpResponse LlamaProps(const HttpRequest& req, TextGenerationBackend&) {
  const std::string model = req.query_param("model");
  if (model.empty()) {
    return Err(400, "Bad Request", "'model' query parameter is required",
               "invalid_request_error", "missing_model");
  }
  json::Value resp = json::Value::object();
  resp["model"] = model;
  resp["template"] = "";
  json::Value model_info = json::Value::object();
  resp["model_info"] = std::move(model_info);
  return Ok(resp);
}

HttpResponse LlamaSlots(const HttpRequest&, TextGenerationBackend& b) {
  json::Value resp = json::Value::array();
  json::Value slot = json::Value::object();
  slot["id"] = 0;
  slot["task_id"] = 0;
  slot["state"] = 0;
  slot["prompt"] = "";
  slot["next_token"] = json::Value();
  slot["model"] = b.model_id();
  resp.push_back(std::move(slot));
  return Ok(resp);
}

HttpResponse LlamaMetrics(const HttpRequest&, TextGenerationBackend&) {
  std::ostringstream out;
  out << "# HELP llamacpp:prompt_tokens_total Total prompt tokens processed\n"
      << "# TYPE llamacpp:prompt_tokens_total counter\n"
      << "llamacpp:prompt_tokens_total "
      << detail::TotalPromptTokens().load(std::memory_order_relaxed) << "\n"
      << "# HELP llamacpp:tokens_predicted_total Total tokens generated\n"
      << "# TYPE llamacpp:tokens_predicted_total counter\n"
      << "llamacpp:tokens_predicted_total "
      << detail::TotalGenTokens().load(std::memory_order_relaxed) << "\n"
      << "# HELP llamacpp:prompt_tokens_seconds Prompt processing speed in "
         "tokens per second\n"
      << "# TYPE llamacpp:prompt_tokens_seconds gauge\n"
      << "llamacpp:prompt_tokens_seconds "
      << detail::LastPromptSpeed().load(std::memory_order_relaxed) << "\n"
      << "# HELP llamacpp:predicted_tokens_seconds Generation speed in tokens "
         "per second\n"
      << "# TYPE llamacpp:predicted_tokens_seconds gauge\n"
      << "llamacpp:predicted_tokens_seconds "
      << detail::LastGenSpeed().load(std::memory_order_relaxed) << "\n"
      << "# HELP llamacpp:kv_cache_usage_ratio KV cache usage ratio\n"
      << "# TYPE llamacpp:kv_cache_usage_ratio gauge\n"
      << "llamacpp:kv_cache_usage_ratio 0.0\n";
  return {.status = 200,
          .reason = "OK",
          .body = out.str(),
          .headers = {{"Content-Type", "text/plain; version=0.0.4"}}};
}

}  // namespace

// ---------------------------------------------------------------------------
// HttpRequest
// ---------------------------------------------------------------------------

std::string HttpRequest::query_param(const std::string& key) const {
  std::string_view remaining = query;
  while (!remaining.empty()) {
    const auto end = remaining.find('&');
    const auto parameter = remaining.substr(0, end);
    const auto equals = parameter.find('=');
    if (UrlDecode(parameter.substr(0, equals)) == key) {
      return equals == std::string_view::npos
                 ? ""
                 : UrlDecode(parameter.substr(equals + 1));
    }
    if (end == std::string_view::npos) {
      break;
    }
    remaining.remove_prefix(end + 1);
  }
  return "";
}

// ---------------------------------------------------------------------------
// HttpServer
// ---------------------------------------------------------------------------

struct HttpServer::ConnectionWorker {
  int fd{-1};  // Protected by workers_mutex_, including shutdown and close.
  std::atomic<bool> done{false};
  std::jthread thread;
};

HttpServer::HttpServer(std::string host, int port,
                       std::shared_ptr<TextGenerationBackend> backend,
                       std::shared_ptr<VideoJobService> video_jobs,
                       std::shared_ptr<TtsService> tts,
                       std::shared_ptr<AsrService> asr,
                       HttpServerOptions options,
                       std::shared_ptr<ImageService> images)
    : host_(std::move(host)),
      port_(port),
      backend_(std::move(backend)),
      video_jobs_(std::move(video_jobs)),
      tts_(std::move(tts)),
      asr_(std::move(asr)),
      images_(std::move(images)),
      options_(std::move(options)) {
  if (options_.max_request_body_bytes == 0 || options_.max_connections == 0) {
    throw std::invalid_argument("HTTP server limits must be positive");
  }
  if (!options_.api_key.empty()) {
    api_key_hash_ = CredentialHash(options_.api_key);
  }
  register_routes();
}

HttpServer::~HttpServer() {
  stop();
  if (listen_fd_ >= 0) {
    ::close(listen_fd_);
  }
}

void HttpServer::add(const std::string& method, const std::string& path,
                     Handler handler) {
  routes_.emplace_back(std::make_pair(method, path), std::move(handler));
}

void HttpServer::register_routes() {
  if (backend_ == nullptr) {
    return;
  }
  // ---- OpenAI ----
  add("POST", "/v1/completions", OpenAiCompletions);
  add("POST", "/v1/chat/completions", HandleOpenAiChat);
  add("POST", "/v1/responses", OpenAiResponses);
  add("POST", "/v1/embeddings", NotImplemented);

  // ---- Anthropic ----
  add("POST", "/v1/messages", AnthropicMessages);
  add("POST", "/v1/messages/count_tokens", NotImplemented);

  // ---- llama-server ----
  add("POST", "/v1/rerank", NotImplemented);
  add("POST", "/v1/reranking", NotImplemented);
  add("POST", "/rerank", NotImplemented);
  add("POST", "/infill", NotImplemented);
  add("POST", "/completion", LlamaCompletion);
  add("GET", "/props", LlamaProps);
  add("GET", "/slots", LlamaSlots);
  add("GET", "/v1/slots", LlamaSlots);
  add("GET", "/metrics", LlamaMetrics);
  add("GET", "/v1/metrics", LlamaMetrics);

  // ---- sdapi ----
  add("POST", "/sdapi/v1/txt2img", NotImplemented);
  add("POST", "/sdapi/v1/img2img", NotImplemented);
  add("GET", "/sdapi/v1/loras", NotImplemented);
}

bool HttpServer::start(std::string* error) {
  if (listen_fd_ >= 0 || stopped_.load(std::memory_order_acquire)) {
    if (error != nullptr) {
      *error = "HTTP server has already been started or stopped";
    }
    return false;
  }
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  if (port_ < 0 || port_ > 65535 ||
      ::inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) != 1) {
    if (error != nullptr) {
      *error =
          "host must be an IPv4 address and port must be between 0 and 65535";
    }
    return false;
  }
  addr.sin_port = htons(static_cast<unsigned short>(port_));
  (void)::signal(SIGPIPE, SIG_IGN);
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) {
    if (error != nullptr)
      *error = "socket() failed";
    return false;
  }
  const int yes = 1;
  ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

  if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) <
      0) {
    if (error != nullptr) {
      *error = "bind() failed on " + host_ + ":" + std::to_string(port_);
    }
    ::close(listen_fd_);
    listen_fd_ = -1;
    return false;
  }
  if (::listen(listen_fd_, 16) < 0) {
    if (error != nullptr)
      *error = "listen() failed";
    ::close(listen_fd_);
    listen_fd_ = -1;
    return false;
  }
  if (port_ == 0) {
    sockaddr_in bound{};
    socklen_t length = sizeof(bound);
    if (::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&bound),
                      &length) != 0) {
      if (error != nullptr) {
        *error = "getsockname() failed";
      }
      ::close(listen_fd_);
      listen_fd_ = -1;
      return false;
    }
    port_ = ntohs(bound.sin_port);
  }
  return true;
}

void HttpServer::run() {
  Logger::Info(
      "server",
      "event=listening address=http://" + host_ + ":" + std::to_string(port_) +
          " auth=" + (options_.api_key.empty() ? "off" : "bearer") +
          " max_connections=" + std::to_string(options_.max_connections) +
          " max_body_bytes=" + std::to_string(options_.max_request_body_bytes));
  while (!stopped_.load(std::memory_order_acquire)) {
    const int client_fd = ::accept(listen_fd_, nullptr, nullptr);
    if (client_fd < 0) {
      if (stopped_.load(std::memory_order_acquire)) {
        break;
      }
      continue;
    }
    if (stopped_.load(std::memory_order_acquire)) {
      ::close(client_fd);
      break;
    }

    reap_workers();
    auto worker = std::make_unique<ConnectionWorker>();
    ConnectionWorker* const worker_ptr = worker.get();
    bool overloaded = false;
    {
      const std::lock_guard<std::mutex> lock(workers_mutex_);
      overloaded = stopped_.load(std::memory_order_acquire) ||
                   workers_.size() >= options_.max_connections;
      if (!overloaded) {
        worker_ptr->fd = client_fd;
        worker_ptr->thread = std::jthread([this, worker_ptr, client_fd] {
          handle_connection(client_fd);
          {
            const std::lock_guard<std::mutex> lock(workers_mutex_);
            ::close(worker_ptr->fd);
            worker_ptr->fd = -1;
          }
          worker_ptr->done.store(true, std::memory_order_release);
        });
        workers_.push_back(std::move(worker));
      }
    }
    if (overloaded) {
      HttpResponse response =
          Err(503, "Service Unavailable", "connection limit reached",
              "server_error", "overloaded");
      response.headers.emplace_back("Retry-After", "1");
      (void)SendAll(client_fd, BuildResponse(response));
      ::close(client_fd);
    }
  }
  reap_workers();
}

void HttpServer::stop() {
  const bool was_stopped = stopped_.exchange(true, std::memory_order_acq_rel);
  if (!was_stopped && listen_fd_ >= 0) {
    // Keep the descriptor owned until destruction: run() may still be inside
    // accept(). Closing here allows it to observe a reused descriptor.
    (void)::shutdown(listen_fd_, SHUT_RDWR);
  }

  std::vector<std::unique_ptr<ConnectionWorker>> workers;
  {
    const std::lock_guard<std::mutex> lock(workers_mutex_);
    workers = std::move(workers_);
    for (const auto& worker : workers) {
      if (worker->fd >= 0) {
        (void)::shutdown(worker->fd, SHUT_RDWR);
      }
    }
  }
}

void HttpServer::reap_workers() {
  std::vector<std::unique_ptr<ConnectionWorker>> finished;
  {
    const std::lock_guard<std::mutex> lock(workers_mutex_);
    auto iterator = workers_.begin();
    while (iterator != workers_.end()) {
      if ((*iterator)->done.load(std::memory_order_acquire)) {
        finished.push_back(std::move(*iterator));
        iterator = workers_.erase(iterator);
      } else {
        ++iterator;
      }
    }
  }
}

HttpResponse HttpServer::handle_request(const HttpRequest& req) {
  if (!IsAuthorized(req, api_key_hash_)) {
    auto response = Err(401, "Unauthorized", "missing or invalid API key",
                        "authentication_error", "invalid_api_key");
    response.headers.emplace_back("WWW-Authenticate", "Bearer");
    return response;
  }
  // `/health` and `/ready` are the native spelling and match llama-server's
  // `/health`. `/healthz` and `/readyz` are aliases so Kubernetes-style probe
  // configuration works unmodified.
  if (req.method == "GET" &&
      (req.path == "/health" || req.path == "/v1/health" ||
       req.path == "/healthz")) {
    json::Value body = json::Value::object();
    body["status"] = "ok";
    return Ok(body);
  }
  if (req.method == "GET" && (req.path == "/ready" || req.path == "/v1/ready" ||
                              req.path == "/readyz")) {
    const bool ready = (backend_ != nullptr && backend_->ready()) ||
                       (video_jobs_ != nullptr && video_jobs_->ready()) ||
                       (tts_ != nullptr && tts_->ready()) ||
                       (asr_ != nullptr && asr_->ready()) || images_ != nullptr;
    if (!ready) {
      return Err(503, "Service Unavailable", "model service is not ready",
                 "server_error", "not_ready");
    }
    json::Value body = json::Value::object();
    body["status"] = "ready";
    if (backend_ != nullptr && backend_->ready()) {
      body["model"] = backend_->model_id();
    } else if (asr_ != nullptr && asr_->ready()) {
      body["model"] = asr_->model_id();
    } else if (tts_ != nullptr && tts_->ready()) {
      body["model"] = tts_->model_id();
    } else if (images_ != nullptr) {
      body["model"] = images_->model_id();
    } else {
      body["model"] = "minimax-h3";
    }
    return Ok(body);
  }
  if (req.method == "GET" &&
      (req.path == "/v1/models" || req.path == "/models")) {
    return ListModels(backend_.get(), video_jobs_.get(), tts_.get(), asr_.get(),
                      images_.get());
  }
  if (req.path == "/v1/images/generations" || req.path == "/v1/images/edits") {
    if (!images_)
      return Err(503, "Service Unavailable", "image service is not configured",
                 "server_error", "image_service_unavailable");
    return HandleImageApiRequest(req, *images_);
  }
  if (req.path == "/v1/audio/speech/stream") {
    if (!tts_ || !tts_->ready())
      return Err(503, "Service Unavailable", "TTS is not configured",
                 "server_error", "tts_service_unavailable");
    return HandleTtsWebSocket(req, *tts_);
  }
  if (req.path == "/v1/realtime") {
    if (!asr_ || !asr_->ready())
      return Err(503, "Service Unavailable", "ASR is not configured",
                 "server_error", "asr_service_unavailable");
    return HandleAsrWebSocket(req, *asr_);
  }
  if (IsVideoApiPath(req.path)) {
    if (video_jobs_ == nullptr || !video_jobs_->ready()) {
      return Err(503, "Service Unavailable",
                 "MiniMax H3 video service is not configured", "server_error",
                 "video_service_unavailable");
    }
    return HandleVideoApiRequest(req, *video_jobs_);
  }
  if (IsAudioTtsApiPath(req.path)) {
    if (tts_ == nullptr || !tts_->ready()) {
      return Err(503, "Service Unavailable",
                 "Qwen3-TTS audio service is not configured", "server_error",
                 "tts_service_unavailable");
    }
    return HandleAudioTtsApiRequest(req, *tts_);
  }
  if (IsAudioAsrApiPath(req.path)) {
    if (asr_ == nullptr || !asr_->ready()) {
      return Err(503, "Service Unavailable",
                 "Qwen3-ASR service is not configured", "server_error",
                 "asr_service_unavailable");
    }
    return HandleAudioAsrApiRequest(req, *asr_);
  }
  for (const auto& entry : routes_) {
    if (entry.first.first == req.method && entry.first.second == req.path) {
      return entry.second(req, *backend_);
    }
  }
  return Err(404, "Not Found", "no route for this path",
             "invalid_request_error", "not_found");
}

void HttpServer::handle_connection(int client_fd) {
  const struct timeval tv{120, 0};
  ::setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  ::setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

  const auto start_time = std::chrono::steady_clock::now();
  HttpRequest req;
  static std::atomic<std::uint64_t> next_request{0};
  sockaddr_in peer{};
  socklen_t peer_size = sizeof(peer);
  char peer_address[INET_ADDRSTRLEN]{};
  if (::getpeername(client_fd, reinterpret_cast<sockaddr*>(&peer),
                    &peer_size) == 0 &&
      ::inet_ntop(AF_INET, &peer.sin_addr, peer_address, sizeof(peer_address)))
    req.client_id = peer_address;
  req.request_id = "r" + std::to_string(next_request.fetch_add(1) + 1);
  bool response_started = false;
  bool http11 = false;
  int response_status = 0;
  const auto elapsed_ms = [&] {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start_time)
        .count();
  };
  try {
    bool ok = false;
    bool payload_too_large = false;
    {
      std::string buffer;
      if (ReadUntil(buffer, client_fd, "\r\n\r\n")) {
        // ReadUntil over-reads: `buffer` holds the headers, the blank line, and
        // possibly some body bytes already. Split at the blank line and carry
        // the over-read body bytes forward so we only read the remainder from
        // the socket.
        const auto delim_pos = buffer.find("\r\n\r\n");
        const std::string headers = buffer.substr(0, delim_pos);
        std::string body = buffer.substr(delim_pos + 4);

        const auto eol = headers.find("\r\n");
        const std::string line =
            (eol == std::string::npos) ? headers : headers.substr(0, eol);
        std::istringstream ls(line);
        std::string target;
        std::string version;
        std::string extra;
        ls >> req.method >> target >> version;
        http11 = version == "HTTP/1.1";
        bool valid_headers = !req.method.empty() && target.starts_with('/') &&
                             (version == "HTTP/1.0" || version == "HTTP/1.1") &&
                             !(ls >> extra);
        const auto qpos = target.find('?');
        if (qpos != std::string::npos) {
          req.query = target.substr(qpos + 1);
          req.path = target.substr(0, qpos);
        } else {
          req.path = target;
        }

        std::size_t cursor =
            eol == std::string::npos ? headers.size() : eol + 2;
        while (cursor < headers.size()) {
          const std::size_t next = headers.find("\r\n", cursor);
          const std::size_t line_end =
              next == std::string::npos ? headers.size() : next;
          const std::string_view header_line(headers.data() + cursor,
                                             line_end - cursor);
          const std::size_t colon = header_line.find(':');
          if (colon != std::string_view::npos && colon > 0) {
            std::string name(header_line.substr(0, colon));
            std::string_view raw_value = header_line.substr(colon + 1);
            while (!raw_value.empty() &&
                   std::isspace(
                       static_cast<unsigned char>(raw_value.front())) != 0) {
              raw_value.remove_prefix(1);
            }
            while (!raw_value.empty() &&
                   std::isspace(static_cast<unsigned char>(raw_value.back())) !=
                       0) {
              raw_value.remove_suffix(1);
            }
            req.headers.emplace_back(std::move(name), std::string(raw_value));
          } else {
            valid_headers = false;
          }
          if (next == std::string::npos) {
            break;
          }
          cursor = next + 2;
        }

        const auto parsed_length = ParseContentLength(req);
        const std::size_t content_length = parsed_length.value_or(0);
        const std::size_t remaining =
            content_length > body.size() ? content_length - body.size() : 0;
        if (!valid_headers || !parsed_length.has_value()) {
          ok = false;
        } else if (content_length > options_.max_request_body_bytes) {
          payload_too_large = true;
        } else if (content_length > 0) {
          if (remaining > 0) {
            ok = ReadN(body, client_fd, remaining);
          } else {
            ok = true;
          }
          if (body.size() > content_length) {
            body.resize(content_length);
          }
        } else {
          ok = true;
          if (!IsWebSocketUpgrade(req))
            body.clear();
        }
        if (ok) {
          req.body = std::move(body);
          req.is_cancelled = [client_fd] {
            return IsPeerDisconnected(client_fd);
          };
        }
      }
    }

    // Successful health/metrics polling and video status polling stay quiet.
    const bool log_request =
        req.method == "POST" || req.method == "DELETE" ||
        req.path == "/v1/models" || req.path.ends_with("/content") ||
        req.path == "/v1/realtime" || req.path == "/v1/audio/speech/stream";
    if (ok && log_request) {
      Logger::Info("http", "request=" + req.request_id +
                               " event=received method=" + req.method +
                               " path=" + req.path + " body_bytes=" +
                               std::to_string(req.body.size()));
    }

    HttpResponse resp;
    if (payload_too_large) {
      resp = Err(413, "Payload Too Large", "request body is too large",
                 "invalid_request_error", "payload_too_large");
    } else if (!ok) {
      resp = Err(400, "Bad Request", "malformed request",
                 "invalid_request_error", "bad_request");
    } else if (req.method == "OPTIONS") {
      resp = {.status = 204, .reason = "No Content"};
    } else {
      resp = handle_request(req);
    }

    resp.headers.emplace_back("X-Request-ID", req.request_id);
    response_status = resp.status;
    bool connected = true;
    if (resp.websocket) {
      response_started = true;
      connected = SendAll(client_fd, BuildResponseHead(resp, std::nullopt));
      if (connected) {
        WebSocket socket(client_fd, std::move(req.body));
        resp.websocket(socket);
      }
    } else if (resp.streaming_body) {
      const bool chunked = http11 && !HasHeader(resp, "content-length");
      if (chunked)
        resp.headers.emplace_back("Transfer-Encoding", "chunked");
      const auto head = BuildResponseHead(resp, std::nullopt);
      response_started = true;
      connected = SendAll(client_fd, head);
      if (connected) {
        resp.streaming_body([&](std::string_view chunk) {
          connected = connected && (chunked ? SendChunk(client_fd, chunk)
                                            : SendAll(client_fd, chunk));
          return connected;
        });
        // Without the final chunk, HTTP clients report an incomplete body.
        // Closing an unframed PCM stream would silently look like shorter
        // audio.
        if (connected && chunked &&
            (!resp.stream_log || resp.stream_log->error_code.empty()))
          connected = SendAll(client_fd, "0\r\n\r\n");
      }
    } else {
      const auto payload = BuildResponse(resp);
      response_started = true;
      connected = SendAll(client_fd, payload);
    }
    std::string outcome = connected ? "completed" : "disconnected";
    if (resp.stream_log) {
      resp.log_details = resp.stream_log->details;
      if (!resp.stream_log->error_code.empty()) {
        outcome = "stream_error";
        resp.log_details += " error_code=" + resp.stream_log->error_code;
      }
    }
    if (resp.status >= 400) {
      try {
        const auto body = json::parse(resp.body);
        if (const auto* error = body.find("error"))
          resp.log_details += " error_code=" + error->member_str("code");
      } catch (const std::exception&) {
        // The status remains useful for an endpoint returning a non-JSON error.
      }
    }
    if (log_request || resp.status >= 400 || !connected) {
      Logger::LogRequest(req.request_id, req.method, req.path, resp.status,
                         elapsed_ms(), resp.log_details, outcome);
    }
  } catch (const TextGenerationError& exception) {
    const auto duration_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - start_time)
                                 .count();
    const char* reason = "Service Unavailable";
    if (exception.http_status() == 408) {
      reason = "Request Timeout";
    } else if (exception.http_status() == 429) {
      reason = "Too Many Requests";
    }
    HttpResponse resp = Err(exception.http_status(), reason, exception.what(),
                            "server_error", exception.stable_code());
    resp.headers.emplace_back("X-Request-ID", req.request_id);
    if (exception.retryable()) {
      resp.headers.emplace_back("Retry-After", "1");
    }
    Logger::LogRequest(req.request_id, req.method, req.path,
                       response_started ? response_status : resp.status,
                       duration_ms,
                       std::string("error_code=") + exception.stable_code(),
                       response_started ? "stream_error" : "failed");
    if (!response_started)
      (void)SendAll(client_fd, BuildResponse(resp));
  } catch (const std::exception& e) {
    const auto duration_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - start_time)
                                 .count();
    HttpResponse resp = Err(500, "Internal Server Error", e.what(),
                            "internal_error", "server_exception");
    resp.headers.emplace_back("X-Request-ID", req.request_id);
    Logger::LogRequest(req.request_id, req.method, req.path,
                       response_started ? response_status : resp.status,
                       duration_ms, "error_code=server_exception",
                       response_started ? "stream_error" : "failed");
    if (!response_started)
      (void)SendAll(client_fd, BuildResponse(resp));
  } catch (...) {
    const auto duration_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - start_time)
                                 .count();
    HttpResponse resp =
        Err(500, "Internal Server Error", "unknown server error",
            "internal_error", "server_exception");
    resp.headers.emplace_back("X-Request-ID", req.request_id);
    Logger::LogRequest(req.request_id, req.method, req.path,
                       response_started ? response_status : resp.status,
                       duration_ms, "error_code=server_exception",
                       response_started ? "stream_error" : "failed");
    if (!response_started)
      (void)SendAll(client_fd, BuildResponse(resp));
  }
}

}  // namespace gufo::server
