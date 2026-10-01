#include "src/cli/serve/http_server.hpp"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <semaphore>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include "src/cli/serve/logging.hpp"

namespace {

using gufo::server::HttpServer;
using gufo::server::TextGenerationBackend;

class FakeBackend final : public TextGenerationBackend {
public:
  struct Call {
    std::string prompt;
    gufo::server::ChatRequest chat;
    std::size_t max_tokens = 0;
    gufo::sampling::SamplingConfig sampling;
    std::string client_id;
  };
  Call LastCall() {
    const std::lock_guard lock(mutex_);
    return last_;
  }
  void SetOutput(std::string text) {
    const std::lock_guard lock(mutex_);
    output_ = std::move(text);
  }
  std::string model_id() const override { return "test"; }
  bool ready() const override { return true; }
  bool supports_json_constraints() const override { return true; }
  InitialOutputState initial_output_state(
      const gufo::server::ChatRequest& request) const override {
    return request.reasoning.enabled.value_or(false)
               ? InitialOutputState::kReasoning
               : InitialOutputState::kContent;
  }
  std::shared_ptr<GenerationRequest> start_chat(
      const gufo::server::ChatRequest& request, std::size_t limit,
      const gufo::sampling::SamplingConfig& sampling,
      const CancellationCheck& cancel, bool stream) override {
    if (reject_chat)
      throw gufo::server::TextGenerationError(
          gufo::server::TextGenerationErrorCode::kQueueFull, "queue full");
    return TextGenerationBackend::start_chat(request, limit, sampling, cancel,
                                             stream);
  }
  std::size_t count_tokens(std::string_view text) const override {
    return text.size();
  }
  Result complete(std::string_view prompt, std::size_t limit,
                  const gufo::sampling::SamplingConfig& sampling,
                  const CancellationCheck& cancel, const TokenCallback& token,
                  std::string_view client_id = "anonymous") override {
    ++calls;
    if (failure == 1)
      throw std::length_error("context exceeded");
    if (failure == 2)
      throw std::invalid_argument("invalid prompt");
    Result result;
    if (wait_for_disconnect) {
      entered.release();
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (!(disconnected = cancel && cancel()) &&
             std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      finished.release();
      result.cancelled = disconnected;
      return result;
    }
    {
      const std::lock_guard lock(mutex_);
      last_ = {.prompt = std::string(prompt),
               .max_tokens = limit,
               .sampling = sampling,
               .client_id = std::string(client_id)};
      result.text = output_;
    }
    result.prompt_tokens = 10;
    result.cached_prompt_tokens = 8;
    result.cache_hit = true;
    result.draft_accepted_tokens = 4;
    result.draft_tokens = 8;
    result.prefill_tokens = 2;
    result.prefill_ms = 4;
    result.completion_tokens = 1;
    result.decode_ms = 2;
    result.finish_reason =
        limit == 1 ? FinishReason::kLength : FinishReason::kStop;
    if (token) {
      bool first_piece = true;
      for (char byte : result.text) {
        if ((cancel && cancel()) || !token(std::string_view(&byte, 1))) {
          result.cancelled = true;
          return result;
        }
        if (wait_after_piece && first_piece) {
          first_piece = false;
          entered.release();
          const auto deadline =
              std::chrono::steady_clock::now() + std::chrono::seconds(2);
          while (!release_output && !(disconnected = cancel && cancel()) &&
                 std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
          finished.release();
          if (disconnected) {
            result.cancelled = true;
            return result;
          }
        }
      }
    }
    if (failure == 3)
      throw gufo::server::TextGenerationError(
          gufo::server::TextGenerationErrorCode::kOutputLimit, "output limit");
    return result;
  }
  Result chat(const gufo::server::ChatRequest& request, std::size_t limit,
              const gufo::sampling::SamplingConfig& sampling,
              const CancellationCheck& cancel,
              const TokenCallback& token) override {
    auto result =
        complete("", limit, sampling, cancel, token, request.client_id);
    {
      const std::lock_guard lock(mutex_);
      last_.chat = request;
    }
    return result;
  }
  std::atomic<int> calls{0};
  std::atomic<int> failure{0};
  bool reject_chat{false};
  bool wait_after_piece{false};
  std::atomic<bool> release_output{false};
  bool wait_for_disconnect{false};
  std::atomic<bool> disconnected{false};
  std::binary_semaphore entered{0};
  std::binary_semaphore finished{0};

private:
  std::mutex mutex_;
  Call last_;
  std::string output_{"ok"};
};

class RunningServer {
public:
  explicit RunningServer(gufo::server::HttpServerOptions options = {})
      : backend(std::make_shared<FakeBackend>()),
        server("127.0.0.1", 0, backend, nullptr, nullptr, nullptr,
               std::move(options)) {
    server.add("POST", "/echo", [](const auto& request, auto&) {
      return gufo::server::HttpResponse{.body = request.body};
    });
    server.add("POST", "/stream-error", [](const auto&, auto&) {
      return gufo::server::HttpResponse{
          .streaming_body = [](const auto& write) {
            (void)write("first chunk");
            throw std::runtime_error("injected stream failure");
          }};
    });
    server.add("POST", "/stream", [](const auto& request, auto&) {
      auto log = std::make_shared<gufo::server::HttpResponse::StreamLog>();
      return gufo::server::HttpResponse{
          .streaming_body =
              [log, fail = request.body == "fail"](const auto& write) {
                (void)write(std::string_view("a\0b", 3));
                (void)write("");
                (void)write("end");
                if (fail)
                  log->error_code = "injected";
              },
          .stream_log = log,
      };
    });
    std::string error;
    assert(server.start(&error));
    worker = std::jthread([this] { server.run(); });
  }
  ~RunningServer() {
    server.stop();
    worker.join();
  }

  int Connect() {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    assert(fd >= 0);
    const timeval timeout{3, 0};
    assert(::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                        sizeof(timeout)) == 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(server.port());
    assert(::inet_pton(AF_INET, "127.0.0.1", &address.sin_addr) == 1);
    assert(::connect(fd, reinterpret_cast<const sockaddr*>(&address),
                     sizeof(address)) == 0);
    return fd;
  }
  std::string Send(std::string_view request, bool half_close = false) {
    const int fd = Connect();
    while (!request.empty()) {
      const auto count =
          ::send(fd, request.data(), request.size(), MSG_NOSIGNAL);
      assert(count > 0);
      request.remove_prefix(static_cast<std::size_t>(count));
    }
    // A half-close allows malformed/truncated-body tests to complete without
    // timing-dependent sleeps.
    if (half_close)
      ::shutdown(fd, SHUT_WR);
    std::string response;
    char buffer[4096];
    for (;;) {
      const auto count = ::read(fd, buffer, sizeof(buffer));
      assert(count >= 0);
      if (count == 0) {
        break;
      }
      response.append(buffer, static_cast<std::size_t>(count));
    }
    ::close(fd);
    return response;
  }

  std::string Post(std::string_view path, std::string_view body) {
    return Send("POST " + std::string(path) + " HTTP/1.1\r\nContent-Length: " +
                std::to_string(body.size()) + "\r\n\r\n" + std::string(body));
  }

  std::shared_ptr<FakeBackend> backend;
  HttpServer server;
  std::jthread worker;
};

void ExpectStatus(const std::string& response, int status) {
  if (!response.starts_with("HTTP/1.1 " + std::to_string(status) + " ")) {
    std::cerr << response << '\n';
    std::abort();
  }
}

void TestAuthorization() {
  RunningServer secured({.api_key = "test-secret"});
  for (const std::string path :
       {"/health", "/ready", "/v1/models", "/v1/chat/completions",
        "/v1/responses", "/v1/messages", "/v1/audio/speech",
        "/v1/audio/transcriptions", "/v1/video/generations", "/echo"}) {
    ExpectStatus(secured.Send("POST " + path + " HTTP/1.1\r\n\r\n"), 401);
  }
  for (const std::string header :
       {"", "Authorization: Bearer wrong\r\n",
        "Authorization: Basic test-secret\r\n",
        "Authorization: Bearer test-secret\r\nAuthorization: Bearer "
        "test-secret\r\n"}) {
    const auto response =
        secured.Send("GET /v1/models HTTP/1.1\r\n" + header + "\r\n");
    ExpectStatus(response, 401);
    assert(response.find("WWW-Authenticate: Bearer") != std::string::npos);
    assert(response.find("test-secret") == std::string::npos);
  }
  assert(secured.backend->calls == 0);
  ExpectStatus(secured.Send("GET /v1/models HTTP/1.1\r\naUtHoRiZaTiOn: bEaReR  "
                            "test-secret\r\n\r\n"),
               200);
  ExpectStatus(secured.Send("OPTIONS /v1/chat/completions HTTP/1.1\r\n\r\n"),
               204);
  const std::string body = R"({"prompt":"hi","max_tokens":1})";
  ExpectStatus(secured.Send("POST /v1/completions HTTP/1.1\r\nAuthorization: "
                            "Bearer test-secret\r\n"
                            "Content-Length: " +
                            std::to_string(body.size()) + "\r\n\r\n" + body),
               200);
  assert(secured.backend->calls == 1);
}

void TestRequestLogging() {
  std::ostringstream output;
  auto* previous = std::clog.rdbuf(output.rdbuf());
  std::string request_id;
  {
    RunningServer server;
    ExpectStatus(server.Send("GET /health HTTP/1.1\r\n\r\n"), 200);
    const auto response = server.Post(
        "/v1/chat/completions?private-query",
        R"({"model":"test","messages":[{"role":"user","content":"private-prompt"}],"stream":true})");
    ExpectStatus(response, 200);
    const auto header = response.find("X-Request-ID: ");
    assert(header != std::string::npos);
    const auto begin = header + std::string("X-Request-ID: ").size();
    request_id = response.substr(begin, response.find("\r\n", begin) - begin);

    const auto failed = server.Post("/stream-error", "");
    ExpectStatus(failed, 200);
    assert(failed.find("first chunk") != std::string::npos);
    assert(failed.find("HTTP/1.1", 1) == std::string::npos);
  }
  gufo::server::Logger::Info("test", "escaped\n\x1b[31m");
  std::clog.rdbuf(previous);
  const auto log = output.str();
  assert(log.find("request=" + request_id + " event=received") !=
         std::string::npos);
  assert(log.find("request=" + request_id + " event=completed") !=
         std::string::npos);
  assert(log.find("cache=memory") != std::string::npos);
  assert(log.find("acceptance_pct=50.0") != std::string::npos);
  assert(log.find("rss_mib=") != std::string::npos);
  assert(log.find("error_code=server_exception") != std::string::npos);
  assert(log.find("path=/health") == std::string::npos);
  assert(log.find("private-query") == std::string::npos);
  assert(log.find("private-prompt") == std::string::npos);
  assert(log.find("escaped\\x0a\\x1b[31m") != std::string::npos);
}

void TestFramingAndMetrics() {
  RunningServer server({.max_request_body_bytes = 8192});
  ExpectStatus(server.Send("GET /health HTTP/1.1\r\n\r\n"), 200);
  for (const std::string header :
       {"Content-Length: nope", "Content-Length: -1", "Content-Length: 4junk",
        "Content-Length: 18446744073709551616",
        "Content-Length: 4\r\nContent-Length: 3", "Transfer-Encoding: chunked",
        "broken-header"}) {
    ExpectStatus(server.Send("POST /echo HTTP/1.1\r\n" + header + "\r\n\r\n"),
                 400);
  }
  ExpectStatus(server.Send("POST /echo\r\n\r\n"), 400);
  ExpectStatus(server.Send("POST /echo HTTP/1.1 extra\r\n\r\n"), 400);
  ExpectStatus(
      server.Send("POST /echo HTTP/1.1\r\nContent-Length: 4\r\n\r\nx", true),
      400);
  ExpectStatus(
      server.Send("POST /echo HTTP/1.1\r\nContent-Length: 8193\r\n\r\n"), 413);
  const std::string payload(5000, 'x');
  const auto echo = server.Send(
      "POST /echo HTTP/1.1\r\nContent-Length: 5000\r\nContent-Length: "
      "5000\r\n\r\n" +
      payload);
  ExpectStatus(echo, 200);
  assert(echo.substr(echo.find("\r\n\r\n") + 4) == payload);

  const std::string body = R"({"prompt":"hi","n_predict":1})";
  const auto response =
      server.Send("POST /completion HTTP/1.1\r\nContent-Length: " +
                  std::to_string(body.size()) + "\r\n\r\n" + body);
  ExpectStatus(response, 200);
  const auto parsed =
      gufo::json::parse(response.substr(response.find("\r\n\r\n") + 4));
  const auto* timings = parsed.find("timings");
  assert(timings != nullptr);
  assert(timings->member_double("prompt_n") == 2);
  assert(timings->member_double("cache_n") == 8);
  assert(timings->member_double("prompt_per_second") == 500);
  assert(timings->member_double("prompt_per_token_ms") == 2);
}

void TestCompatibilityRequests() {
  RunningServer server;
  using gufo::json::parse;
  const auto response_body = [](const std::string& response) {
    ExpectStatus(response, 200);
    return parse(response.substr(response.find("\r\n\r\n") + 4));
  };
  struct Endpoint {
    const char *path, *body, *limit;
  };
  for (const auto& endpoint : {
           Endpoint{"/v1/completions", R"({"prompt":"hi"})", "max_tokens"},
           Endpoint{"/v1/responses", R"({"input":"hi"})", "max_output_tokens"},
           Endpoint{"/v1/messages",
                    R"({"messages":[{"role":"user","content":"hi"}]})",
                    "max_tokens"},
           Endpoint{"/completion", R"({"prompt":"hi"})", "n_predict"},
       }) {
    auto body = parse(endpoint.body);
    body["model"] = "test";
    body[endpoint.limit] = 1;
    body["temperature"] = 0.6;
    body["top_k"] = 40;
    body["top_p"] = 0.9;
    body["seed"] = 123;
    body["repeat_penalty"] = 1.1;
    const auto output = response_body(server.Post(endpoint.path, body.dump()));
    const auto last = server.backend->LastCall();
    for (int failure : {1, 2}) {
      server.backend->failure = failure;
      const auto rejected = server.Post(endpoint.path, body.dump());
      ExpectStatus(rejected, 400);
      assert(rejected.find(failure == 1
                               ? "context_length_exceeded"
                               : "invalid_prompt") != std::string::npos);
    }
    server.backend->failure = 0;
    assert(last.client_id == "127.0.0.1");
    assert(last.max_tokens == 1 && last.sampling.temperature == 0.6F &&
           last.sampling.top_k == 40 && last.sampling.top_p == 0.9F &&
           last.sampling.seed == 123 && last.sampling.repeat_penalty == 1.1F);
    if (std::string_view(endpoint.path) == "/v1/responses") {
      assert(output.member_str("status") == "incomplete");
      assert(output.find("incomplete_details")->member_str("reason") ==
             "max_output_tokens");
      assert(output.find("usage")->contains("input_tokens_details"));
    } else if (std::string_view(endpoint.path) == "/v1/messages") {
      assert(output.member_str("stop_reason") == "max_tokens");
    } else if (std::string_view(endpoint.path) == "/completion") {
      assert(output.find("stopped_length")->as_bool());
      assert(!output.find("stopped_eos")->as_bool());
    } else {
      assert(output.find("choices")->items()[0].member_str("finish_reason") ==
             "length");
    }
    const int calls = server.backend->calls;
    for (const auto value : {"0", "-1", "1.5", "1e100", "\"1\"", "null"}) {
      auto invalid = body;
      invalid[endpoint.limit] = parse(value);
      ExpectStatus(server.Post(endpoint.path, invalid.dump()), 400);
    }
    for (const auto field : {"stream", "echo", "store", "background", "tools",
                             "stop", "reasoning", "logit_bias"}) {
      auto invalid = body;
      invalid[field] = true;
      if (std::string_view(endpoint.path) == "/v1/responses" &&
          std::string_view(field) == "stream")
        invalid[field] = "true";
      ExpectStatus(server.Post(endpoint.path, invalid.dump()), 400);
    }
    auto invalid = body;
    invalid["n"] = 1.4;
    ExpectStatus(server.Post(endpoint.path, invalid.dump()), 400);
    invalid = body;
    invalid["model"] = "wrong";
    ExpectStatus(server.Post(endpoint.path, invalid.dump()), 404);
    ExpectStatus(server.Post(endpoint.path, "[]"), 400);
    ExpectStatus(server.Post(endpoint.path, "{"), 400);
    assert(server.backend->calls == calls);
  }
  const int calls = server.backend->calls;
  ExpectStatus(server.Post("/v1/completions", R"({"prompt":["one","two"]})"),
               400);
  ExpectStatus(server.Post("/v1/responses",
                           R"({"input":[{"role":"user","content":[
                           {"type":"input_text","text":"describe"},
                           {"type":"input_image","image_url":"data:image/png;base64,AA=="}]}]})"),
               400);
  ExpectStatus(server.Post("/v1/messages",
                           R"({"messages":[{"role":"tool","content":"hi"}]})"),
               400);
  ExpectStatus(server.Post("/v1/messages", R"({"messages":[]})"), 400);
  ExpectStatus(
      server.Post("/infill", R"({"input_prefix":"one","input_suffix":"two"})"),
      501);
  ExpectStatus(server.Post("/v1/messages/count_tokens",
                           R"({"messages":[{"role":"user","content":"hi"}]})"),
               501);
  assert(server.backend->calls == calls);

  const auto response = response_body(server.Post(
      "/v1/responses",
      R"({"instructions":"Be concise.","input":[{"role":"user","content":[
          {"type":"input_text","text":"hi"}]}],"max_output_tokens":2,"store":false})"));
  assert(response.member_str("status") == "completed");
  const auto messages = server.backend->LastCall().chat.messages;
  assert(messages.size() == 2);
  assert(messages[0].role == gufo::tokenization::ChatRole::kSystem &&
         messages[0].content == "Be concise." && messages[1].content == "hi");

  const auto anthropic = response_body(
      server.Post("/v1/messages",
                  R"({"system":[{"type":"text","text":"Be concise."}],
          "messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],
          "max_tokens":2})"));
  assert(anthropic.member_str("stop_reason") == "end_turn");
  assert(server.backend->LastCall().chat.messages[0].content == "Be concise.");
}

std::vector<gufo::json::Value> ResponsesEvents(const std::string& response) {
  ExpectStatus(response, 200);
  assert(response.find("Content-Type: text/event-stream\r\n") !=
         std::string::npos);
  assert(response.find("Transfer-Encoding: chunked\r\n") != std::string::npos);
  std::string payload;
  std::size_t cursor = response.find("\r\n\r\n") + 4;
  for (;;) {
    const auto end = response.find("\r\n", cursor);
    assert(end != std::string::npos);
    const auto size =
        std::stoull(response.substr(cursor, end - cursor), nullptr, 16);
    cursor = end + 2;
    assert(cursor + size + 2 <= response.size());
    payload.append(response, cursor, size);
    cursor += size;
    assert(response.substr(cursor, 2) == "\r\n");
    cursor += 2;
    if (!size)
      break;
  }
  assert(cursor == response.size());
  std::vector<gufo::json::Value> events;
  cursor = 0;
  while (cursor < payload.size()) {
    assert(payload.compare(cursor, 7, "event: ") == 0);
    const auto newline = payload.find('\n', cursor);
    const auto type = payload.substr(cursor + 7, newline - cursor - 7);
    cursor = newline + 1;
    assert(payload.compare(cursor, 6, "data: ") == 0);
    const auto end = payload.find("\n\n", cursor);
    assert(end != std::string::npos);
    auto event =
        gufo::json::parse(payload.substr(cursor + 6, end - cursor - 6));
    assert(event.member_str("type") == type);
    assert(event.find("sequence_number") &&
           event.member_size("sequence_number") == events.size());
    assert(!event.contains("choices"));
    events.push_back(std::move(event));
    cursor = end + 2;
  }
  assert(events.size() >= 3);
  assert(events[0].member_str("type") == "response.created");
  assert(events[1].member_str("type") == "response.in_progress");
  const auto* first = events.front().find("response");
  assert(first && first->member_str("id").starts_with("resp_") &&
         first->member_str("status") == "in_progress" &&
         first->find("output")->empty() && first->find("usage")->is_null());
  for (const auto& event : events) {
    if (const auto* envelope = event.find("response"))
      assert(envelope->member_str("id") == first->member_str("id") &&
             envelope->member_size("created_at") ==
                 first->member_size("created_at"));
  }
  return events;
}

void TestResponsesStreaming() {
  RunningServer server;
  struct Case {
    const char* raw;
    const char* reasoning;
    const char* text;
  };
  for (const auto& item : {Case{"Answer", "", "Answer"}, Case{"", "", ""},
                           Case{"Think</think>Answer", "Think", "Answer"},
                           Case{"Think", "Think", ""},
                           Case{"é中😀\xE2\x94!", "", "é中😀\xEF\xBF\xBD!"}}) {
    for (const auto limit : {1, 32}) {
      auto request = gufo::json::parse(R"({"input":"Hi","stream":true})");
      request["max_output_tokens"] = limit;
      if (*item.reasoning)
        request["reasoning"]["effort"] = "high";
      server.backend->SetOutput(item.raw);
      const auto events =
          ResponsesEvents(server.Post("/v1/responses", request.dump()));
      assert(events.back().member_str("type") ==
             (limit == 1 ? "response.incomplete" : "response.completed"));
      const auto& final = *events.back().find("response");
      assert(final.member_str("status") ==
             (limit == 1 ? "incomplete" : "completed"));
      assert(final.member_str("object") == "response" &&
             final.member_str("model") == "test");
      assert(final.find("error")->is_null());
      if (limit == 1)
        assert(final.find("incomplete_details")->member_str("reason") ==
               "max_output_tokens");
      else
        assert(final.find("incomplete_details")->is_null() &&
               final.contains("completed_at"));
      const auto* usage = final.find("usage");
      assert(
          usage->member_size("input_tokens") == 10 &&
          usage->member_size("output_tokens") == 1 &&
          usage->member_size("total_tokens") == 11 &&
          usage->find("input_tokens_details")->member_size("cached_tokens") ==
              8);
      assert(!usage->contains("output_tokens_details"));
      assert(final.find("timings")->member_size("prompt_n") == 2);
      assert(server.backend->LastCall().max_tokens ==
             static_cast<std::size_t>(limit));

      std::size_t cursor = 2;
      const auto& output = final.find("output")->items();
      assert(output.size() == (*item.reasoning && *item.text ? 2 : 1));
      for (std::size_t index = 0; index < output.size(); ++index) {
        const auto& result = output[index];
        const bool reasoning = result.member_str("type") == "reasoning";
        const std::string expected = reasoning ? item.reasoning : item.text;
        const auto id = result.member_str("id");
        assert(id.starts_with(reasoning ? "rs_" : "msg_"));
        assert(result.member_str("status") ==
               (limit == 1 && index + 1 == output.size() ? "incomplete"
                                                         : "completed"));
        if (reasoning)
          assert(result.find("summary")->empty());
        else
          assert(result.member_str("role") == "assistant");
        const auto& added = events[cursor++];
        assert(added.member_str("type") == "response.output_item.added" &&
               added.member_size("output_index") == index &&
               added.find("item")->member_str("id") == id &&
               added.find("item")->member_str("status") == "in_progress" &&
               added.find("item")->find("content")->empty());
        const auto& part = events[cursor++];
        assert(part.member_str("type") == "response.content_part.added" &&
               part.member_str("item_id") == id &&
               part.member_size("output_index") == index &&
               part.member_size("content_index") == 0 &&
               part.find("part")->member_str("text").empty());
        const std::string prefix =
            reasoning ? "response.reasoning_text." : "response.output_text.";
        std::string text;
        while (events[cursor].member_str("type") == prefix + "delta") {
          const auto& delta = events[cursor++];
          assert(delta.member_str("item_id") == id &&
                 delta.member_size("output_index") == index &&
                 delta.member_size("content_index") == 0);
          text += delta.member_str("delta");
          assert(expected.starts_with(text));
          if (!reasoning)
            assert(delta.find("logprobs")->empty());
        }
        assert(text == expected);
        const auto& done = events[cursor++];
        assert(done.member_str("type") == prefix + "done" &&
               done.member_str("item_id") == id &&
               done.member_str("text") == expected);
        const auto& part_done = events[cursor++];
        assert(part_done.member_str("type") == "response.content_part.done" &&
               part_done.find("part")->dump() ==
                   result.find("content")->items()[0].dump());
        const auto& item_done = events[cursor++];
        assert(item_done.member_str("type") == "response.output_item.done" &&
               item_done.find("item")->dump() == result.dump());
      }
      assert(cursor + 1 == events.size());
      request["stream"] = false;
      const auto buffered = server.Post("/v1/responses", request.dump());
      ExpectStatus(buffered, 200);
      const auto complete =
          gufo::json::parse(buffered.substr(buffered.find("\r\n\r\n") + 4));
      assert(complete.member_str("status") == final.member_str("status") &&
             complete.find("usage")->dump() == final.find("usage")->dump());
      const auto& buffered_output = complete.find("output")->items();
      assert(buffered_output.size() == output.size());
      for (std::size_t index = 0; index < output.size(); ++index) {
        assert(buffered_output[index].find("content")->dump() ==
               output[index].find("content")->dump());
        assert(buffered_output[index].member_str("type") ==
               output[index].member_str("type"));
      }
    }
  }
}

void TestResponsesReasoningHistoryRoundtrip() {
  RunningServer server;
  const auto complete = [&](const gufo::json::Value& request, bool stream) {
    const auto response = server.Post("/v1/responses", request.dump());
    ExpectStatus(response, 200);
    if (stream) {
      const auto events = ResponsesEvents(response);
      assert(events.back().member_str("type") == "response.completed");
      return *events.back().find("response");
    }
    return gufo::json::parse(response.substr(response.find("\r\n\r\n") + 4));
  };
  for (bool first_stream : {false, true}) {
    for (bool second_stream : {false, true}) {
      auto request = gufo::json::parse(R"({
        "input":[{"role":"user","content":"First question"}],
        "store":false,"reasoning":{"effort":"high"},
        "chat_template_kwargs":{"preserve_thinking":true}
      })");
      request["stream"] = first_stream;
      server.backend->SetOutput("First reasoning</think>First answer");
      const auto first = complete(request, first_stream);
      const auto& output = first.find("output")->items();
      assert(output.size() == 2 &&
             output[0].member_str("type") == "reasoning" &&
             output[1].member_str("type") == "message");
      // Replay the actual output objects, including IDs, status and content
      // part types, exactly as a stateless Responses client does.
      for (const auto& item : output)
        request["input"].push_back(item);
      request["input"].push_back(
          gufo::json::parse(R"({"role":"user","content":"Second question"})"));
      request["stream"] = second_stream;
      server.backend->SetOutput("Second reasoning</think>Second answer");
      const auto second = complete(request, second_stream);
      const auto history = server.backend->LastCall().chat;
      assert(history.messages.size() == 3);
      assert(history.messages[0].role == gufo::tokenization::ChatRole::kUser &&
             history.messages[0].content == "First question");
      assert(history.messages[1].role ==
                 gufo::tokenization::ChatRole::kAssistant &&
             history.messages[1].content == "First answer" &&
             history.messages[1].thought == "First reasoning");
      assert(history.messages[2].role == gufo::tokenization::ChatRole::kUser &&
             history.messages[2].content == "Second question");
      assert(history.reasoning.preserve_thinking == true);
      const auto rendered = gufo::tokenization::QwenChatTemplate::Render(
          history.messages,
          gufo::tokenization::ResolveQwenChatOptions(history.reasoning));
      assert(rendered &&
             rendered->find(
                 "<think>\nFirst reasoning\n</think>\n\nFirst answer") !=
                 std::string::npos);
      const auto& second_output = second.find("output")->items();
      assert(second_output.size() == 2 &&
             second_output[0].find("content")->items()[0].member_str("text") ==
                 "Second reasoning" &&
             second_output[1].find("content")->items()[0].member_str("text") ==
                 "Second answer");
    }
  }
}

void TestResponsesReasoningOnlyHistoryRoundtrip() {
  RunningServer server;
  const auto complete = [&](const gufo::json::Value& request, bool stream) {
    const auto response = server.Post("/v1/responses", request.dump());
    ExpectStatus(response, 200);
    if (stream) {
      const auto events = ResponsesEvents(response);
      return *events.back().find("response");
    }
    return gufo::json::parse(response.substr(response.find("\r\n\r\n") + 4));
  };
  for (bool first_stream : {false, true}) {
    auto first_request = gufo::json::parse(R"({
      "input":[{"role":"user","content":"First question"}],
      "max_output_tokens":1,"store":false,"reasoning":{"effort":"high"}
    })");
    first_request["stream"] = first_stream;
    server.backend->SetOutput("Unfinished reasoning");
    const auto first = complete(first_request, first_stream);
    assert(first.member_str("status") == "incomplete" &&
           first.find("incomplete_details")->member_str("reason") ==
               "max_output_tokens");
    const auto& output = *first.find("output");
    assert(output.size() == 1 &&
           output.items()[0].member_str("type") == "reasoning" &&
           output.items()[0].member_str("status") == "incomplete");
    for (bool second_stream : {false, true}) {
      for (bool append_user : {false, true}) {
        for (bool include_original : {false, true}) {
          auto request = gufo::json::parse(R"({
            "input":[],"max_output_tokens":8,"store":false,
            "reasoning":{"effort":"high"},
            "chat_template_kwargs":{"preserve_thinking":true}
          })");
          if (include_original)
            request["input"].push_back(first_request.find("input")->items()[0]);
          for (const auto& item : output.items())
            request["input"].push_back(item);
          if (append_user)
            request["input"].push_back(
                gufo::json::parse(R"({"role":"user","content":"Continue"})"));
          request["stream"] = second_stream;
          server.backend->SetOutput("Finished reasoning</think>Answer");
          const auto second = complete(request, second_stream);
          assert(second.member_str("status") == "completed");
          const auto history = server.backend->LastCall().chat;
          const std::size_t index = include_original ? 1 : 0;
          assert(history.messages.size() == index + 1 + (append_user ? 1 : 0));
          assert(history.messages[index].role ==
                     gufo::tokenization::ChatRole::kAssistant &&
                 history.messages[index].content.empty() &&
                 history.messages[index].thought == "Unfinished reasoning");
          if (append_user)
            assert(history.messages.back().role ==
                       gufo::tokenization::ChatRole::kUser &&
                   history.messages.back().content == "Continue" &&
                   history.messages.back().thought.empty());
          if (include_original || append_user) {
            const auto rendered = gufo::tokenization::QwenChatTemplate::Render(
                history.messages,
                gufo::tokenization::ResolveQwenChatOptions(history.reasoning));
            assert(
                rendered &&
                rendered->find(
                    "<think>\nUnfinished reasoning\n</think>\n\n<|im_end|>") !=
                    std::string::npos);
          }
        }
      }
    }
  }
}

void TestResponsesReasoningHistoryValidation() {
  RunningServer server;
  const auto reasoning = gufo::json::parse(R"({
    "type":"reasoning","id":"rs_previous","status":"completed",
    "summary":[],"content":[{"type":"reasoning_text","text":"First "},
                              {"type":"reasoning_text","text":"thought"}]
  })");
  const auto assistant = gufo::json::parse(R"({
    "type":"message","id":"msg_previous","role":"assistant","status":"completed",
    "content":[{"type":"output_text","text":"First answer","annotations":[],"logprobs":[]}]
  })");
  const auto make_request = [&](const gufo::json::Value& thought) {
    auto request =
        gufo::json::parse(R"({"input":[{"role":"user","content":"First"}]})");
    request["input"].push_back(thought);
    request["input"].push_back(assistant);
    request["input"].push_back(
        gufo::json::parse(R"({"role":"user","content":"Next"})"));
    return request;
  };
  for (bool stream : {false, true}) {
    auto valid = make_request(reasoning);
    valid["stream"] = stream;
    const auto response = server.Post("/v1/responses", valid.dump());
    ExpectStatus(response, 200);
    if (stream)
      (void)ResponsesEvents(response);
    assert(server.backend->LastCall().chat.messages[1].thought ==
           "First thought");

    for (const auto* next_role : {"user", "system", "developer"}) {
      auto request =
          gufo::json::parse(R"({"input":[{"role":"user","content":"First"}]})");
      request["input"].push_back(reasoning);
      auto next = assistant;
      next["role"] = next_role;
      request["input"].push_back(next);
      request["stream"] = stream;
      const auto response = server.Post("/v1/responses", request.dump());
      ExpectStatus(response, 200);
      if (stream)
        (void)ResponsesEvents(response);
      const auto history = server.backend->LastCall().chat.messages;
      assert(history.size() == 3 &&
             history[1].role == gufo::tokenization::ChatRole::kAssistant &&
             history[1].content.empty() &&
             history[1].thought == "First thought" &&
             history[2].thought.empty());
    }
    auto consecutive = make_request(reasoning);
    consecutive["input"] = gufo::json::Value::array();
    consecutive["input"].push_back(reasoning);
    auto next_reasoning = reasoning;
    next_reasoning["content"] = gufo::json::parse(
        R"([{"type":"reasoning_text","text":"Second thought"}])");
    consecutive["input"].push_back(next_reasoning);
    consecutive["input"].push_back(assistant);
    consecutive["stream"] = stream;
    const auto paired = server.Post("/v1/responses", consecutive.dump());
    ExpectStatus(paired, 200);
    if (stream)
      (void)ResponsesEvents(paired);
    const auto history = server.backend->LastCall().chat.messages;
    assert(history.size() == 2 && history[0].content.empty() &&
           history[0].thought == "First thought" &&
           history[1].content == "First answer" &&
           history[1].thought == "Second thought");

    const int before = server.backend->calls;
    const auto reject = [&](gufo::json::Value request) {
      request["stream"] = stream;
      const auto response = server.Post("/v1/responses", request.dump());
      ExpectStatus(response, 400);
      assert(response.find("text/event-stream") == std::string::npos &&
             response.find("invalid_request") != std::string::npos);
    };
    for (
        const auto* malformed :
        {R"({"type":"reasoning"})",
         R"({"type":"reasoning","summary":[{"type":"summary_text","text":"Summary"}]})",
         R"({"type":"reasoning","content":"thought"})",
         R"({"type":"reasoning","content":null})",
         R"({"type":"reasoning","content":[]})",
         R"({"type":"reasoning","content":[{"type":"reasoning_text","text":""}]})",
         R"({"type":"reasoning","content":[{"type":"reasoning_text","text":""},{"type":"reasoning_text","text":""}]})",
         R"({"type":"reasoning","content":[null]})",
         R"({"type":"reasoning","content":[{"type":"output_text","text":"thought"}]})",
         R"({"type":"reasoning","content":[{"type":"reasoning_text"}]})",
         R"({"type":"reasoning","content":[{"type":"reasoning_text","text":7}]})"})
      reject(make_request(gufo::json::parse(malformed)));
    for (const auto* invalid_fields :
         {R"({"id":7})", R"({"id":""})", R"({"status":null})",
          R"({"status":"in_progress"})", R"({"status":"failed"})",
          R"({"summary":null})", R"({"summary":{}})",
          R"({"summary":[{"type":"summary_text","text":"Not raw reasoning"}]})",
          R"({"encrypted_content":"opaque"})", R"({"role":"assistant"})",
          R"({"tool_calls":[]})"}) {
      auto item = reasoning;
      const auto fields = gufo::json::parse(invalid_fields);
      for (const auto& [name, value] : fields.members())
        item[name] = value;
      reject(make_request(item));
    }
    auto conflicting = gufo::json::parse(R"({"input":[]})");
    conflicting["input"].push_back(reasoning);
    auto message = assistant;
    message["reasoning_content"] = "Different reasoning";
    conflicting["input"].push_back(message);
    reject(std::move(conflicting));
    assert(server.backend->calls == before);
  }
  auto anthropic = gufo::json::Value::object();
  anthropic["messages"] = *make_request(reasoning).find("input");
  ExpectStatus(server.Post("/v1/messages", anthropic.dump()), 400);
}

void TestResponsesReasoningAndValidation() {
  RunningServer server;
  for (bool stream : {false, true}) {
    for (const auto* effort :
         {"off", "none", "minimal", "low", "medium", "high", "xhigh", "max"}) {
      auto request =
          gufo::json::parse(R"({"input":"Hi","max_output_tokens":3})");
      request["reasoning_effort"] = effort;
      request["stream"] = stream;
      const auto response = server.Post("/v1/responses", request.dump());
      ExpectStatus(response, 200);
      if (stream)
        (void)ResponsesEvents(response);
      const auto call = server.backend->LastCall();
      const bool enabled = std::string_view(effort) != "off" &&
                           std::string_view(effort) != "none";
      assert(call.chat.reasoning.enabled == enabled);
      assert(call.chat.reasoning.effort.has_value() == enabled);
      assert(call.max_tokens == 3);
    }
    const int before = server.backend->calls;
    for (const auto* invalid :
         {R"({"reasoning_effort":7})", R"({"reasoning_effort":"huge"})",
          R"({"reasoning":{"effort":"huge"}})", R"({"reasoning":true})",
          R"({"reasoning":{"summary":"auto"}})",
          R"({"reasoning":{"effort":"low"},"reasoning_effort":"high"})",
          R"({"reasoning_effort":"high","thinking":{"type":"disabled"}})",
          R"({"tools":{}})", R"({"tool_choice":"sometimes"})",
          R"({"parallel_tool_calls":"false"})", R"({"store":true})",
          R"({"background":true})",
          R"({"previous_response_id":"resp_previous"})",
          R"({"stream_options":{"include_usage":true}})",
          R"({"text":{"format":{"type":"not_a_format"}}})",
          R"({"max_tokens":3})", R"({"max_output_tokens":0})",
          R"({"input":[{"type":"function_call","name":"f"}]})"}) {
      auto request = gufo::json::parse(invalid);
      if (!request.contains("input"))
        request["input"] = "Hi";
      request["stream"] = stream;
      const auto response = server.Post("/v1/responses", request.dump());
      ExpectStatus(response, 400);
      assert(response.find("text/event-stream") == std::string::npos);
    }
    assert(server.backend->calls == before);
    server.backend->reject_chat = true;
    const auto response =
        server.Post("/v1/responses", stream ? R"({"input":"Hi","stream":true})"
                                            : R"({"input":"Hi"})");
    ExpectStatus(response, 429);
    assert(response.find("queue_full") != std::string::npos &&
           response.find("Retry-After: 1") != std::string::npos &&
           response.find("text/event-stream") == std::string::npos);
    server.backend->reject_chat = false;
  }
}

gufo::json::Value CompleteResponses(RunningServer& server,
                                    const gufo::json::Value& request) {
  const auto response = server.Post("/v1/responses", request.dump());
  ExpectStatus(response, 200);
  if (request.find("stream") && request.find("stream")->as_bool()) {
    const auto events = ResponsesEvents(response);
    assert(events.back().member_str("type") == "response.completed" ||
           events.back().member_str("type") == "response.incomplete");
    return *events.back().find("response");
  }
  return gufo::json::parse(response.substr(response.find("\r\n\r\n") + 4));
}

gufo::json::Value FunctionRequest() {
  return gufo::json::parse(R"({"input":"Look up the weather and clock",
    "max_output_tokens":64,"tools":[
      {"type":"function","name":"weather","description":"Local weather",
       "parameters":{"type":"object","properties":{"city":{"type":"string"}}},"strict":false},
      {"type":"function","name":"clock","parameters":{"type":"object","properties":{}}}
    ]})");
}

constexpr const char* kWeatherCall =
    R"(<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>)";
constexpr const char* kClockCall =
    R"(<tool_call>{"name":"clock","arguments":{}}</tool_call>)";

void CheckFunctionEvents(const std::vector<gufo::json::Value>& events) {
  const auto& output = events.back().find("response")->find("output")->items();
  for (std::size_t index = 0; index < output.size(); ++index) {
    const auto& item = output[index];
    if (item.member_str("type") != "function_call")
      continue;
    const auto id = item.member_str("id");
    const auto call_id = item.member_str("call_id");
    assert(id.starts_with("fc_") && call_id.starts_with("call_") &&
           id != call_id);
    std::size_t added = 0, argument_done = 0, item_done = 0;
    std::string arguments;
    for (const auto& event : events) {
      const auto type = event.member_str("type");
      if (const auto* value = event.find("item");
          value && value->member_str("id") == id) {
        assert(event.member_size("output_index") == index);
        if (type == "response.output_item.added") {
          assert(added++ == 0 && !argument_done && !item_done);
          assert(value->member_str("arguments").empty() &&
                 value->member_str("status") == "in_progress" &&
                 value->member_str("name") == item.member_str("name") &&
                 value->member_str("call_id") == call_id);
        } else {
          assert(type == "response.output_item.done" && added == 1 &&
                 argument_done == 1 && item_done++ == 0);
          assert(value->dump() == item.dump());
        }
      }
      if (event.member_str("item_id") != id)
        continue;
      assert(event.member_size("output_index") == index && added == 1 &&
             !item_done && !event.contains("content_index"));
      if (type == "response.function_call_arguments.delta") {
        assert(!argument_done);
        arguments += event.member_str("delta");
        assert(item.member_str("arguments").starts_with(arguments));
      } else {
        assert(type == "response.function_call_arguments.done" &&
               argument_done++ == 0);
        assert(event.member_str("arguments") == arguments &&
               arguments == item.member_str("arguments") &&
               event.member_str("name") == item.member_str("name"));
      }
    }
    assert(added == 1 && argument_done == 1 && item_done == 1);
    for (std::size_t prior = 0; prior < index; ++prior) {
      assert(output[prior].member_str("id") != id);
      if (output[prior].member_str("type") == "function_call")
        assert(output[prior].member_str("call_id") != call_id);
    }
  }
}

void TestResponsesFunctionOutput() {
  RunningServer server;
  struct Case {
    std::string raw;
    bool reasoning;
    std::size_t calls;
    std::size_t prefix;
  };
  for (
      const auto& item : {
          Case{kClockCall, false, 1, 0},
          Case{std::string(kWeatherCall) + kClockCall, false, 2, 0},
          Case{std::string("Thought</think>Checking now.") + kWeatherCall +
                   kClockCall,
               true, 2, 2},
          Case{std::string("Thought</think>") + kClockCall, true, 1, 1},
          Case{
              R"(<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>)",
              false, 1, 0},
      }) {
    for (const auto limit : {1, 64}) {
      for (const bool stream : {false, true}) {
        auto request = FunctionRequest();
        request["max_output_tokens"] = limit;
        request["stream"] = stream;
        request["parallel_tool_calls"] = true;
        if (item.reasoning)
          request["reasoning"]["effort"] = "high";
        server.backend->SetOutput(item.raw);
        const auto wire = server.Post("/v1/responses", request.dump());
        ExpectStatus(wire, 200);
        gufo::json::Value response;
        if (stream) {
          const auto events = ResponsesEvents(wire);
          CheckFunctionEvents(events);
          assert(events.back().member_str("type") ==
                 (limit == 1 ? "response.incomplete" : "response.completed"));
          response = *events.back().find("response");
        } else {
          response = gufo::json::parse(wire.substr(wire.find("\r\n\r\n") + 4));
        }
        assert(response.member_str("status") ==
               (limit == 1 ? "incomplete" : "completed"));
        const auto& output = response.find("output")->items();
        assert(output.size() == item.prefix + item.calls);
        for (std::size_t index = 0; index < output.size(); ++index) {
          const auto& value = output[index];
          if (index < item.prefix) {
            assert(value.member_str("status") == "completed");
            assert(value.member_str("type") ==
                   (index == 0 && item.reasoning ? "reasoning" : "message"));
          } else {
            assert(value.member_str("type") == "function_call" &&
                   value.member_str("id").starts_with("fc_") &&
                   value.member_str("call_id").starts_with("call_") &&
                   value.member_str("status") ==
                       (limit == 1 ? "incomplete" : "completed"));
            assert(!value.contains("content") && !value.contains("role"));
            const auto name = value.member_str("name");
            assert(name == "weather" || name == "clock");
            assert(value.member_str("arguments") ==
                   (name == "weather" ? R"({"city":"Paris"})" : "{}"));
          }
        }
        assert(response.find("usage")->member_size("input_tokens") == 10 &&
               response.find("usage")->member_size("output_tokens") == 1 &&
               response.find("usage")->member_size("total_tokens") == 11 &&
               response.find("usage")
                       ->find("input_tokens_details")
                       ->member_size("cached_tokens") == 8);
        assert(server.backend->LastCall().max_tokens ==
               static_cast<std::size_t>(limit));
        assert(response.find("parallel_tool_calls")->as_bool() &&
               response.member_str("tool_choice") == "auto" &&
               response.find("tools")->size() == 2);
      }
    }
  }
}

void TestResponsesFunctionChoices() {
  RunningServer server;
  for (const bool stream : {false, true}) {
    for (const auto* choice : {
             R"("auto")",
             R"("none")",
             R"("required")",
             R"({"type":"function","name":"clock"})",
             R"({"type":"function","function":{"name":"clock"}})",
         }) {
      auto request = FunctionRequest();
      request["stream"] = stream;
      request["tool_choice"] = gufo::json::parse(choice);
      request["parallel_tool_calls"] = false;
      server.backend->SetOutput(kClockCall);
      const auto response = CompleteResponses(server, request);
      const auto call = server.backend->LastCall();
      assert(call.chat.tools.size() ==
             (request.find("tool_choice")->is_object() ? 1 : 2));
      using ToolChoice = gufo::server::ChatRequest::ToolChoice;
      assert(call.chat.tool_choice == (std::string_view(choice) == R"("none")"
                                           ? ToolChoice::kNone
                                       : std::string_view(choice) == R"("auto")"
                                           ? ToolChoice::kAuto
                                           : ToolChoice::kRequired));
      const auto& output = response.find("output")->items();
      assert(output.size() == 1 &&
             output[0].member_str("type") ==
                 (call.chat.tool_choice == ToolChoice::kNone
                      ? "message"
                      : "function_call"));
      if (request.find("tool_choice")->is_object()) {
        assert(response.find("tool_choice")->member_str("name") == "clock");
        assert(!response.find("tool_choice")->contains("function"));
      }
    }
    auto conventional = FunctionRequest();
    conventional["stream"] = stream;
    conventional["tools"] = gufo::json::parse(R"([
      {"type":"function","function":{"name":"clock","description":"Clock"}}
    ])");
    server.backend->SetOutput(kClockCall);
    auto response = CompleteResponses(server, conventional);
    const auto definition = gufo::json::parse(
        server.backend->LastCall().chat.tools[0].definition_json);
    assert(
        definition.find("function")->member_str("description") == "Clock" &&
        definition.find("function")->find("parameters")->member_str("type") ==
            "object");
    assert(response.find("tools")->items()[0].member_str("name") == "clock");

    conventional["tools"] = gufo::json::Value::array();
    conventional["tool_choice"] = "none";
    server.backend->SetOutput("No tools.");
    response = CompleteResponses(server, conventional);
    assert(server.backend->LastCall().chat.tools.empty());
    assert(response.find("tools")->empty());
  }
}

void TestResponsesFunctionRounds() {
  RunningServer server;
  for (const bool first_stream : {false, true}) {
    for (const bool next_stream : {false, true}) {
      auto request = FunctionRequest();
      request["instructions"] = "Be concise.";
      request["input"] = gufo::json::parse(R"([
        {"role":"system","content":"System context."},
        {"role":"developer","content":[{"type":"input_text","text":"Developer context."}]},
        {"role":"user","content":"Weather and time?"}
      ])");
      request["reasoning"]["effort"] = "high";
      request["chat_template_kwargs"]["preserve_thinking"] = true;
      request["stream"] = first_stream;
      server.backend->SetOutput(
          std::string("Need information</think>Checking.") + kWeatherCall +
          kClockCall);
      const auto first = CompleteResponses(server, request);
      const auto& output = first.find("output")->items();
      assert(output.size() == 4);
      for (const auto& item : output)
        request["input"].push_back(item);
      auto result =
          gufo::json::parse(R"({"type":"function_call_output","output":[]})");
      result["call_id"] = output[3].member_str("call_id");
      result["output"] = gufo::json::parse(R"([
        {"type":"input_text","text":"It is "},{"type":"text","text":"12:00"}
      ])");
      request["input"].push_back(result);
      result["call_id"] = output[2].member_str("call_id");
      result["output"] = R"({"temperature":20})";
      request["input"].push_back(result);
      request["stream"] = next_stream;
      // Restrict current tools without removing historical calls to weather.
      request["tool_choice"] =
          gufo::json::parse(R"({"type":"function","name":"clock"})");
      server.backend->SetOutput(std::string("Check again</think>") +
                                kClockCall);
      const auto second = CompleteResponses(server, request);
      const auto history = server.backend->LastCall().chat;
      const auto& messages = history.messages;
      assert(messages.size() == 7 && history.tools.size() == 1 &&
             history.tools[0].name == "clock");
      using Role = gufo::tokenization::ChatRole;
      assert(messages[0].role == Role::kSystem &&
             messages[0].content == "Be concise.");
      assert(messages[1].role == Role::kSystem &&
             messages[1].content == "System context.");
      assert(messages[2].role == Role::kDeveloper &&
             messages[2].content == "Developer context.");
      assert(messages[3].role == Role::kUser &&
             messages[3].content == "Weather and time?");
      assert(messages[4].role == Role::kAssistant &&
             messages[4].thought == "Need information" &&
             messages[4].content == "Checking." &&
             messages[4].tool_calls.size() == 2);
      for (std::size_t index = 0; index < 2; ++index) {
        const auto& call = messages[4].tool_calls[index];
        assert(call.id == output[index + 2].member_str("call_id"));
        assert(messages[index + 5].role == Role::kTool &&
               messages[index + 5].tool_call_id == call.id);
        assert(messages[index + 5].name == call.name);
      }
      assert(messages[5].content == R"({"temperature":20})" &&
             messages[6].content == "It is 12:00");
      const auto prompt = gufo::tokenization::QwenChatTemplate::Render(
          messages, history.tools,
          gufo::tokenization::ResolveQwenChatOptions(history.reasoning));
      assert(prompt &&
             prompt->find("Be concise.\nSystem context.\nDeveloper context.") !=
                 std::string::npos &&
             prompt->find("Need information\n</think>\n\nChecking.") !=
                 std::string::npos &&
             prompt->find("<function=weather>") != std::string::npos &&
             prompt->find(R"(<tool_response>
{"temperature":20}
</tool_response>
<tool_response>
It is 12:00
</tool_response>)") != std::string::npos);
      for (const auto& item : second.find("output")->items())
        request["input"].push_back(item);
      result["call_id"] =
          second.find("output")->items().back().member_str("call_id");
      result["output"] = "12:01";
      request["input"].push_back(result);
      request["tool_choice"] = "auto";
      server.backend->SetOutput("Ready</think>It is warm and 12:01.");
      const auto third = CompleteResponses(server, request);
      const auto continued = server.backend->LastCall().chat.messages;
      assert(continued.size() == 9 && continued[7].thought == "Check again" &&
             continued[7].content.empty() &&
             continued[7].tool_calls.size() == 1 &&
             continued[8].tool_call_id == continued[7].tool_calls[0].id &&
             continued[8].content == "12:01");
      for (std::size_t index = 0; index < messages.size(); ++index) {
        assert(continued[index].role == messages[index].role &&
               continued[index].content == messages[index].content &&
               continued[index].thought == messages[index].thought &&
               continued[index].tool_call_id == messages[index].tool_call_id);
      }
      for (const auto& item : third.find("output")->items())
        request["input"].push_back(item);
      request["input"].push_back(
          gufo::json::parse(R"({"role":"user","content":"Thanks"})"));
      server.backend->SetOutput("Done</think>Welcome.");
      (void)CompleteResponses(server, request);
      const auto last = server.backend->LastCall().chat.messages;
      assert(last.size() == 11 && last[9].thought == "Ready" &&
             last[9].content == "It is warm and 12:01." &&
             last[10].content == "Thanks");
    }
  }
}

void TestResponsesFunctionValidation() {
  RunningServer server;
  for (const bool stream : {false, true}) {
    const int before = server.backend->calls;
    const auto reject = [&](gufo::json::Value request) {
      request["stream"] = stream;
      const auto response = server.Post("/v1/responses", request.dump());
      ExpectStatus(response, 400);
      assert(response.find("text/event-stream") == std::string::npos);
    };
    for (
        const auto* tools : {
            R"(null)",
            R"({})",
            R"([null])",
            R"([{"type":"web_search"}])",
            R"([{"type":"function"}])",
            R"([{"type":"function","name":7}])",
            R"([{"type":"function","name":"bad name"}])",
            R"([{"type":"function","name":"f","parameters":[]}])",
            R"([{"type":"function","name":"f","parameters":{"type":"array"}}])",
            R"([{"type":"function","name":"f","parameters":{"properties":[]}}])",
            R"([{"type":"function","name":"f","parameters":{"required":"x"}}])",
            R"([{"type":"function","name":"f","parameters":{"required":[1]}}])",
            R"([{"type":"function","name":"f","description":5}])",
            R"([{"type":"function","name":"f","strict":"true"}])",
            R"([{"type":"function","name":"f"},{"type":"function","name":"f"}])",
            R"([{"type":"function","name":"f","function":{"name":"g"}}])",
            R"([{"type":"function","name":"f","unknown_option":true}])",
        }) {
      auto request = FunctionRequest();
      request["tools"] = gufo::json::parse(tools);
      reject(request);
    }
    for (
        const auto* choice : {
            R"(true)",
            R"(null)",
            R"("always")",
            R"({"type":"function","name":"unknown"})",
            R"({"type":"function","name":7})",
            R"({"type":"function","name":"clock","function":{"name":"weather"}})",
            R"({"type":"function","function":{}})",
            R"({"type":"function","function":{"name":"unknown"}})",
            R"({"type":"custom","name":"clock"})",
        }) {
      auto request = FunctionRequest();
      request["tool_choice"] = gufo::json::parse(choice);
      reject(request);
    }
    auto required =
        gufo::json::parse(R"({"input":"Hi","tool_choice":"required"})");
    reject(required);
    const auto call = gufo::json::parse(
        R"({"type":"function_call","id":"fc_prior","call_id":"call_prior","name":"historical","arguments":"{}"})");
    const auto result = gufo::json::parse(
        R"({"type":"function_call_output","call_id":"call_prior","output":""})");
    const auto with_pair = [&](const gufo::json::Value& first,
                               const gufo::json::Value& second) {
      auto request = FunctionRequest();
      request["input"] =
          gufo::json::parse(R"([{"role":"user","content":"Previous"}])");
      request["input"].push_back(first);
      request["input"].push_back(second);
      return request;
    };
    for (const auto* fields : {
             R"({"call_id":""})",
             R"({"call_id":7})",
             R"({"name":""})",
             R"({"name":"bad name"})",
             R"({"arguments":{}})",
             R"({"arguments":""})",
             R"({"arguments":"{"})",
             R"({"arguments":"[]"})",
             R"({"arguments":"null"})",
             R"({"status":"in_progress"})",
             R"({"id":5})",
             R"({"role":"assistant"})",
         }) {
      auto malformed = call;
      const auto values = gufo::json::parse(fields);
      for (const auto& [name, value] : values.members())
        malformed[name] = value;
      reject(with_pair(malformed, result));
    }
    for (const auto* fields : {
             R"({"call_id":"orphan"})",
             R"({"call_id":null})",
             R"({"output":null})",
             R"({"output":{"structured":true}})",
             R"({"output":[{"type":"input_image","image_url":"unsupported"}]})",
             R"({"output":[{"type":"input_text","text":7}]})",
             R"({"output":["text"]})",
             R"({"role":"tool"})",
             R"({"status":"failed"})",
         }) {
      auto malformed = result;
      const auto values = gufo::json::parse(fields);
      for (const auto& [name, value] : values.members())
        malformed[name] = value;
      reject(with_pair(call, malformed));
    }
    auto invalid = with_pair(call, call);
    reject(invalid);
    invalid = with_pair(result, call);
    reject(invalid);
    invalid = with_pair(call, result);
    invalid["input"].push_back(result);
    reject(invalid);
    invalid = with_pair(call, result);
    invalid["input"].push_back(call);
    invalid["input"].push_back(result);
    reject(invalid);
    invalid = with_pair(
        call, gufo::json::parse(R"({"role":"user","content":"No result"})"));
    reject(invalid);
    invalid = FunctionRequest();
    invalid["input"] = gufo::json::Value::array();
    invalid["input"].push_back(call);
    reject(invalid);
    auto another = call;
    another["call_id"] = "call_second";
    invalid = with_pair(call, another);
    invalid["input"].push_back(result);
    reject(invalid);
    assert(server.backend->calls == before);
    // Historical tools may no longer be declared, but IDs still must link.
    auto valid = with_pair(call, result);
    valid["stream"] = stream;
    server.backend->SetOutput("Historical result received.");
    (void)CompleteResponses(server, valid);
    const auto history = server.backend->LastCall().chat.messages;
    assert(history.size() == 3 &&
           history[1].tool_calls[0].name == "historical" &&
           history[1].tool_calls[0].arguments.empty() &&
           history[2].content.empty());
  }
}

void TestResponsesFunctionFailures() {
  RunningServer server;
  for (const bool stream : {false, true}) {
    for (const auto* raw : {
             R"(<tool_call>{"name":"clock","arguments":)</tool_call>)",
             R"(<tool_call>{"name":"clock","arguments":{)</tool_call>)",
             R"(<tool_call>{"name":"unknown","arguments":{}}</tool_call>)",
             R"(<tool_call>{"name":"clock","arguments":""}</tool_call>)",
         }) {
      auto request = FunctionRequest();
      request["stream"] = stream;
      server.backend->SetOutput(raw);
      const auto fallback = CompleteResponses(server, request);
      assert(fallback.find("output")->size() == 1 &&
             fallback.find("output")->items()[0].member_str("type") ==
                 "message" &&
             fallback.find("output")
                     ->items()[0]
                     .find("content")
                     ->items()[0]
                     .member_str("text") == raw);
      request["tool_choice"] = "required";
      const auto rejected = server.Post("/v1/responses", request.dump());
      if (stream) {
        const auto events = ResponsesEvents(rejected);
        assert(events.back().member_str("type") == "response.failed");
        const auto* failed = events.back().find("response");
        assert(failed->find("error")->member_str("code") ==
                   "tool_choice_unsatisfied" &&
               failed->find("output")->empty());
      } else {
        ExpectStatus(rejected, 502);
        assert(rejected.find("tool_choice_unsatisfied") != std::string::npos);
      }
    }
    auto request = FunctionRequest();
    request["stream"] = stream;
    request["parallel_tool_calls"] = false;
    server.backend->SetOutput(std::string(kWeatherCall) + kClockCall);
    const auto rejected = server.Post("/v1/responses", request.dump());
    if (stream) {
      const auto events = ResponsesEvents(rejected);
      assert(events.back().member_str("type") == "response.failed" &&
             events.back().find("response")->find("output")->empty());
      for (const auto& event : events)
        assert(event.member_str("type") !=
               "response.function_call_arguments.delta");
    } else {
      ExpectStatus(rejected, 502);
    }
  }
}

void TestResponsesStrictFunctions() {
  RunningServer server;
  for (const bool stream : {false, true}) {
    auto request = gufo::json::parse(R"({"input":"Call the strict function",
      "tools":[{"type":"function","name":"namespace.clock","strict":true,
        "parameters":{"type":"object","properties":{"hour":{"type":"integer"}},"required":["hour"],"additionalProperties":false}}],
      "tool_choice":{"type":"function","name":"namespace.clock"}
    })");
    request["stream"] = stream;
    server.backend->SetOutput(
        R"(<tool_call>{"name":"namespace.clock","arguments":{"hour":12}}</tool_call>)");
    const auto valid = CompleteResponses(server, request);
    const auto& call = valid.find("output")->items()[0];
    assert(call.member_str("name") == "namespace.clock" &&
           call.member_str("arguments") == R"({"hour":12})");
    assert(server.backend->LastCall().chat.tools[0].name == "namespace.clock");
    for (const auto* arguments :
         {R"({})", R"({"hour":"12"})", R"({"hour":12,"extra":true})"}) {
      // A valid first call must not escape when a later strict call is invalid.
      server.backend->SetOutput(
          std::string(
              R"(<tool_call>{"name":"namespace.clock","arguments":{"hour":12}}</tool_call><tool_call>{"name":"namespace.clock","arguments":)") +
          arguments + "}</tool_call>");
      const auto rejected = server.Post("/v1/responses", request.dump());
      if (stream) {
        const auto events = ResponsesEvents(rejected);
        assert(events.back().member_str("type") == "response.failed" &&
               events.back().find("response")->find("output")->empty());
        for (const auto& event : events)
          assert(event.member_str("type") != "response.output_item.added");
      } else {
        ExpectStatus(rejected, 502);
      }
    }
    const int before = server.backend->calls;
    request["tools"] = gufo::json::parse(R"([
      {"type":"function","name":"namespace.clock","strict":true,"parameters":{"type":"object","properties":{"hour":{"type":"integer"}}}}
    ])");
    const auto rejected = server.Post("/v1/responses", request.dump());
    ExpectStatus(rejected, 400);
    assert(rejected.find("text/event-stream") == std::string::npos &&
           server.backend->calls == before);
  }
}

void TestResponsesFormatNormalization() {
  RunningServer server;
  for (const bool stream : {false, true}) {
    for (
        const auto* format : {
            R"({"type":"text"})",
            R"({"type":"json_object"})",
            R"({"type":"json_schema","name":"answer","schema":{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false},"strict":true})",
        }) {
      auto request = gufo::json::parse(
          R"({"input":"Return JSON","include":["reasoning.encrypted_content"],"metadata":{"client":"test"}})");
      request["stream"] = stream;
      request["text"]["format"] = gufo::json::parse(format);
      server.backend->SetOutput(R"({"ok":true})");
      const auto response = CompleteResponses(server, request);
      assert(response.find("text")->find("format")->member_str("type") ==
             request.find("text")->find("format")->member_str("type"));
      assert(response.find("metadata")->member_str("client") == "test");
      assert(!response.contains("encrypted_content"));
    }
    const int before = server.backend->calls;
    for (
        const auto* fields : {
            R"({"text":null})",
            R"({"text":{"verbosity":"low"}})",
            R"({"text":{"format":null}})",
            R"({"text":{"format":{"type":"unknown"}}})",
            R"({"text":{"format":{"type":"json_object"}},"response_format":{"type":"text"}})",
            R"({"text":{"format":{"type":"json_schema","name":"f"}}})",
            R"({"include":["unsupported"]})",
            R"({"include":true})",
            R"({"metadata":{"value":5}})",
        }) {
      auto request = gufo::json::parse(fields);
      request["input"] = "JSON";
      request["stream"] = stream;
      const auto rejected = server.Post("/v1/responses", request.dump());
      ExpectStatus(rejected, 400);
      assert(rejected.find("text/event-stream") == std::string::npos);
    }
    assert(server.backend->calls == before);
    auto same = gufo::json::parse(R"({"input":"JSON",
      "text":{"format":{"type":"json_schema","name":"f","schema":{"type":"object","properties":{},"additionalProperties":false},"strict":false}},
      "response_format":{"json_schema":{"strict":false,"schema":{"properties":{},"additionalProperties":false,"type":"object"},"name":"f"},"type":"json_schema"}
    })");
    same["stream"] = stream;
    server.backend->SetOutput("{}");
    (void)CompleteResponses(server, same);
  }
}

void TestResponsesJsonFunctionRoundtrip() {
  RunningServer server;
  for (const bool first_stream : {false, true}) {
    for (const bool followup_stream : {false, true}) {
      auto request = FunctionRequest();
      request["input"] = gufo::json::parse(
          R"([{"role":"user","content":"Check the clock then return JSON"}])");
      request["reasoning"]["effort"] = "high";
      request["tool_choice"] =
          gufo::json::parse(R"({"type":"function","name":"clock"})");
      request["text"]["format"] = gufo::json::parse(
          R"({"type":"json_schema","name":"answer","strict":true,
        "schema":{"type":"object","properties":{"time":{"type":"string"}},"required":["time"],"additionalProperties":false}})");
      request["stream"] = first_stream;
      server.backend->SetOutput(std::string("Need clock</think>") + kClockCall);
      const auto first = CompleteResponses(server, request);
      const auto& items = first.find("output")->items();
      assert(items.size() == 2 && items[0].member_str("type") == "reasoning" &&
             items[1].member_str("type") == "function_call");
      for (const auto& item : items)
        request["input"].push_back(item);
      auto result = gufo::json::parse(
          R"({"type":"function_call_output","output":"12:00"})");
      result["call_id"] = items[1].member_str("call_id");
      request["input"].push_back(result);
      request["tool_choice"] = "auto";
      request["stream"] = followup_stream;
      server.backend->SetOutput(R"(Got time</think>{"time":"12:00"})");
      const auto next = CompleteResponses(server, request);
      const auto& answer = next.find("output")->items();
      assert(answer.size() == 2 &&
             answer[0].member_str("type") == "reasoning" &&
             answer[1].member_str("type") == "message" &&
             answer[1].find("content")->items()[0].member_str("text") ==
                 R"({"time":"12:00"})");
      const auto history = server.backend->LastCall().chat;
      assert(history.json_constraint && history.messages.size() == 3 &&
             history.messages[1].thought == "Need clock" &&
             history.messages[1].tool_calls[0].id ==
                 items[1].member_str("call_id") &&
             history.messages[2].tool_call_id ==
                 items[1].member_str("call_id") &&
             history.messages[2].content == "12:00");
    }
  }
}

void TestResponsesStreamFailure() {
  RunningServer server;
  for (int failure : {1, 2, 3}) {
    server.backend->failure = failure;
    const auto events = ResponsesEvents(
        server.Post("/v1/responses", R"({"input":"Hi","stream":true})"));
    assert(events.back().member_str("type") == "response.failed");
    const auto& response = *events.back().find("response");
    const auto code = failure == 3 ? "output_limit" : "generation_failed";
    assert(response.member_str("status") == "failed" &&
           response.find("error")->member_str("code") == code &&
           response.find("usage")->is_null());
    assert(events[events.size() - 2].member_str("type") == "error" &&
           events[events.size() - 2].member_str("code") == code);
    const auto& output = response.find("output")->items();
    assert(output.size() == (failure == 3 ? 1 : 0));
    if (!output.empty())
      assert(output[0].member_str("status") == "incomplete" &&
             output[0].find("content")->items()[0].member_str("text") == "ok");
  }
}

void TestResponsesStreamIsLiveAndCancellable() {
  for (bool disconnect : {false, true}) {
    RunningServer server;
    server.backend->wait_after_piece = true;
    const int fd = server.Connect();
    const std::string body = R"({"input":"Hi","stream":true})";
    const auto request = "POST /v1/responses HTTP/1.1\r\nContent-Length: " +
                         std::to_string(body.size()) + "\r\n\r\n" + body;
    assert(::send(fd, request.data(), request.size(), MSG_NOSIGNAL) ==
           static_cast<ssize_t>(request.size()));
    assert(server.backend->entered.try_acquire_for(std::chrono::seconds(2)));
    std::string response;
    char buffer[4096];
    while (response.find("event: response.output_text.delta\n") ==
           std::string::npos) {
      const auto count = ::read(fd, buffer, sizeof(buffer));
      assert(count > 0);
      response.append(buffer, static_cast<std::size_t>(count));
    }
    assert(response.find("event: response.completed\n") == std::string::npos);
    if (disconnect) {
      ::close(fd);
      assert(server.backend->finished.try_acquire_for(std::chrono::seconds(2)));
      assert(server.backend->disconnected);
    } else {
      server.backend->release_output = true;
      for (;;) {
        const auto count = ::read(fd, buffer, sizeof(buffer));
        assert(count >= 0);
        if (!count)
          break;
        response.append(buffer, static_cast<std::size_t>(count));
      }
      ::close(fd);
      const auto events = ResponsesEvents(response);
      assert(events.back().member_str("type") == "response.completed");
    }
  }
}

void TestInvalidBindSettings() {
  for (const int port : {-1, 65536}) {
    HttpServer server("127.0.0.1", port, nullptr);
    std::string error;
    assert(!server.start(&error));
    assert(!error.empty());
  }
  HttpServer invalid("bad.address", 0, nullptr);
  std::string error;
  assert(!invalid.start(&error));
  assert(!error.empty());
}

void TestCompatibilityUtf8() {
  RunningServer server;
  server.backend->SetOutput("é中😀\xE2\x94!\xF0\x9F");
  const std::string expected = "é中😀\xEF\xBF\xBD!\xEF\xBF\xBD";
  for (const auto& [path, input] : {
           std::pair{"/v1/completions", R"({"prompt":"hi"})"},
           std::pair{"/v1/responses", R"({"input":"hi"})"},
           std::pair{"/v1/messages",
                     R"({"messages":[{"role":"user","content":"hi"}]})"},
           std::pair{"/completion", R"({"prompt":"hi"})"},
       }) {
    const auto response = server.Post(path, input);
    ExpectStatus(response, 200);
    const auto output =
        gufo::json::parse(response.substr(response.find("\r\n\r\n") + 4));
    std::string text;
    if (std::string_view(path) == "/v1/completions")
      text = output.find("choices")->items()[0].member_str("text");
    else if (std::string_view(path) == "/v1/responses")
      text = output.find("output")
                 ->items()[0]
                 .find("content")
                 ->items()[0]
                 .member_str("text");
    else if (std::string_view(path) == "/v1/messages")
      text = output.find("content")->items()[0].member_str("text");
    else
      text = output.member_str("content");
    assert(text == expected);
  }
}

void TestQueryParameters() {
  gufo::server::HttpRequest request;
  request.query = "notafter=wrong&note=after=wrong&after=right+value%26x";
  assert(request.query_param("after") == "right value&x");
  request.query = "notafter=wrong&note=after=wrong";
  assert(request.query_param("after").empty());
  request.query = "%61fter=encoded&broken=%xz&empty";
  assert(request.query_param("after") == "encoded");
  assert(request.query_param("broken") == "%xz");
  assert(request.query_param("empty").empty());
}

void TestPeerDisconnect() {
  RunningServer server;
  server.backend->wait_for_disconnect = true;
  const int fd = server.Connect();
  const std::string body = R"({"prompt":"hi","max_tokens":128})";
  const std::string request =
      "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n"
      "Content-Type: application/json\r\nContent-Length: " +
      std::to_string(body.size()) + "\r\n\r\n" + body;
  assert(::send(fd, request.data(), request.size(), MSG_NOSIGNAL) ==
         static_cast<ssize_t>(request.size()));
  assert(server.backend->entered.try_acquire_for(std::chrono::seconds(2)));
  ::close(fd);
  assert(server.backend->finished.try_acquire_for(std::chrono::seconds(2)));
  assert(server.backend->disconnected);
}

void TestStreamingFraming() {
  RunningServer server;
  const std::string chunks = std::string("3\r\na\0b\r\n", 8) + "3\r\nend\r\n";
  for (const bool fail : {false, true}) {
    const auto response = server.Post("/stream", fail ? "fail" : "");
    ExpectStatus(response, 200);
    assert(response.find("Transfer-Encoding: chunked\r\n") !=
           std::string::npos);
    assert(response.substr(response.find("\r\n\r\n") + 4) ==
           chunks + (fail ? "" : "0\r\n\r\n"));
  }
  const auto thrown = server.Post("/stream-error", "");
  assert(thrown.substr(thrown.find("\r\n\r\n") + 4) == "b\r\nfirst chunk\r\n");
  // Existing HTTP/1.0 clients retain close-delimited framing.
  const auto legacy = server.Send("POST /stream HTTP/1.0\r\n\r\n");
  assert(legacy.find("Transfer-Encoding:") == std::string::npos);
  assert(legacy.substr(legacy.find("\r\n\r\n") + 4) ==
         std::string("a\0bend", 6));
}

}  // namespace

int main() {
  TestRequestLogging();
  TestInvalidBindSettings();
  TestQueryParameters();
  TestAuthorization();
  TestFramingAndMetrics();
  TestCompatibilityRequests();
  TestCompatibilityUtf8();
  TestResponsesStreaming();
  TestResponsesReasoningHistoryRoundtrip();
  TestResponsesReasoningOnlyHistoryRoundtrip();
  TestResponsesReasoningHistoryValidation();
  TestResponsesReasoningAndValidation();
  TestResponsesFunctionOutput();
  TestResponsesFunctionChoices();
  TestResponsesFunctionRounds();
  TestResponsesFunctionValidation();
  TestResponsesFunctionFailures();
  TestResponsesStrictFunctions();
  TestResponsesFormatNormalization();
  TestResponsesJsonFunctionRoundtrip();
  TestResponsesStreamFailure();
  TestResponsesStreamIsLiveAndCancellable();
  TestPeerDisconnect();
  TestStreamingFraming();
  std::cout << "HTTP transport checks passed.\n";
}
