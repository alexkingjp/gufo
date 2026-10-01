#ifndef GUFO_CORE_HIP_WAIT_POLICY_HPP_
#define GUFO_CORE_HIP_WAIT_POLICY_HPP_

#include <hip/hip_runtime.h>

#include <chrono>
#include <stdexcept>
#include <string>
#include <thread>

namespace gufo::hip {

// Busy-waiting synchronization (the HIP default for hipStreamSynchronize and
// hipEventSynchronize) pins one core at 100% for the entire GPU wait: seconds
// per prefill chunk, for the scheduler thread and for every snapshot-capture
// worker. Poll instead: spin only long enough to keep short waits free of a
// scheduler round trip, then sleep between polls so admission, detokenization,
// snapshot capture, and the n-gram readers keep their CPU time. A poll costs
// a few hundred nanoseconds; the sleep bounds the added latency.
constexpr std::chrono::microseconds kSpinBudget{1000};
constexpr std::chrono::microseconds kPollSleep{50};

template <typename Poll, typename OnError>
inline bool WaitReady(Poll&& poll, OnError&& on_error) {
  const auto start = std::chrono::steady_clock::now();
  for (;;) {
    const hipError_t status = poll();
    if (status == hipSuccess) {
      return true;
    }
    if (status != hipErrorNotReady) {
      return on_error(status);
    }
    if (std::chrono::steady_clock::now() - start < kSpinBudget) {
      continue;
    }
    std::this_thread::sleep_for(kPollSleep);
  }
}

/// Reports failures through `error_msg` the way the executor's Check() does.
inline bool WaitEvent(hipEvent_t event, const char* what,
                      std::string* error_msg) {
  return WaitReady(
      [&] { return hipEventQuery(event); }, [&](hipError_t status) {
        if (error_msg != nullptr) {
          *error_msg = std::string(what) + ": " + hipGetErrorString(status);
        }
        return false;
      });
}

inline bool WaitStream(hipStream_t stream, const char* what,
                       std::string* error_msg) {
  return WaitReady(
      [&] { return hipStreamQuery(stream); }, [&](hipError_t status) {
        if (error_msg != nullptr) {
          *error_msg = std::string(what) + ": " + hipGetErrorString(status);
        }
        return false;
      });
}

/// Throwing variant for callers that signal failures by exception.
template <typename Poll>
inline void WaitReadyThrow(Poll&& poll) {
  WaitReady(std::forward<Poll>(poll), [](hipError_t status) {
    throw std::runtime_error(hipGetErrorString(status));
    return false;
  });
}

inline void WaitStreamThrow(hipStream_t stream) {
  WaitReadyThrow([&] { return hipStreamQuery(stream); });
}

}  // namespace gufo::hip

#endif  // GUFO_CORE_HIP_WAIT_POLICY_HPP_
