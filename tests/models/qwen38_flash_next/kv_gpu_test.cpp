#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kv_quant.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace kv = gufo::models::qwen38_flash_next::rocm::kv;

namespace {

constexpr std::uint32_t kHeads = 24;
constexpr std::uint32_t kKvHeads = 2;
constexpr std::uint32_t kDim = 256;
constexpr std::uint32_t kKvWidth = kKvHeads * kDim;
constexpr std::uint32_t kRatio = 4;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

template<typename T>
class HipBuffer {
public:
  explicit HipBuffer(std::size_t count) : count_(count) {
    void* allocation = nullptr;
    CheckHip(hipMalloc(&allocation, bytes()), "hipMalloc");
    CheckHip(hipMemset(allocation, 0, bytes()), "hipMemset");
    data_ = static_cast<T*>(allocation);
  }
  ~HipBuffer() {
    if (data_ != nullptr) (void)hipFree(data_);
  }
  HipBuffer(const HipBuffer&) = delete;
  HipBuffer& operator=(const HipBuffer&) = delete;
  [[nodiscard]] T* get() noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return count_ * sizeof(T); }

private:
  T* data_;
  std::size_t count_;
};

template<typename T>
void Upload(HipBuffer<T>* destination, const std::vector<T>& source) {
  CheckHip(hipMemcpy(destination->get(), source.data(), source.size() * sizeof(T),
                     hipMemcpyHostToDevice),
           "upload");
}

std::vector<float> Download(HipBuffer<float>* source, std::size_t count) {
  std::vector<float> data(count);
  CheckHip(hipMemcpy(data.data(), source->get(), count * sizeof(float),
                     hipMemcpyDeviceToHost),
           "download");
  return data;
}

/// Fills a device cache with deterministic f16 rows and returns the same
/// values as floats (the read-side reference).
std::vector<float> FillF16(HipBuffer<__half>* cache, std::uint32_t positions,
                           std::mt19937_64& rng) {
  std::uniform_real_distribution<float> value{-8.0f, 8.0f};
  std::vector<float> reference(std::size_t{positions} * kKvWidth);
  std::vector<__half> halves(reference.size());
  for (std::size_t i = 0; i < reference.size(); ++i) {
    reference[i] = value(rng);
    halves[i] = __float2half(reference[i]);
  }
  Upload(cache, halves);
  return reference;
}

std::vector<float> ReferenceAttention(const std::vector<float>& q,
                                      const std::vector<float>& k,
                                      const std::vector<float>& v,
                                      std::uint32_t positions,
                                      std::uint32_t heads) {
  // Single query row (t = 0), softmax(qK^T / sqrt(d)) V per head.
  const float scale = 1.0f / std::sqrt(static_cast<float>(kDim));
  std::vector<float> out(std::size_t{heads} * kDim);
  for (std::uint32_t h = 0; h < heads; ++h) {
    const std::uint32_t kvh = h / (heads / kKvHeads);
    std::vector<float> scores(positions);
    float max_score = -1e30f;
    for (std::uint32_t p = 0; p < positions; ++p) {
      float dot = 0.0f;
      for (std::uint32_t i = 0; i < kDim; ++i) {
        dot += q[std::size_t{h} * kDim + i] *
               k[std::size_t{p} * kKvWidth + std::size_t{kvh} * kDim + i];
      }
      scores[p] = dot * scale;
      max_score = std::fmaxf(max_score, scores[p]);
    }
    float total = 0.0f;
    for (auto& s : scores) {
      s = std::exp(s - max_score);
      total += s;
    }
    for (std::uint32_t i = 0; i < kDim; ++i) {
      float acc = 0.0f;
      for (std::uint32_t p = 0; p < positions; ++p) {
        acc += scores[p] * v[std::size_t{p} * kKvWidth +
                             std::size_t{kvh} * kDim + i];
      }
      out[std::size_t{h} * kDim + i] = acc / total;
    }
  }
  return out;
}

void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
  try {
    std::mt19937_64 rng{4242};
    const std::uint32_t positions = 32;

    // ---- 1. QuantizeKv round trip on device ----
    {
      const std::size_t rows = positions;
      HipBuffer<float> src(rows * kKvWidth);
      std::uniform_real_distribution<float> value{-8.0f, 8.0f};
      std::vector<float> floats(rows * kKvWidth);
      for (auto& x : floats) x = value(rng);
      Upload(&src, floats);

      HipBuffer<std::byte> store(rows * kv::RowBytes(kKvWidth));
      std::vector<std::uint32_t> pos{0};
      HipBuffer<std::uint32_t> d_pos(1);
      Upload(&d_pos, pos);
      q::QuantizeKv(src.get(), store.get(), rows, kKvWidth, d_pos.get(),
                    nullptr);
      CheckHip(hipDeviceSynchronize(), "quantize sync");

      std::vector<std::byte> packed(kv::RowBytes(kKvWidth));
      CheckHip(hipMemcpy(packed.data(), store.get(), packed.size(),
                         hipMemcpyDeviceToHost),
               "download store");
      std::vector<float> restored(kKvWidth);
      kv::DequantRow(packed.data(), restored.data(), kKvWidth);
      float amax = 0.0f;
      for (std::size_t i = 0; i < kKvWidth; ++i) {
        amax = std::fmaxf(amax, std::fabsf(floats[i]));
      }
      const float bound = amax / 127.0f * 0.6f;
      for (std::size_t i = 0; i < kKvWidth; ++i) {
        Require(std::fabsf(floats[i] - restored[i]) <= bound,
                "device quantize round trip exceeded the error bound at " +
                    std::to_string(i));
      }
      std::cout << "quantize round trip ok (amax " << amax << ")\n";
    }

    // ---- 2. Dense attention: q8 vs f16 ----
    {
      HipBuffer<__half> k_cache(std::size_t{positions} * kKvWidth);
      HipBuffer<__half> v_cache(std::size_t{positions} * kKvWidth);
      const auto k_ref = FillF16(&k_cache, positions, rng);
      const auto v_ref = FillF16(&v_cache, positions, rng);

      std::uniform_real_distribution<float> value{-1.0f, 1.0f};
      std::vector<float> q_host(kHeads * kDim);
      for (auto& x : q_host) x = value(rng);
      HipBuffer<float> d_q(q_host.size());
      Upload(&d_q, q_host);

      // One query at absolute position 31 attends over all 32 cached rows.
      std::vector<std::uint32_t> pos{positions - 1};
      HipBuffer<std::uint32_t> d_pos(1);
      Upload(&d_pos, pos);

      HipBuffer<float> out_f16(q_host.size());
      q::Attention(d_q.get(), k_cache.get(), v_cache.get(), false, nullptr, 0,
                   out_f16.get(), nullptr, 1, 1, d_pos.get(), kHeads, kKvHeads,
                   kDim, kRatio, nullptr);
      CheckHip(hipDeviceSynchronize(), "f16 attention sync");

      // Quantize the same cache contents into rows 0..31 (own position
      // buffer; the attention query position is unrelated).
      HipBuffer<float> k_floats(std::size_t{positions} * kKvWidth);
      HipBuffer<float> v_floats(std::size_t{positions} * kKvWidth);
      Upload(&k_floats, k_ref);
      Upload(&v_floats, v_ref);
      HipBuffer<std::byte> k_store(std::size_t{positions} *
                                   kv::RowBytes(kKvWidth));
      HipBuffer<std::byte> v_store(std::size_t{positions} *
                                   kv::RowBytes(kKvWidth));
      std::vector<std::uint32_t> zero{0};
      HipBuffer<std::uint32_t> d_zero(1);
      Upload(&d_zero, zero);
      q::QuantizeKv(k_floats.get(), k_store.get(), positions, kKvWidth,
                    d_zero.get(), nullptr);
      q::QuantizeKv(v_floats.get(), v_store.get(), positions, kKvWidth,
                    d_zero.get(), nullptr);
      CheckHip(hipDeviceSynchronize(), "quantize sync");

      HipBuffer<float> out_q8(q_host.size());
      q::Attention(d_q.get(), k_store.get(), v_store.get(), true, nullptr, 0,
                   out_q8.get(), nullptr, 1, 1, d_pos.get(), kHeads, kKvHeads,
                   kDim, kRatio, nullptr);
      CheckHip(hipDeviceSynchronize(), "q8 attention sync");

      const auto ref = ReferenceAttention(q_host, k_ref, v_ref, positions,
                                          kHeads);
      const auto a = Download(&out_f16, q_host.size());
      const auto b = Download(&out_q8, q_host.size());
      float worst_f16 = 0.0f;
      float worst_q8 = 0.0f;
      float worst_mutual = 0.0f;
      for (std::size_t i = 0; i < a.size(); ++i) {
        Require(std::isfinite(a[i]), "f16 attention produced non-finite");
        Require(std::isfinite(b[i]), "q8 attention produced non-finite");
        worst_f16 = std::fmaxf(worst_f16, std::fabsf(a[i] - ref[i]));
        worst_q8 = std::fmaxf(worst_q8, std::fabsf(b[i] - ref[i]));
        worst_mutual = std::fmaxf(worst_mutual, std::fabsf(a[i] - b[i]));
      }
      std::cout << "dense attention worst |q8-ref|=" << worst_q8
                << " |f16-ref|=" << worst_f16
                << " |q8-f16|=" << worst_mutual << "\n";
      Require(worst_q8 < 0.5f, "q8 dense attention diverged from reference");
      Require(worst_q8 < worst_f16 + 0.25f, "q8 far worse than f16 noise");
    }

    // ---- 3. WMMA attention: q8 vs f16 (prefill-shaped, dense window) ----
    {
      const std::uint32_t n_tokens = 16;
      HipBuffer<__half> k_cache(std::size_t{positions} * kKvWidth);
      HipBuffer<__half> v_cache(std::size_t{positions} * kKvWidth);
      const auto k_ref = FillF16(&k_cache, positions, rng);
      const auto v_ref = FillF16(&v_cache, positions, rng);

      std::uniform_real_distribution<float> value{-1.0f, 1.0f};
      std::vector<float> q_host(std::size_t{n_tokens} * kHeads * kDim);
      std::vector<float> gate_host(std::size_t{n_tokens} * kHeads * kDim);
      for (auto& x : q_host) x = value(rng);
      for (auto& x : gate_host) x = 1.0f;  // neutral sigmoid gate input
      HipBuffer<float> d_q(q_host.size());
      HipBuffer<float> d_gate(gate_host.size());
      Upload(&d_q, q_host);
      Upload(&d_gate, gate_host);

      HipBuffer<float> out_f16(q_host.size());
      Require(q::WmmaCausalAttention(d_q.get(), d_gate.get(), k_cache.get(),
                                     v_cache.get(), false, nullptr, 0,
                                     out_f16.get(), n_tokens, 0, kHeads,
                                     kKvHeads, kDim, kRatio, nullptr),
              "WMMA f16 rejected the geometry");
      CheckHip(hipDeviceSynchronize(), "wmma f16 sync");

      HipBuffer<float> k_floats(std::size_t{positions} * kKvWidth);
      HipBuffer<float> v_floats(std::size_t{positions} * kKvWidth);
      Upload(&k_floats, k_ref);
      Upload(&v_floats, v_ref);
      HipBuffer<std::byte> k_store(std::size_t{positions} *
                                   kv::RowBytes(kKvWidth));
      HipBuffer<std::byte> v_store(std::size_t{positions} *
                                   kv::RowBytes(kKvWidth));
      std::vector<std::uint32_t> pos{0};
      HipBuffer<std::uint32_t> d_pos(1);
      Upload(&d_pos, pos);
      q::QuantizeKv(k_floats.get(), k_store.get(), positions, kKvWidth,
                    d_pos.get(), nullptr);
      q::QuantizeKv(v_floats.get(), v_store.get(), positions, kKvWidth,
                    d_pos.get(), nullptr);
      CheckHip(hipDeviceSynchronize(), "quantize sync");

      HipBuffer<float> out_q8(q_host.size());
      Require(q::WmmaCausalAttention(d_q.get(), d_gate.get(), k_store.get(),
                                     v_store.get(), true, nullptr, 0,
                                     out_q8.get(), n_tokens, 0, kHeads,
                                     kKvHeads, kDim, kRatio, nullptr),
              "WMMA q8 rejected the geometry");
      CheckHip(hipDeviceSynchronize(), "wmma q8 sync");

      const auto a = Download(&out_f16, q_host.size());
      const auto b = Download(&out_q8, q_host.size());
      float worst = 0.0f;
      std::size_t nonfinite = 0;
      for (std::size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(b[i])) ++nonfinite;
        worst = std::fmaxf(worst, std::fabsf(a[i] - b[i]));
      }
      std::cout << "wmma attention worst |q8-f16|=" << worst
                << " nonfinite=" << nonfinite << "\n";
      Require(nonfinite == 0, "q8 WMMA attention produced non-finite values");
      Require(worst < 0.5f, "q8 WMMA attention diverged from f16");
    }

    // ---- 4. Sparse masked WMMA at depth: q8 vs f16 ----
    {
      const std::uint32_t n_tokens = 16;
      const std::uint32_t start_pos = 4090;
      const std::uint32_t n_kv = start_pos + n_tokens;
      const std::uint32_t max_blocks = (n_kv + kRatio - 1) / kRatio;
      const std::uint32_t mask_words = (max_blocks + 31) / 32;
      HipBuffer<__half> k_cache(std::size_t{n_kv} * kKvWidth);
      HipBuffer<__half> v_cache(std::size_t{n_kv} * kKvWidth);
      const auto k_ref = FillF16(&k_cache, n_kv, rng);
      const auto v_ref = FillF16(&v_cache, n_kv, rng);

      // Causal-ish sparse mask: each query selects ~1/16 of eligible blocks
      // plus its own completed blocks.
      std::vector<std::uint32_t> mask(std::size_t{n_tokens} * mask_words);
      std::mt19937_64 mask_rng{777};
      for (std::uint32_t t = 0; t < n_tokens; ++t) {
        const std::uint32_t complete = (start_pos + t + 1) / kRatio;
        auto* row = mask.data() + std::size_t{t} * mask_words;
        for (std::uint32_t b = 0; b < complete; ++b) {
          if (mask_rng() % 16 == 0 || b == complete - 1) {
            row[b / 32] |= 1u << (b % 32);
          }
        }
      }
      HipBuffer<std::uint32_t> d_mask(mask.size());
      Upload(&d_mask, mask);

      std::uniform_real_distribution<float> value{-1.0f, 1.0f};
      std::vector<float> q_host(std::size_t{n_tokens} * kHeads * kDim);
      std::vector<float> gate_host(std::size_t{n_tokens} * kHeads * kDim);
      for (auto& x : q_host) x = value(rng);
      for (auto& x : gate_host) x = 1.0f;
      HipBuffer<float> d_q(q_host.size());
      HipBuffer<float> d_gate(gate_host.size());
      Upload(&d_q, q_host);
      Upload(&d_gate, gate_host);

      HipBuffer<float> out_f16(q_host.size());
      Require(q::WmmaCausalAttention(d_q.get(), d_gate.get(), k_cache.get(),
                                     v_cache.get(), false, d_mask.get(),
                                     mask_words, out_f16.get(), n_tokens,
                                     start_pos, kHeads, kKvHeads, kDim, kRatio,
                                     nullptr),
              "sparse WMMA f16 rejected the geometry");
      CheckHip(hipDeviceSynchronize(), "sparse wmma f16 sync");

      HipBuffer<float> k_floats(std::size_t{n_kv} * kKvWidth);
      HipBuffer<float> v_floats(std::size_t{n_kv} * kKvWidth);
      Upload(&k_floats, k_ref);
      Upload(&v_floats, v_ref);
      HipBuffer<std::byte> k_store(std::size_t{n_kv} * kv::RowBytes(kKvWidth));
      HipBuffer<std::byte> v_store(std::size_t{n_kv} * kv::RowBytes(kKvWidth));
      std::vector<std::uint32_t> zero{0};
      HipBuffer<std::uint32_t> d_zero(1);
      Upload(&d_zero, zero);
      q::QuantizeKv(k_floats.get(), k_store.get(), n_kv, kKvWidth,
                    d_zero.get(), nullptr);
      q::QuantizeKv(v_floats.get(), v_store.get(), n_kv, kKvWidth,
                    d_zero.get(), nullptr);
      CheckHip(hipDeviceSynchronize(), "sparse quantize sync");

      HipBuffer<float> out_q8(q_host.size());
      Require(q::WmmaCausalAttention(d_q.get(), d_gate.get(), k_store.get(),
                                     v_store.get(), true, d_mask.get(),
                                     mask_words, out_q8.get(), n_tokens,
                                     start_pos, kHeads, kKvHeads, kDim, kRatio,
                                     nullptr),
              "sparse WMMA q8 rejected the geometry");
      CheckHip(hipDeviceSynchronize(), "sparse wmma q8 sync");

      const auto a = Download(&out_f16, q_host.size());
      const auto b = Download(&out_q8, q_host.size());
      float worst = 0.0f;
      std::size_t nonfinite = 0;
      for (std::size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(b[i])) ++nonfinite;
        worst = std::fmaxf(worst, std::fabsf(a[i] - b[i]));
      }
      std::cout << "sparse wmma worst |q8-f16|=" << worst
                << " nonfinite=" << nonfinite << "\n";
      Require(nonfinite == 0,
              "sparse q8 WMMA attention produced non-finite values");
      Require(worst < 0.5f, "sparse q8 WMMA diverged from f16");
    }

    // ---- 5. Prefill-shaped WMMA chunk (2048 queries) at depth: q8 vs f16 ----
    for (const std::uint32_t start_pos : {0u, 65536u + 4096u}) {
      const std::uint32_t n_tokens = 2048;
      const std::uint32_t n_kv = start_pos + n_tokens;
      const std::uint32_t max_blocks = (n_kv + kRatio - 1) / kRatio;
      const std::uint32_t mask_words = (max_blocks + 31) / 32;
      HipBuffer<__half> k_cache(std::size_t{n_kv} * kKvWidth);
      HipBuffer<__half> v_cache(std::size_t{n_kv} * kKvWidth);
      const auto k_ref = FillF16(&k_cache, n_kv, rng);
      const auto v_ref = FillF16(&v_cache, n_kv, rng);

      // Sparse selection: each 2048-query chunk selects a block subset per
      // query group, exactly like SelectBlocks output at depth.
      std::vector<std::uint32_t> mask(std::size_t{n_tokens} * mask_words);
      std::mt19937_64 mask_rng{start_pos + 13};
      for (std::uint32_t t = 0; t < n_tokens; ++t) {
        const std::uint32_t complete = (start_pos + t + 1) / kRatio;
        auto* row = mask.data() + std::size_t{t} * mask_words;
        for (std::uint32_t b = 0; b < complete; ++b) {
          if (mask_rng() % 16 == 0 || b == complete - 1) {
            row[b / 32] |= 1u << (b % 32);
          }
        }
      }
      HipBuffer<std::uint32_t> d_mask(mask.size());
      Upload(&d_mask, mask);

      std::uniform_real_distribution<float> value{-1.0f, 1.0f};
      std::vector<float> q_host(std::size_t{n_tokens} * kHeads * kDim);
      std::vector<float> gate_host(std::size_t{n_tokens} * kHeads * kDim);
      for (auto& x : q_host) x = value(rng);
      for (auto& x : gate_host) x = 1.0f;
      HipBuffer<float> d_q(q_host.size());
      HipBuffer<float> d_gate(gate_host.size());
      Upload(&d_q, q_host);
      Upload(&d_gate, gate_host);

      HipBuffer<float> out_f16(q_host.size());
      Require(q::WmmaCausalAttention(d_q.get(), d_gate.get(), k_cache.get(),
                                     v_cache.get(), false, d_mask.get(),
                                     mask_words, out_f16.get(), n_tokens,
                                     start_pos, kHeads, kKvHeads, kDim, kRatio,
                                     nullptr),
              "prefill WMMA f16 rejected the geometry");
      CheckHip(hipDeviceSynchronize(), "prefill wmma f16 sync");

      HipBuffer<float> k_floats(std::size_t{n_kv} * kKvWidth);
      HipBuffer<float> v_floats(std::size_t{n_kv} * kKvWidth);
      Upload(&k_floats, k_ref);
      Upload(&v_floats, v_ref);
      HipBuffer<std::byte> k_store(std::size_t{n_kv} * kv::RowBytes(kKvWidth));
      HipBuffer<std::byte> v_store(std::size_t{n_kv} * kv::RowBytes(kKvWidth));
      std::vector<std::uint32_t> zero{0};
      HipBuffer<std::uint32_t> d_zero(1);
      Upload(&d_zero, zero);
      q::QuantizeKv(k_floats.get(), k_store.get(), n_kv, kKvWidth,
                    d_zero.get(), nullptr);
      q::QuantizeKv(v_floats.get(), v_store.get(), n_kv, kKvWidth,
                    d_zero.get(), nullptr);
      CheckHip(hipDeviceSynchronize(), "prefill quantize sync");

      HipBuffer<float> out_q8(q_host.size());
      Require(q::WmmaCausalAttention(d_q.get(), d_gate.get(), k_store.get(),
                                     v_store.get(), true, d_mask.get(),
                                     mask_words, out_q8.get(), n_tokens,
                                     start_pos, kHeads, kKvHeads, kDim, kRatio,
                                     nullptr),
              "prefill WMMA q8 rejected the geometry");
      CheckHip(hipDeviceSynchronize(), "prefill wmma q8 sync");

      const auto a = Download(&out_f16, q_host.size());
      const auto b = Download(&out_q8, q_host.size());
      float worst = 0.0f;
      std::size_t nonfinite = 0;
      for (std::size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(b[i])) ++nonfinite;
        worst = std::fmaxf(worst, std::fabsf(a[i] - b[i]));
      }
      std::cout << "prefill chunk (start " << start_pos
                << ") worst |q8-f16|=" << worst
                << " nonfinite=" << nonfinite << "\n";
      Require(nonfinite == 0,
              "prefill-shape q8 WMMA produced non-finite values");
      Require(worst < 0.5f, "prefill-shape q8 WMMA diverged from f16");
    }

    std::cout << "kv_gpu tests passed\n";
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return 1;
  }
}
