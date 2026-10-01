#include "src/models/qwen38_flash_next/kernels/rocm/kv_quant.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace kv = gufo::models::qwen38_flash_next::rocm::kv;

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

void CheckRowBytes() {
  Require(kv::RowBytes(512) == 16 * 34, "512-wide row is 16 blocks");
  Require(kv::RowBytes(0) == 0, "empty row is free");
  // 53% of the f16 row plus change.
  Require(kv::RowBytes(1024) * 2 < 1024 * 2 * 2,
          "q8 rows must be smaller than f16 rows");
  // 34/32 bytes per element vs 2: a ~15.6% overhead over raw int8.
  Require(kv::RowBytes(1024) == 32 * 34, "1024-wide row is 32 blocks");
}

void CheckRoundTripErrorBound() {
  std::mt19937_64 rng{20260927};
  std::uniform_real_distribution<float> magnitude{1e-4f, 64.0f};
  for (int trial = 0; trial < 2000; ++trial) {
    const std::size_t width = 512;
    std::vector<float> row(width);
    const float scale = magnitude(rng);
    std::uniform_real_distribution<float> value{-scale, scale};
    for (auto& x : row) x = value(rng);

    std::vector<std::byte> packed(kv::RowBytes(width));
    std::vector<float> restored(width);
    kv::QuantizeRow(row.data(), packed.data(), width);
    kv::DequantRow(packed.data(), restored.data(), width);

    float amax = 0.0f;
    for (const auto x : row) {
      amax = std::fmaxf(amax, std::fabs(x));
    }
    // Per-block quantization bounds the absolute error at half a quantum
    // step plus the f16 scale's own rounding.
    const float bound = amax / 127.0f * 0.501f + amax * 5e-4f;
    for (std::size_t i = 0; i < width; ++i) {
      const float error = std::fabs(row[i] - restored[i]);
      if (!(error <= bound)) {
        std::cerr << "error " << error << " exceeds bound " << bound
                  << " at trial " << trial << '\n';
        throw std::runtime_error("round-trip error bound violated");
      }
    }
  }
}

void CheckEdgeCases() {
  // Zero rows quantize to scale 0 and restore as exact zeros.
  std::vector<float> row(256, 0.0f);
  std::vector<std::byte> packed(kv::RowBytes(256));
  std::vector<float> restored(256);
  kv::QuantizeRow(row.data(), packed.data(), row.size());
  kv::DequantRow(packed.data(), restored.data(), restored.size());
  for (const auto x : restored) {
    Require(x == 0.0f, "zero rows must restore to zero");
  }

  // Extreme magnitudes clamp into int8 without saturation errors.
  row[0] = 1e30f;
  row[33] = -1e30f;
  kv::QuantizeRow(row.data(), packed.data(), row.size());
  kv::DequantRow(packed.data(), restored.data(), restored.size());
  Require(restored[0] > 1e29f && restored[33] < -1e29f,
          "dominant magnitudes survive quantization");

  // A 100-point ramp has spacing amax/50 > the quantum amax/127, so
  // quantization cannot collapse neighbors; the tail stays at zero.
  for (std::size_t i = 0; i < row.size(); ++i) {
    row[i] = i < 100 ? static_cast<float>(static_cast<int>(i) - 50) * 2.56f
                     : 0.0f;
  }
  kv::QuantizeRow(row.data(), packed.data(), row.size());
  kv::DequantRow(packed.data(), restored.data(), restored.size());
  int signs = 0;
  for (std::size_t i = 1; i < row.size(); ++i) {
    if ((restored[i] > restored[i - 1]) != (row[i] > row[i - 1])) ++signs;
  }
  Require(signs == 0, "monotone rows must stay monotone after quantization");
}

void CheckSymmetry() {
  // Quantization is scale-invariant: scaling a row scales the restoration.
  std::vector<float> row(128);
  for (std::size_t i = 0; i < row.size(); ++i) {
    row[i] = std::sin(static_cast<float>(i) * 0.7f) * 3.0f;
  }
  std::vector<std::byte> packed(kv::RowBytes(128));
  std::vector<float> a(128), b(128);
  kv::QuantizeRow(row.data(), packed.data(), row.size());
  kv::DequantRow(packed.data(), a.data(), a.size());
  for (auto& x : row) x *= 16.0f;
  kv::QuantizeRow(row.data(), packed.data(), row.size());
  kv::DequantRow(packed.data(), b.data(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    const float expected = a[i] * 16.0f;
    const float tolerance = std::fabs(expected) * 0.02f + 1e-3f;
    Require(std::fabs(b[i] - expected) <= tolerance,
            "quantization must be scale-invariant");
  }
}

int main() {
  try {
    CheckRowBytes();
    CheckRoundTripErrorBound();
    CheckEdgeCases();
    CheckSymmetry();
  } catch (const std::exception& exception) {
    std::cerr << "FAILED: " << exception.what() << '\n';
    return 1;
  }
  std::cout << "kv_quant tests passed\n";
  return 0;
}
