#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_ROCM_KV_QUANT_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_ROCM_KV_QUANT_HPP_

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace gufo::models::qwen38_flash_next::rocm::kv {

/// Packed q8_0 K/V rows: blocks of 32 elements, each block one f16 scale
/// followed by 32 int8 quanta (34 bytes per 32 elements — 53% of the f16
/// row). Quantized decode halves the bytes the attention read touches,
/// which on a bandwidth-walled GPU is the one kernel-side change that
/// moves the wall, and doubles the reach of the elastic history budget.
/// Per-block scales (not per-tensor) keep the quantization near-lossless;
/// direct per-tensor INT8 KV measurably degrades outputs.
constexpr std::uint32_t kBlock = 32;
constexpr std::uint32_t kBlockBytes = kBlock + 2;  // int8 quanta + f16 scale

/// Packed bytes of one cache row of `kv_width` elements.
[[nodiscard]] inline std::size_t RowBytes(std::size_t kv_width) noexcept {
  return (kv_width / kBlock) * kBlockBytes;
}

/// Reference quantize/dequant (host and device): the device kernels
/// implement the same arithmetic. `amax == 0` rows store scale 0 and
/// all-zero quanta.
inline void QuantizeRow(const float* src, std::byte* dst,
                        std::size_t count) noexcept {
  std::size_t block = 0;
  for (std::size_t offset = 0; offset < count; offset += kBlock, ++block) {
    const std::size_t span =
        count - offset < kBlock ? count - offset : kBlock;
    float amax = 0.0f;
    for (std::size_t i = 0; i < span; ++i) {
      const float magnitude =
          src[offset + i] < 0.0f ? -src[offset + i] : src[offset + i];
      amax = amax > magnitude ? amax : magnitude;
    }
    const float d = amax / 127.0f;
    const auto scale =
        std::bit_cast<std::uint16_t, _Float16>(d == 0.0f ? _Float16(0.0f)
                                                         : _Float16(d));
    dst[block * kBlockBytes] = static_cast<std::byte>(scale & 0xff);
    dst[block * kBlockBytes + 1] = static_cast<std::byte>(scale >> 8);
    for (std::size_t i = 0; i < span; ++i) {
      const float scaled = src[offset + i] / d;
      const int q = static_cast<int>(std::lrintf(scaled));
      const int clamped = q < -128 ? -128 : (q > 127 ? 127 : q);
      dst[block * kBlockBytes + 2 + i] = static_cast<std::byte>(clamped);
    }
  }
}

inline void DequantRow(const std::byte* src, float* dst,
                       std::size_t count) noexcept {
  std::size_t block = 0;
  for (std::size_t offset = 0; offset < count; offset += kBlock, ++block) {
    const std::size_t span =
        count - offset < kBlock ? count - offset : kBlock;
    const auto scale = std::bit_cast<_Float16, std::uint16_t>(
        static_cast<std::uint16_t>(
            static_cast<unsigned>(src[block * kBlockBytes]) |
            (static_cast<unsigned>(src[block * kBlockBytes + 1]) << 8)));
    const float d = static_cast<float>(scale);
    for (std::size_t i = 0; i < span; ++i) {
      dst[offset + i] =
          static_cast<float>(
              static_cast<signed char>(src[block * kBlockBytes + 2 + i])) *
          d;
    }
  }
}

}  // namespace gufo::models::qwen38_flash_next::rocm::kv

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_ROCM_KV_QUANT_HPP_
