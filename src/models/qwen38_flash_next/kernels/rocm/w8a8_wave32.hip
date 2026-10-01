/// Wave32 variant of the one prefill-shape GEMM that otherwise runs the
/// wave64 four-row-group schedule (see w8a8_wave64.hip). Compiled with
/// -mwavefrontsize32 so warpSize() matches the W32 WMMA builtins; the
/// template requires WM*WN == 8 at wave32. Selected at runtime by
/// GUFO_WAVE32_MOE=1 for A/B benchmarking only — wave64 stays the default.
#include "src/models/qwen/hip/kernels/prefill_quant_gemm.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

void W8A8GemmWave32(const void* w, const void* x_tiled, float* out,
                    std::size_t batch, std::size_t m, std::size_t k,
                    hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>((batch + 127) / 128),
                  static_cast<unsigned>((m + 127) / 128));
  gufo::hip::WKQuantA8BlockedWmmaGEMMKernel<128, 128, 2, 4, 2,
                                            core::GgmlType::kQ8_0, 32>
      <<<grid, 256, 0, stream>>>(w, x_tiled, out, batch, m, k);
}

}  // namespace gufo::models::qwen38_flash_next::rocm
