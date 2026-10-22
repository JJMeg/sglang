/*
 * Fused orphan-slot collection for beam search KV reclamation.
 *
 * Replaces the torch chain (flatten -> unique -> isin -> masked_select)
 * whose three synchronizing ops stall the scheduler thread per entry. One
 * kernel computes the distinct slots of old_mapping that new_mapping does
 * not reference: per-thread candidate collection into a fixed smem buffer,
 * a bitonic sort (sentinels pad the buffer and sort to the end), then a
 * single-thread dedup that compacts the result and writes the count. The
 * caller pays exactly one tiny count D2H instead of three syncs.
 *
 * Scope: E = old_mapping.numel() <= kMaxE (2048, i.e. beam_width x window
 * up to about 100 steps at width 20). Larger E falls back to the torch
 * reference in the caller.
 */

#include <sgl_kernel/tensor.h>  // TensorMatcher, SymbolicSize
#include <sgl_kernel/utils.h>   // CHECK_HOST
#include <sgl_kernel/utils.cuh> // LaunchKernel, SGL_DEVICE

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <limits>

namespace sglang {

static constexpr int kMaxE = 2048;
static constexpr int kBlockSize = 128;
static constexpr int kMaxLocal = kMaxE / kBlockSize;  // per-thread candidates

/**
 * \brief Distinct slots of old_mapping absent from new_mapping.
 *
 * \tparam kUsePDL    Whether to emit the PDL wait/trigger pair
 * \param old_mapping Slots before the remap, `E` (int32)
 * \param new_mapping Slots after the remap, `E`
 * \param out         Compacted distinct orphans, `E` (valid in [0, count))
 * \param count       Scalar int32 with the distinct orphan count
 * \param E           Number of slots (<= kMaxE)
 */
template <bool kUsePDL>
__global__ void orphan_collect_kernel(const int32_t* __restrict__ old_mapping,
                                      const int32_t* __restrict__ new_mapping,
                                      int64_t* __restrict__ out,
                                      int32_t* __restrict__ count,
                                      int E) {
  device::PDLWaitPrimary<kUsePDL>();

  const int tid = threadIdx.x;

  __shared__ int32_t new_s[kMaxE];
  __shared__ int32_t buf[kMaxE];

  for (int i = tid; i < E; i += kBlockSize) {
    new_s[i] = new_mapping[i];
  }
  // Sentinel padding: sorts to the end and never matches a real slot (real
  // slots are < 2^31; the allocator never hands out slot values >= INT32_MAX).
  for (int i = tid; i < kMaxE; i += kBlockSize) {
    buf[i] = INT32_MAX;
  }
  __syncthreads();

  // Per-thread candidate collection (linear membership scan over the smem
  // copy of new_mapping; kMaxE / kBlockSize elements per thread at most).
  int cnt = 0;
  for (int i = tid; i < E; i += kBlockSize) {
    const int32_t v = old_mapping[i];
    bool in_new = false;
    for (int j = 0; j < E; ++j) {
      if (new_s[j] == v) {
        in_new = true;
        break;
      }
    }
    if (!in_new) {
      buf[tid * kMaxLocal + cnt] = v;
      ++cnt;
    }
  }
  __syncthreads();

  // Bitonic sort (ascending) over the full buffer; sentinels rise to the
  // end. Standard shared-memory bitonic; 66 stages for 2048 elements.
  for (int k = 2; k <= kMaxE; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      __syncthreads();
      for (int base = tid; base < kMaxE; base += kBlockSize) {
        const int ixj = base ^ j;
        if (ixj > base) {
          const int32_t a = buf[base];
          const int32_t b = buf[ixj];
          const bool swap = ((base & k) == 0) ? (a > b) : (a < b);
          if (swap) {
            buf[base] = b;
            buf[ixj] = a;
          }
        }
      }
    }
  }
  __syncthreads();

  // Dedup + compact + count (single thread; <= kMaxE sequential steps).
  if (tid == 0) {
    int out_cnt = 0;
    int32_t prev = INT32_MAX;
    for (int i = 0; i < kMaxE; ++i) {
      const int32_t v = buf[i];
      if (v == INT32_MAX) {
        break;  // sorted: everything from here is padding
      }
      if (v != prev) {
        out[out_cnt] = static_cast<int64_t>(v);
        ++out_cnt;
        prev = v;
      }
    }
    count[0] = out_cnt;
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

/**
 * \brief Validate the tensors and launch `orphan_collect_kernel`.
 *
 * \tparam kUsePDL    Whether to launch with PDL enabled
 * \param old_mapping Input tensor, `E`, int32, contiguous
 * \param new_mapping Input tensor, `E`, int32, contiguous
 * \param out         Output tensor, `E`, int64
 * \param count       Output scalar int32
 * \param E           Number of slots (<= kMaxE)
 */
template <bool kUsePDL>
void orphan_collect(tvm::ffi::TensorView old_mapping,
                    tvm::ffi::TensorView new_mapping,
                    tvm::ffi::TensorView out,
                    tvm::ffi::TensorView count,
                    int64_t E) {
  using namespace host;

  SymbolicSize E_s = {"e"};
  TensorMatcher({E_s}).with_dtype<int32_t>().verify(old_mapping);
  TensorMatcher({E_s}).with_dtype<int32_t>().verify(new_mapping);
  TensorMatcher({E_s}).with_dtype<int64_t>().verify(out);
  TensorMatcher({1}).with_dtype<int32_t>().verify(count);
  const int64_t e = E_s.unwrap();

  CHECK_HOST(old_mapping.device().device_type == kDLCUDA)
      << "orphan_collect: old_mapping must be a CUDA tensor";
  CHECK_HOST(E == e) << "orphan_collect: E must match the tensors, got " << E;
  CHECK_HOST(E > 0 && E <= kMaxE)
      << "orphan_collect: E must be in [1, " << kMaxE << "], got " << E;

  LaunchKernel(1, kBlockSize, old_mapping.device())
      .enable_pdl(kUsePDL)(orphan_collect_kernel<kUsePDL>,
                           static_cast<const int32_t*>(old_mapping.data_ptr()),
                           static_cast<const int32_t*>(new_mapping.data_ptr()),
                           static_cast<int64_t*>(out.data_ptr()),
                           static_cast<int32_t*>(count.data_ptr()),
                           static_cast<int>(E));
}

}  // namespace sglang
