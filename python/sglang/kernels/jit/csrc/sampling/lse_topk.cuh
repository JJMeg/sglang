/*
 * Fused per-row logsumexp + top-k for beam search selection.
 *
 * Replaces the three-op chain (cast -> logsumexp -> topk) that the beam
 * coordinator runs per decode step. Each row is reduced with a fixed order
 * (warp tree -> CTA tree), so a row's lse is bit-identical regardless of the
 * batch it rides in -- the property torch.logsumexp does not have on CUDA,
 * and the one that lets the coordinator merge the leader and member logits
 * into a single tensor without shifting any row's lse by an ulp.
 *
 * Algorithm per row (one block of kBlockSize threads):
 *   1. CTA max reduction over the row (scratch region in shared memory).
 *   2. Single strided scan: accumulate exp(v - max) and keep a per-thread
 *      top-k list (insertion into a descending list; ties keep the lower
 *      index first, so the order is deterministic).
 *   3. Block sum: warp reductions -> scratch -> final warp reduction.
 *   4. Merge tree over the per-thread lists in shared memory; the merged
 *      result lands in slot 0.
 *   5. Thread 0 writes out vals = topk - lse and the indices (int64).
 *
 * Scope: k <= kMaxCandidates (64, i.e. beam_width <= 32). Larger k falls
 * back to the torch reference in the caller.
 */

#include <sgl_kernel/cta.cuh>   // cta::reduce_max
#include <sgl_kernel/tensor.h>  // TensorMatcher, SymbolicSize
#include <sgl_kernel/type.cuh>  // fp16_t, bf16_t, fp32_t
#include <sgl_kernel/utils.h>   // CHECK_HOST
#include <sgl_kernel/utils.cuh> // LaunchKernel, SGL_DEVICE, kWarpThreads
#include <sgl_kernel/warp.cuh>  // warp::reduce_sum

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cfloat>
#include <cstdint>
#include <limits>

namespace sglang {

// Covers beam_width <= 32 (k = 2 * beam_width). Bump to raise the covered
// width; shared memory grows as kBlockSize * kMaxCandidates * 8 bytes.
static constexpr int kMaxCandidates = 64;
static constexpr int kBlockSize = 64;

struct LseTopkPair {
  float value;
  int32_t index;  // vocab < 2^31 is checked in the launcher
};

template <typename T>
SGL_DEVICE float lse_topk_to_float(T x) {
  return static_cast<float>(x);
}

/**
 * \brief Fused per-row logsumexp + top-k.
 *
 * \tparam T             Input element type: fp16_t | bf16_t | fp32_t
 * \tparam kUsePDL       Whether to emit the PDL wait/trigger pair
 * \param x              Input logits, `num_rows x vocab`
 * \param vals           Output top-k values minus the row lse, `num_rows x k`
 * \param indices        Output top-k indices (int64), `num_rows x k`
 * \param vocab          Row width
 * \param k              Number of candidates per row (<= kMaxCandidates)
 */
template <typename T, bool kUsePDL>
__global__ void lse_topk_kernel(const T* __restrict__ x,
                                float* __restrict__ vals,
                                int64_t* __restrict__ indices,
                                int64_t vocab,
                                int k) {
  device::PDLWaitPrimary<kUsePDL>();

  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;

  // The pair region doubles as float scratch for the reductions; the phases
  // are separated by __syncthreads, so no live data ever overlaps.
  __shared__ LseTopkPair pairs[kBlockSize * kMaxCandidates];
  float* scratch = reinterpret_cast<float*>(pairs);

  // -- row max: strided scan -> warp tree -> CTA tree (fixed order) --
  float max_val = -FLT_MAX;
  for (int64_t i = tid; i < vocab; i += kBlockSize) {
    const float v = lse_topk_to_float(x[row * vocab + i]);
    max_val = v > max_val ? v : max_val;
  }
  device::cta::reduce_max(max_val, scratch, -FLT_MAX);
  __syncthreads();
  max_val = scratch[0];

  // -- exp-sum + per-thread top-k in one strided pass --
  float sum = 0.f;
  float lv[kMaxCandidates];
  int32_t li[kMaxCandidates];
  int cnt = 0;
  for (int64_t i = tid; i < vocab; i += kBlockSize) {
    const float v = lse_topk_to_float(x[row * vocab + i]);
    sum += expf(v - max_val);
    if (cnt < k || v > lv[k - 1]) {
      // Insertion into a descending list; the strictly-less shift keeps the
      // lower index first among equal values (deterministic tie order).
      int p = cnt < k ? cnt : k - 1;
      while (p > 0 && lv[p - 1] < v) {
        lv[p] = lv[p - 1];
        li[p] = li[p - 1];
        --p;
      }
      lv[p] = v;
      li[p] = static_cast<int32_t>(i);
      if (cnt < k) {
        ++cnt;
      }
    }
  }

  // -- block sum: warp reductions -> scratch -> final warp reduction --
  __syncthreads();
  sum = device::warp::reduce_sum<device::kWarpThreads, true>(sum);
  if ((tid & (device::kWarpThreads - 1)) == 0) {
    scratch[tid / device::kWarpThreads] = sum;
  }
  __syncthreads();
  float block_sum = 0.f;
  if (tid < kBlockSize / device::kWarpThreads) {
    block_sum = scratch[tid];
  }
  // All scratch reads are done; the slot writes below may now overwrite it.
  __syncthreads();
  block_sum = device::warp::reduce_sum<device::kWarpThreads, true>(block_sum);
  // block_sum is valid in warp 0 lanes; only thread 0 consumes it.
  const float lse = max_val + logf(block_sum);

  // -- per-thread lists into shared memory --
  LseTopkPair* slot = pairs + static_cast<int64_t>(tid) * kMaxCandidates;
  for (int j = 0; j < k; ++j) {
    slot[j].value = lv[j];
    slot[j].index = li[j];
  }
  __syncthreads();

  // -- merge tree over the thread lists --
  // Each round merges the sorted lists in slots tid and tid + stride into
  // slot tid. B is read and written by the same thread, so it is staged in
  // local arrays first; A is read-only during the merge.
  float bv[kMaxCandidates];
  int32_t bi[kMaxCandidates];
  for (int stride = 1; stride < kBlockSize; stride *= 2) {
    if (tid % (2 * stride) == 0 && tid + stride < kBlockSize) {
      LseTopkPair* A = pairs + static_cast<int64_t>(tid) * kMaxCandidates;
      LseTopkPair* B = A + static_cast<int64_t>(stride) * kMaxCandidates;
      for (int j = 0; j < k; ++j) {
        bv[j] = B[j].value;
        bi[j] = B[j].index;
      }
      int a = 0;
      int b = 0;
      // a + b == o < k throughout, so both heads stay in range.
      for (int o = 0; o < k; ++o) {
        const bool take_a =
            A[a].value > bv[b] || (A[a].value == bv[b] && A[a].index <= bi[b]);
        if (take_a) {
          B[o] = A[a++];
        } else {
          B[o].value = bv[b];
          B[o].index = bi[b];
          ++b;
        }
      }
      for (int j = 0; j < k; ++j) {
        A[j] = B[j];
      }
    }
    __syncthreads();
  }

  // -- epilogue --
  if (tid == 0) {
    for (int j = 0; j < k; ++j) {
      vals[row * k + j] = pairs[j].value - lse;
      indices[row * k + j] = static_cast<int64_t>(pairs[j].index);
    }
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

/**
 * \brief Validate the tensors and launch `lse_topk_kernel`.
 *
 * \tparam T       Input element type: fp16_t | bf16_t | fp32_t
 * \tparam kUsePDL Whether to launch with PDL enabled
 * \param vals     Output tensor, `num_rows x k`, fp32, contiguous
 * \param indices  Output tensor, `num_rows x k`, int64, contiguous
 * \param x        Input tensor, `num_rows x vocab`, contiguous
 * \param k        Number of candidates per row (<= kMaxCandidates)
 */
template <typename T, bool kUsePDL>
void lse_topk(tvm::ffi::TensorView vals,
              tvm::ffi::TensorView indices,
              tvm::ffi::TensorView x,
              int64_t k) {
  using namespace host;

  SymbolicSize num_rows = {"num_rows"};
  SymbolicSize vocab = {"vocab"};

  TensorMatcher({num_rows, vocab})
      .with_dtype<T>()
      .verify(x);
  const int64_t rows = num_rows.unwrap();
  const int64_t n_vocab = vocab.unwrap();
  TensorMatcher({num_rows, k})
      .with_dtype<fp32_t>()
      .verify(vals);
  TensorMatcher({num_rows, k})
      .with_dtype<int64_t>()
      .verify(indices);

  // Device checks are explicit instead of TensorMatcher.with_device, whose
  // signature differs across installed jit header versions.
  CHECK_HOST(x.device().device_type == kDLCUDA)
      << "lse_topk: x must be a CUDA tensor";
  CHECK_HOST(vals.device().device_type == kDLCUDA)
      << "lse_topk: vals must be a CUDA tensor";
  CHECK_HOST(indices.device().device_type == kDLCUDA)
      << "lse_topk: indices must be a CUDA tensor";

  CHECK_HOST(rows > 0) << "lse_topk: num_rows must be > 0, got " << rows;
  CHECK_HOST(k > 0) << "lse_topk: k must be > 0, got " << k;
  CHECK_HOST(k <= kMaxCandidates) << "lse_topk: k exceeds the compiled max of "
                                  << kMaxCandidates << ", got " << k;
  CHECK_HOST(k <= n_vocab) << "lse_topk: k exceeds vocab " << k << " > "
                           << n_vocab;
  CHECK_HOST(n_vocab <=
             static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
      << "lse_topk: vocab exceeds the int32 index range, got " << n_vocab;

  // The launch device comes from the tensor itself; the TensorMatcher chains
  // above stay on shape/dtype checks only, so the launcher compiles against
  // both old and new jit header versions.
  LaunchKernel(static_cast<uint32_t>(rows), kBlockSize, x.device())
      .enable_pdl(kUsePDL)(lse_topk_kernel<T, kUsePDL>,
                           static_cast<const T*>(x.data_ptr()),
                           static_cast<float*>(vals.data_ptr()),
                           static_cast<int64_t*>(indices.data_ptr()),
                           n_vocab,
                           static_cast<int>(k));
}

}  // namespace sglang
