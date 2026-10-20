/*
 * Fused per-row logsumexp + top-k for beam search selection.
 *
 * Replaces the three-op chain (cast -> logsumexp -> topk) that the beam
 * coordinator runs per decode step. Three kernels, per-row segments in
 * parallel:
 *
 *   1. lse_topk_segment_scan  (rows x S blocks): each block scans one
 *      segment of one row with float4 loads, computes the segment max
 *      (fixed two-level tree) and the segment top-k (per-thread insertion
 *      lists + shared-memory merge tree), and writes both to scratch.
 *   2. lse_topk_segment_lse   (rows x S blocks): each block re-scans its
 *      segment with expf(v - row_max) and writes the partial sum.
 *   3. lse_topk_finalize      (rows blocks): thread 0 reduces the S partial
 *      max/sums in a fixed order (lse = max + log(sum)), merges the S
 *      sorted segment lists, and writes vals = topk - lse and the indices.
 *
 * Determinism contract: every reduction and merge order is fixed per row
 * (segment geometry, reduction trees, tie rule value-desc / index-asc), so
 * a row's output is bit-identical regardless of the batch it rides in --
 * the property that lets the coordinator merge leader and member logits.
 *
 * Scope: k <= kMaxCandidates (64, i.e. beam_width <= 32), vocab % 4 == 0
 * for the float4 path. Larger k or odd vocab fall back to the torch
 * reference in the wrapper.
 */

#include <sgl_kernel/cta.cuh>   // cta::reduce_max
#include <sgl_kernel/tensor.h>  // TensorMatcher, SymbolicSize
#include <sgl_kernel/type.cuh>  // fp32_t
#include <sgl_kernel/utils.h>   // CHECK_HOST
#include <sgl_kernel/utils.cuh> // LaunchKernel, SGL_DEVICE, kWarpThreads
#include <sgl_kernel/warp.cuh>  // warp::reduce_sum

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>

namespace sglang {

// Covers beam_width <= 32 (k = 2 * beam_width). Segments per row: enough to
// fill the GPU for the beam row counts (1..64) so the scan is
// bandwidth-bound instead of latency-bound (each block contributes ~4
// concurrent float4 loads per warp; ~7 blocks per SM saturate DRAM).
static constexpr int kMaxCandidates = 64;
static constexpr int kBlockSize = 64;
static constexpr int kMaxSegments = 64;
// The finalize kernel loads one list per thread, so the thread count must
// cover the segment count.
static_assert(kBlockSize == kMaxSegments, "kBlockSize must equal kMaxSegments");

struct LseTopkPair {
  float value;
  int32_t index;  // vocab < 2^31 is checked in the launcher
};

// Insertion into a descending list; the strictly-less shift keeps the lower
// index first among equal values (deterministic tie order).
SGL_DEVICE void lse_topk_insert(
    float v, int32_t idx, float* lv, int32_t* li, int& cnt, int k) {
  if (cnt < k || v > lv[k - 1]) {
    int p = cnt < k ? cnt : k - 1;
    while (p > 0 && lv[p - 1] < v) {
      lv[p] = lv[p - 1];
      li[p] = li[p - 1];
      --p;
    }
    lv[p] = v;
    li[p] = idx;
    if (cnt < k) {
      ++cnt;
    }
  }
}

/**
 * \brief Segment scan: partial max + segment top-k.
 *
 * \tparam kUsePDL   Whether to emit the PDL wait/trigger pair
 * \param x          Input logits, `num_rows x vocab` (fp32, vocab % 4 == 0)
 * \param seg_max    Partial segment maxes, `num_rows x S`
 * \param topk_values Segment top-k values, `num_rows x S x kMax` (sentinel
 *                   -infinity past the segment's real entries)
 * \param topk_idx   Segment top-k indices (int32), same layout
 * \param vocab      Row width
 * \param k          Candidates per row (<= kMaxCandidates)
 * \param S          Segments per row
 * \param seg_len    Segment width in elements (multiple of 4)
 */
template <bool kUsePDL>
__global__ void lse_topk_segment_scan_kernel(const float* __restrict__ x,
                                             float* __restrict__ seg_max,
                                             float* __restrict__ topk_values,
                                             int32_t* __restrict__ topk_idx,
                                             int64_t vocab,
                                             int k,
                                             int S,
                                             int seg_len) {
  device::PDLWaitPrimary<kUsePDL>();

  const int64_t row = blockIdx.x / S;
  const int seg = blockIdx.x % S;
  const int tid = threadIdx.x;

  // The pair region doubles as float scratch for the max reduction; the
  // phases are separated by __syncthreads, so no live data ever overlaps.
  __shared__ LseTopkPair pairs[kBlockSize * kMaxCandidates];
  __shared__ int cnts[kBlockSize];
  float* scratch = reinterpret_cast<float*>(pairs);

  const float* row_ptr = x + row * vocab;
  const int64_t start = static_cast<int64_t>(seg) * seg_len;
  const int64_t end = start + seg_len < vocab ? start + seg_len : vocab;
  const int64_t n = end - start;
  const int64_t n_vec = n & ~static_cast<int64_t>(3);

  // -- one scan: segment max + top-k (the top-k works on raw values, so it
  //    does not need the max first; the max is reduced after the scan) --
  float max_val = -FLT_MAX;
  float lv[kMaxCandidates];
  int32_t li[kMaxCandidates];
  int cnt = 0;
  for (int64_t i = start + static_cast<int64_t>(tid) * 4; i < start + n_vec;
       i += static_cast<int64_t>(kBlockSize) * 4) {
    const float4 v4 = reinterpret_cast<const float4*>(row_ptr + i)[0];
    max_val = fmaxf(max_val, fmaxf(fmaxf(v4.x, v4.y), fmaxf(v4.z, v4.w)));
    lse_topk_insert(v4.x, static_cast<int32_t>(i), lv, li, cnt, k);
    lse_topk_insert(v4.y, static_cast<int32_t>(i + 1), lv, li, cnt, k);
    lse_topk_insert(v4.z, static_cast<int32_t>(i + 2), lv, li, cnt, k);
    lse_topk_insert(v4.w, static_cast<int32_t>(i + 3), lv, li, cnt, k);
  }
  for (int64_t i = start + n_vec + tid; i < end; i += kBlockSize) {
    max_val = fmaxf(max_val, row_ptr[i]);
    lse_topk_insert(row_ptr[i], static_cast<int32_t>(i), lv, li, cnt, k);
  }
  // Fixed two-level tree: warp -> CTA.
  device::cta::reduce_max(max_val, scratch, -FLT_MAX);
  __syncthreads();
  max_val = scratch[0];

  // -- per-thread lists into shared memory --
  LseTopkPair* slot = pairs + tid * kMaxCandidates;
  for (int j = 0; j < cnt; ++j) {
    slot[j].value = lv[j];
    slot[j].index = li[j];
  }
  cnts[tid] = cnt;
  __syncthreads();

  // -- merge tree over the variable-length lists --
  float bv[kMaxCandidates];
  int32_t bi[kMaxCandidates];
  for (int stride = 1; stride < kBlockSize; stride *= 2) {
    if (tid % (2 * stride) == 0 && tid + stride < kBlockSize) {
      LseTopkPair* A = pairs + tid * kMaxCandidates;
      LseTopkPair* B = A + stride * kMaxCandidates;
      const int la = cnts[tid];
      const int lb = cnts[tid + stride];
      for (int j = 0; j < lb; ++j) {
        bv[j] = B[j].value;
        bi[j] = B[j].index;
      }
      int a = 0;
      int b = 0;
      const int out_len = k < la + lb ? k : la + lb;
      for (int o = 0; o < out_len; ++o) {
        const bool take_a =
            a < la && (b >= lb || A[a].value > bv[b] ||
                       (A[a].value == bv[b] && A[a].index <= bi[b]));
        if (take_a) {
          B[o] = A[a++];
        } else {
          B[o].value = bv[b];
          B[o].index = bi[b];
          ++b;
        }
      }
      for (int j = 0; j < out_len; ++j) {
        A[j] = B[j];
      }
      cnts[tid] = out_len;
    }
    __syncthreads();
  }

  // -- write out: partial max + segment top-k with sentinel padding --
  if (tid == 0) {
    const int64_t out_row = row * S + seg;
    seg_max[out_row] = max_val;
    const int out_len = cnts[0];
    float* vals_out = topk_values + out_row * kMaxCandidates;
    int32_t* idx_out = topk_idx + out_row * kMaxCandidates;
    for (int j = 0; j < out_len; ++j) {
      vals_out[j] = pairs[j].value;
      idx_out[j] = pairs[j].index;
    }
    for (int j = out_len; j < kMaxCandidates; ++j) {
      // -infinity so a padded entry can never beat a real (possibly
      // finfo.min-suppressed) value in the finalize merge.
      vals_out[j] = -INFINITY;
      idx_out[j] = -1;
    }
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

/**
 * \brief Segment lse partials: expf(v - row_max) sums.
 *
 * \tparam kUsePDL Whether to emit the PDL wait/trigger pair
 * \param x        Input logits, `num_rows x vocab`
 * \param seg_max  Segment maxes from the scan kernel, `num_rows x S`
 * \param seg_sum  Partial exp sums, `num_rows x S`
 * \param vocab    Row width
 * \param S        Segments per row
 * \param seg_len  Segment width in elements (multiple of 4)
 */
template <bool kUsePDL>
__global__ void lse_topk_segment_lse_kernel(const float* __restrict__ x,
                                            const float* __restrict__ seg_max,
                                            float* __restrict__ seg_sum,
                                            int64_t vocab,
                                            int S,
                                            int seg_len) {
  device::PDLWaitPrimary<kUsePDL>();

  const int64_t row = blockIdx.x / S;
  const int seg = blockIdx.x % S;
  const int tid = threadIdx.x;

  __shared__ float scratch[kBlockSize / 32 + 1];

  const float* row_ptr = x + row * vocab;
  const int64_t start = static_cast<int64_t>(seg) * seg_len;
  const int64_t end = start + seg_len < vocab ? start + seg_len : vocab;
  const int64_t n = end - start;
  const int64_t n_vec = n & ~static_cast<int64_t>(3);

  // The row max is fixed: every thread reduces the S partials in order (the
  // scan kernel completed in a previous launch). Redundant across threads
  // but branch-free and cheaper than a broadcast.
  float row_max = -INFINITY;
  {
    const float* p = seg_max + row * S;
    for (int s = 0; s < S; ++s) {
      row_max = fmaxf(row_max, p[s]);
    }
  }

  float sum = 0.f;
  for (int64_t i = start + static_cast<int64_t>(tid) * 4; i < start + n_vec;
       i += static_cast<int64_t>(kBlockSize) * 4) {
    const float4 v4 = reinterpret_cast<const float4*>(row_ptr + i)[0];
    sum += expf(v4.x - row_max) + expf(v4.y - row_max) + expf(v4.z - row_max) +
           expf(v4.w - row_max);
  }
  for (int64_t i = start + n_vec + tid; i < end; i += kBlockSize) {
    sum += expf(row_ptr[i] - row_max);
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
  block_sum = device::warp::reduce_sum<device::kWarpThreads, true>(block_sum);
  // block_sum is valid in warp 0 lanes; only thread 0 writes it out.
  if (tid == 0) {
    seg_sum[row * S + seg] = block_sum;
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

/**
 * \brief Finalize one row: fixed-order lse + S-list merge + outputs.
 *
 * \tparam kUsePDL Whether to emit the PDL wait/trigger pair
 * \param seg_max   Segment maxes, `num_rows x S`
 * \param seg_sum   Partial exp sums, `num_rows x S`
 * \param topk_values Segment top-k values, `num_rows x S x kMax`
 * \param topk_idx  Segment top-k indices, same layout
 * \param vals      Output top-k minus lse, `num_rows x k`
 * \param indices   Output top-k indices (int64), `num_rows x k`
 * \param k         Candidates per row
 * \param S         Segments per row
 */
template <bool kUsePDL>
__global__ void lse_topk_finalize_kernel(const float* __restrict__ seg_max,
                                         const float* __restrict__ seg_sum,
                                         const float* __restrict__ topk_values,
                                         const int32_t* __restrict__ topk_idx,
                                         float* __restrict__ vals,
                                         int64_t* __restrict__ indices,
                                         int k,
                                         int S) {
  device::PDLWaitPrimary<kUsePDL>();

  __shared__ LseTopkPair lists[kMaxSegments * kMaxCandidates];
  const int tid = threadIdx.x;
  const int64_t row = blockIdx.x;

  // Load: thread t copies list t (all lists are sentinel-padded to the
  // full kMaxCandidates width by the scan kernel).
  const float* src = topk_values + row * S * kMaxCandidates + tid * kMaxCandidates;
  const int32_t* src_idx =
      topk_idx + row * S * kMaxCandidates + tid * kMaxCandidates;
  LseTopkPair* dst = lists + tid * kMaxCandidates;
  for (int j = 0; j < kMaxCandidates; ++j) {
    dst[j].value = src[j];
    dst[j].index = src_idx[j];
  }
  __syncthreads();

  // Parallel merge tree: 64 lists -> 1 (log2 rounds; each round's pairs are
  // merged by distinct threads, so the whole merge is ~6 sequential rounds
  // instead of 63).
  float bv[kMaxCandidates];
  int32_t bi[kMaxCandidates];
  for (int stride = 1; stride < kMaxSegments; stride *= 2) {
    if (tid % (2 * stride) == 0 && tid + stride < kMaxSegments) {
      LseTopkPair* A = lists + tid * kMaxCandidates;
      LseTopkPair* B = A + stride * kMaxCandidates;
      for (int j = 0; j < kMaxCandidates; ++j) {
        bv[j] = B[j].value;
        bi[j] = B[j].index;
      }
      int a = 0;
      int b = 0;
      for (int o = 0; o < kMaxCandidates; ++o) {
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
      for (int j = 0; j < kMaxCandidates; ++j) {
        A[j] = B[j];
      }
    }
    __syncthreads();
  }

  // Fixed-order lse reduction + outputs (thread 0; the merge above ended
  // with a __syncthreads).
  if (tid == 0) {
    float row_max = -INFINITY;
    float sum = 0.f;
    for (int s = 0; s < S; ++s) {
      row_max = fmaxf(row_max, seg_max[row * S + s]);
      sum += seg_sum[row * S + s];
    }
    const float lse = row_max + logf(sum);
    for (int j = 0; j < k; ++j) {
      vals[row * k + j] = lists[j].value - lse;
      indices[row * k + j] = static_cast<int64_t>(lists[j].index);
    }
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

/**
 * \brief Validate the tensors and launch `lse_topk_segment_scan_kernel`.
 *
 * \tparam kUsePDL Whether to launch with PDL enabled
 * \param x          Input tensor, `num_rows x vocab`, fp32, contiguous
 * \param seg_max    Output scratch, `num_rows x S`, fp32
 * \param topk_values Output scratch, `num_rows x S x kMax`, fp32
 * \param topk_idx   Output scratch, `num_rows x S x kMax`, int32
 * \param k          Candidates per row (<= kMaxCandidates)
 * \param vocab      Row width (multiple of 4)
 * \param seg_len    Segment width in elements (multiple of 4)
 */
template <bool kUsePDL>
void lse_topk_segment_scan(tvm::ffi::TensorView x,
                           tvm::ffi::TensorView seg_max,
                           tvm::ffi::TensorView topk_values,
                           tvm::ffi::TensorView topk_idx,
                           int64_t k,
                           int64_t vocab,
                           int64_t seg_len) {
  using namespace host;

  SymbolicSize rows = {"rows"};
  SymbolicSize vocab_s = {"vocab"};
  TensorMatcher({rows, vocab_s}).with_dtype<fp32_t>().verify(x);
  TensorMatcher({rows, kMaxSegments}).with_dtype<fp32_t>().verify(seg_max);
  TensorMatcher({rows, kMaxSegments, kMaxCandidates})
      .with_dtype<fp32_t>()
      .verify(topk_values);
  TensorMatcher({rows, kMaxSegments, kMaxCandidates})
      .with_dtype<int32_t>()
      .verify(topk_idx);
  const int64_t n_rows = rows.unwrap();
  const int64_t n_vocab = vocab_s.unwrap();

  CHECK_HOST(x.device().device_type == kDLCUDA)
      << "lse_topk: x must be a CUDA tensor";
  CHECK_HOST(n_rows > 0) << "lse_topk: num_rows must be > 0, got " << n_rows;
  CHECK_HOST(k > 0 && k <= kMaxCandidates)
      << "lse_topk: k must be in [1, " << kMaxCandidates << "], got " << k;
  CHECK_HOST(k <= n_vocab) << "lse_topk: k exceeds vocab " << k << " > "
                           << n_vocab;
  CHECK_HOST(vocab == n_vocab && vocab % 4 == 0)
      << "lse_topk: vocab must match x and be a multiple of 4, got " << vocab;
  CHECK_HOST(seg_len > 0 && seg_len % 4 == 0)
      << "lse_topk: seg_len must be a positive multiple of 4, got " << seg_len;
  CHECK_HOST(n_vocab <=
             static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
      << "lse_topk: vocab exceeds the int32 index range, got " << n_vocab;

  LaunchKernel(static_cast<uint32_t>(n_rows) * kMaxSegments,
               kBlockSize,
               x.device())
      .enable_pdl(kUsePDL)(lse_topk_segment_scan_kernel<kUsePDL>,
                           static_cast<const float*>(x.data_ptr()),
                           static_cast<float*>(seg_max.data_ptr()),
                           static_cast<float*>(topk_values.data_ptr()),
                           static_cast<int32_t*>(topk_idx.data_ptr()),
                           n_vocab,
                           static_cast<int>(k),
                           kMaxSegments,
                           static_cast<int>(seg_len));
}

/**
 * \brief Validate the tensors and launch `lse_topk_segment_lse_kernel`.
 *
 * \tparam kUsePDL Whether to launch with PDL enabled
 * \param x       Input tensor, `num_rows x vocab`, fp32
 * \param seg_max Segment maxes from the scan kernel, `num_rows x S`
 * \param seg_sum Output partial sums, `num_rows x S`
 * \param vocab   Row width
 * \param seg_len Segment width in elements (multiple of 4)
 */
template <bool kUsePDL>
void lse_topk_segment_lse(tvm::ffi::TensorView x,
                          tvm::ffi::TensorView seg_max,
                          tvm::ffi::TensorView seg_sum,
                          int64_t vocab,
                          int64_t seg_len) {
  using namespace host;

  SymbolicSize rows = {"rows"};
  SymbolicSize vocab_s = {"vocab"};
  TensorMatcher({rows, vocab_s}).with_dtype<fp32_t>().verify(x);
  TensorMatcher({rows, kMaxSegments}).with_dtype<fp32_t>().verify(seg_max);
  TensorMatcher({rows, kMaxSegments}).with_dtype<fp32_t>().verify(seg_sum);
  const int64_t n_rows = rows.unwrap();
  const int64_t n_vocab = vocab_s.unwrap();

  CHECK_HOST(x.device().device_type == kDLCUDA)
      << "lse_topk: x must be a CUDA tensor";
  CHECK_HOST(n_rows > 0) << "lse_topk: num_rows must be > 0, got " << n_rows;
  CHECK_HOST(vocab == n_vocab && vocab % 4 == 0)
      << "lse_topk: vocab must match x and be a multiple of 4, got " << vocab;
  CHECK_HOST(seg_len > 0 && seg_len % 4 == 0)
      << "lse_topk: seg_len must be a positive multiple of 4, got " << seg_len;

  LaunchKernel(static_cast<uint32_t>(n_rows) * kMaxSegments,
               kBlockSize,
               x.device())
      .enable_pdl(kUsePDL)(lse_topk_segment_lse_kernel<kUsePDL>,
                           static_cast<const float*>(x.data_ptr()),
                           static_cast<const float*>(seg_max.data_ptr()),
                           static_cast<float*>(seg_sum.data_ptr()),
                           n_vocab,
                           kMaxSegments,
                           static_cast<int>(seg_len));
}

/**
 * \brief Validate the tensors and launch `lse_topk_finalize_kernel`.
 *
 * \tparam kUsePDL Whether to launch with PDL enabled
 * \param seg_max / seg_sum / topk_values / topk_idx  Scratch from the
 *        segment kernels
 * \param vals      Output top-k minus lse, `num_rows x k`, fp32
 * \param indices   Output top-k indices (int64), `num_rows x k`
 * \param k         Candidates per row
 */
template <bool kUsePDL>
void lse_topk_finalize(tvm::ffi::TensorView seg_max,
                       tvm::ffi::TensorView seg_sum,
                       tvm::ffi::TensorView topk_values,
                       tvm::ffi::TensorView topk_idx,
                       tvm::ffi::TensorView vals,
                       tvm::ffi::TensorView indices,
                       int64_t k) {
  using namespace host;

  SymbolicSize rows = {"rows"};
  SymbolicSize k_s = {"k"};
  TensorMatcher({rows, kMaxSegments}).with_dtype<fp32_t>().verify(seg_max);
  TensorMatcher({rows, kMaxSegments}).with_dtype<fp32_t>().verify(seg_sum);
  TensorMatcher({rows, kMaxSegments, kMaxCandidates})
      .with_dtype<fp32_t>()
      .verify(topk_values);
  TensorMatcher({rows, kMaxSegments, kMaxCandidates})
      .with_dtype<int32_t>()
      .verify(topk_idx);
  TensorMatcher({rows, k_s}).with_dtype<fp32_t>().verify(vals);
  TensorMatcher({rows, k_s}).with_dtype<int64_t>().verify(indices);
  const int64_t n_rows = rows.unwrap();
  const int64_t k_out = k_s.unwrap();

  CHECK_HOST(seg_max.device().device_type == kDLCUDA)
      << "lse_topk: seg_max must be a CUDA tensor";
  CHECK_HOST(n_rows > 0) << "lse_topk: num_rows must be > 0, got " << n_rows;
  CHECK_HOST(k > 0 && k <= kMaxCandidates)
      << "lse_topk: k must be in [1, " << kMaxCandidates << "], got " << k;
  CHECK_HOST(k_out == k)
      << "lse_topk: output size must equal k, got " << k_out;

  LaunchKernel(static_cast<uint32_t>(n_rows), kBlockSize, seg_max.device())
      .enable_pdl(kUsePDL)(lse_topk_finalize_kernel<kUsePDL>,
                           static_cast<const float*>(seg_max.data_ptr()),
                           static_cast<const float*>(seg_sum.data_ptr()),
                           static_cast<const float*>(topk_values.data_ptr()),
                           static_cast<const int32_t*>(topk_idx.data_ptr()),
                           static_cast<float*>(vals.data_ptr()),
                           static_cast<int64_t*>(indices.data_ptr()),
                           static_cast<int>(k),
                           kMaxSegments);
}

}  // namespace sglang
