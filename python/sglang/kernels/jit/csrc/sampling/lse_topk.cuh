/*
 * Batch-invariant per-row logsumexp for beam search selection.
 *
 * Replaces torch.logsumexp, whose CUDA reduction order is not
 * batch-invariant: folding the leader and member logits into one tensor
 * would shift a row's lse by an ulp and can flip near-equal beams. This
 * kernel reduces every row with a fixed order (segment layout, warp tree,
 * CTA tree), so a row's lse is bit-identical regardless of the batch it
 * rides in -- which lets the coordinator merge the pieces into a single
 * tensor. The top-k itself stays torch.topk (fast, per-row independent).
 *
 * Three kernels, per-row segments in parallel:
 *   1. lse_topk_segment_max (rows x S blocks): float4 scan, segment max
 *      (fixed two-level tree) into scratch.
 *   2. lse_topk_segment_lse (rows x S blocks): re-scan the segment with
 *      expf(v - row_max), partial sum into scratch.
 *   3. lse_topk_finalize (rows blocks): thread 0 reduces the S partial
 *      max/sums in a fixed order: lse = max + log(sum).
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

namespace sglang {

static constexpr int kBlockSize = 128;
static constexpr int kMaxSegments = 64;

/**
 * \brief Segment max scan: partial max per (row, segment).
 *
 * \tparam kUsePDL Whether to emit the PDL wait/trigger pair
 * \param x        Input logits, `num_rows x vocab` (fp32, vocab % 4 == 0)
 * \param seg_max  Partial maxes, `num_rows x S`
 * \param vocab    Row width
 * \param S        Segments per row
 * \param seg_len  Segment width in elements (multiple of 4)
 */
template <bool kUsePDL>
__global__ void lse_topk_segment_max_kernel(const float* __restrict__ x,
                                            float* __restrict__ seg_max,
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

  float max_val = -FLT_MAX;
  for (int64_t i = start + static_cast<int64_t>(tid) * 4; i < start + n_vec;
       i += static_cast<int64_t>(kBlockSize) * 4) {
    const float4 v4 = reinterpret_cast<const float4*>(row_ptr + i)[0];
    max_val = fmaxf(max_val, fmaxf(fmaxf(v4.x, v4.y), fmaxf(v4.z, v4.w)));
  }
  for (int64_t i = start + n_vec + tid; i < end; i += kBlockSize) {
    max_val = fmaxf(max_val, row_ptr[i]);
  }

  // Fixed two-level tree: warp -> CTA.
  device::cta::reduce_max(max_val, scratch, -FLT_MAX);
  __syncthreads();
  if (tid == 0) {
    seg_max[row * S + seg] = scratch[0];
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

/**
 * \brief Segment lse partials: expf(v - row_max) sums.
 *
 * \tparam kUsePDL Whether to emit the PDL wait/trigger pair
 * \param x        Input logits, `num_rows x vocab`
 * \param seg_max  Segment maxes from the max kernel, `num_rows x S`
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
  // max kernel completed in a previous launch). Redundant across threads
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
 * \brief Fixed-order lse per row.
 *
 * \tparam kUsePDL Whether to emit the PDL wait/trigger pair
 * \param seg_max Segment maxes, `num_rows x S`
 * \param seg_sum Partial exp sums, `num_rows x S`
 * \param lse     Output logsumexp per row, `num_rows`
 * \param S       Segments per row
 */
template <bool kUsePDL>
__global__ void lse_topk_finalize_kernel(const float* __restrict__ seg_max,
                                         const float* __restrict__ seg_sum,
                                         float* __restrict__ lse,
                                         int S) {
  device::PDLWaitPrimary<kUsePDL>();

  if (threadIdx.x != 0) {
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  const int64_t row = blockIdx.x;

  float row_max = -INFINITY;
  float sum = 0.f;
  for (int s = 0; s < S; ++s) {
    row_max = fmaxf(row_max, seg_max[row * S + s]);
    sum += seg_sum[row * S + s];
  }
  lse[row] = row_max + logf(sum);

  device::PDLTriggerSecondary<kUsePDL>();
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

/**
 * \brief Validate the tensors and launch `lse_topk_segment_max_kernel`.
 *
 * \tparam kUsePDL Whether to launch with PDL enabled
 * \param x       Input tensor, `num_rows x vocab`, fp32, contiguous
 * \param seg_max Output scratch, `num_rows x S`, fp32
 * \param vocab   Row width (multiple of 4)
 * \param seg_len Segment width in elements (multiple of 4)
 * \param seg_count Segments per row (power of two <= kMaxSegments)
 */
template <bool kUsePDL>
void lse_topk_segment_max(tvm::ffi::TensorView x,
                          tvm::ffi::TensorView seg_max,
                          int64_t vocab,
                          int64_t seg_len,
                          int64_t seg_count) {
  using namespace host;

  SymbolicSize rows = {"rows"};
  SymbolicSize vocab_s = {"vocab"};
  SymbolicSize S_s = {"segments"};
  TensorMatcher({rows, vocab_s}).with_dtype<fp32_t>().verify(x);
  TensorMatcher({rows, S_s}).with_dtype<fp32_t>().verify(seg_max);
  const int64_t n_rows = rows.unwrap();
  const int64_t n_vocab = vocab_s.unwrap();
  const int64_t S = S_s.unwrap();

  CHECK_HOST(x.device().device_type == kDLCUDA)
      << "lse_topk: x must be a CUDA tensor";
  CHECK_HOST(n_rows > 0) << "lse_topk: num_rows must be > 0, got " << n_rows;
  CHECK_HOST(vocab == n_vocab && vocab % 4 == 0)
      << "lse_topk: vocab must match x and be a multiple of 4, got " << vocab;
  CHECK_HOST(seg_len > 0 && seg_len % 4 == 0)
      << "lse_topk: seg_len must be a positive multiple of 4, got " << seg_len;
  CHECK_HOST(S == seg_count && S >= 1 && S <= kMaxSegments &&
             (S & (S - 1)) == 0)
      << "lse_topk: segments must be a power of two in [1, " << kMaxSegments
      << "], got " << seg_count;

  LaunchKernel(static_cast<uint32_t>(n_rows * S), kBlockSize, x.device())
      .enable_pdl(kUsePDL)(lse_topk_segment_max_kernel<kUsePDL>,
                           static_cast<const float*>(x.data_ptr()),
                           static_cast<float*>(seg_max.data_ptr()),
                           n_vocab,
                           static_cast<int>(S),
                           static_cast<int>(seg_len));
}

/**
 * \brief Validate the tensors and launch `lse_topk_segment_lse_kernel`.
 *
 * \tparam kUsePDL Whether to launch with PDL enabled
 * \param x       Input tensor, `num_rows x vocab`, fp32
 * \param seg_max Segment maxes from the max kernel, `num_rows x S`
 * \param seg_sum Output partial sums, `num_rows x S`
 * \param vocab   Row width
 * \param seg_len Segment width in elements (multiple of 4)
 * \param seg_count Segments per row
 */
template <bool kUsePDL>
void lse_topk_segment_lse(tvm::ffi::TensorView x,
                          tvm::ffi::TensorView seg_max,
                          tvm::ffi::TensorView seg_sum,
                          int64_t vocab,
                          int64_t seg_len,
                          int64_t seg_count) {
  using namespace host;

  SymbolicSize rows = {"rows"};
  SymbolicSize vocab_s = {"vocab"};
  SymbolicSize S_s = {"segments"};
  TensorMatcher({rows, vocab_s}).with_dtype<fp32_t>().verify(x);
  TensorMatcher({rows, S_s}).with_dtype<fp32_t>().verify(seg_max);
  TensorMatcher({rows, S_s}).with_dtype<fp32_t>().verify(seg_sum);
  const int64_t n_rows = rows.unwrap();
  const int64_t n_vocab = vocab_s.unwrap();
  const int64_t S = S_s.unwrap();

  CHECK_HOST(x.device().device_type == kDLCUDA)
      << "lse_topk: x must be a CUDA tensor";
  CHECK_HOST(n_rows > 0) << "lse_topk: num_rows must be > 0, got " << n_rows;
  CHECK_HOST(vocab == n_vocab && vocab % 4 == 0)
      << "lse_topk: vocab must match x and be a multiple of 4, got " << vocab;
  CHECK_HOST(seg_len > 0 && seg_len % 4 == 0)
      << "lse_topk: seg_len must be a positive multiple of 4, got " << seg_len;
  CHECK_HOST(S == seg_count && S >= 1 && S <= kMaxSegments &&
             (S & (S - 1)) == 0)
      << "lse_topk: segments must be a power of two in [1, " << kMaxSegments
      << "], got " << seg_count;

  LaunchKernel(static_cast<uint32_t>(n_rows * S), kBlockSize, x.device())
      .enable_pdl(kUsePDL)(lse_topk_segment_lse_kernel<kUsePDL>,
                           static_cast<const float*>(x.data_ptr()),
                           static_cast<const float*>(seg_max.data_ptr()),
                           static_cast<float*>(seg_sum.data_ptr()),
                           n_vocab,
                           static_cast<int>(S),
                           static_cast<int>(seg_len));
}

/**
 * \brief Validate the tensors and launch `lse_topk_finalize_kernel`.
 *
 * \tparam kUsePDL Whether to launch with PDL enabled
 * \param seg_max / seg_sum  Scratch from the segment kernels
 * \param lse     Output logsumexp per row, `num_rows`, fp32
 * \param seg_count Segments per row
 */
template <bool kUsePDL>
void lse_topk_finalize(tvm::ffi::TensorView seg_max,
                       tvm::ffi::TensorView seg_sum,
                       tvm::ffi::TensorView lse,
                       int64_t seg_count) {
  using namespace host;

  SymbolicSize rows = {"rows"};
  SymbolicSize S_s = {"segments"};
  TensorMatcher({rows, S_s}).with_dtype<fp32_t>().verify(seg_max);
  TensorMatcher({rows, S_s}).with_dtype<fp32_t>().verify(seg_sum);
  TensorMatcher({rows}).with_dtype<fp32_t>().verify(lse);
  const int64_t n_rows = rows.unwrap();
  const int64_t S = S_s.unwrap();

  CHECK_HOST(seg_max.device().device_type == kDLCUDA)
      << "lse_topk: seg_max must be a CUDA tensor";
  CHECK_HOST(n_rows > 0) << "lse_topk: num_rows must be > 0, got " << n_rows;
  CHECK_HOST(S == seg_count && S >= 1 && S <= kMaxSegments &&
             (S & (S - 1)) == 0)
      << "lse_topk: segments must be a power of two in [1, " << kMaxSegments
      << "], got " << seg_count;

  LaunchKernel(static_cast<uint32_t>(n_rows), kBlockSize, seg_max.device())
      .enable_pdl(kUsePDL)(lse_topk_finalize_kernel<kUsePDL>,
                           static_cast<const float*>(seg_max.data_ptr()),
                           static_cast<const float*>(seg_sum.data_ptr()),
                           static_cast<float*>(lse.data_ptr()),
                           static_cast<int>(S));
}

}  // namespace sglang
