/*
 * Fused beam-search joint selection: ranked candidates + stop check + the
 * survivor/finished split, in one kernel instead of the ~10-op torch chain
 * (broadcast add -> topk -> floor_divide -> gather -> isin -> cumsum ->
 * where -> scatter ...).
 *
 * Semantics (identical to srt/beam_search/joint_select.py):
 *   - scores[r, c] = cum_logprobs[r] + top_logprobs[r, c] over the flattened
 *     n * c candidate grid; keep the top num_out sorted (ties keep the lower
 *     flat index first -- deterministic).
 *   - walk the sorted list: stop candidates finish while fewer than k
 *     survivors strictly precede them, non-stop candidates fill the k
 *     survivor slots. Dump slots past the counts are zeroed, bit-faithful to
 *     _scatter_fixed.
 * The same kernel serves the length-terminated step: num_out = k with an
 * empty stop list returns the plain top-k (select_final_topk semantics).
 *
 * n (frontier rows) and k (beam width) are independent: the prefill first
 * selection has n = 1 row expanding to k survivors, decode steps have
 * n = k. c = 2 * k is the candidate count per row.
 *
 * Algorithm (one block per group):
 *   1. Load cum_logprobs and the stop ids into shared memory.
 *   2. Strided scan over the n * c candidates: per-thread top-num_out list
 *      (insertion sort, descending).
 *   3. Merge tree over the variable-length thread lists in shared memory.
 *   4. Thread 0 walks the final list and writes the outputs.
 *
 * Scope: beam_width <= kJMaxWidth (32), so c = 2 * width <= 64 and
 * n * c <= 2048. Larger widths fall back to the torch reference in the
 * caller.
 */

#include <sgl_kernel/tensor.h>  // TensorMatcher, SymbolicSize
#include <sgl_kernel/type.cuh>  // fp32_t
#include <sgl_kernel/utils.h>   // CHECK_HOST
#include <sgl_kernel/utils.cuh> // LaunchKernel, SGL_DEVICE

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

// Covers beam_width <= 32 (num_out = 2 * width <= 64). Bump to raise the
// covered width; shared memory grows as kJBlockSize * kJMaxOut * 8 bytes.
static constexpr int kJMaxWidth = 32;
static constexpr int kJMaxOut = 64;
static constexpr int kJBlockSize = 64;
static constexpr int kJMaxStop = 64;

struct JointPair {
  float score;
  int32_t idx;  // flat index into [n, c]; n * c <= 2048 fits comfortably
};

/**
 * \brief Fused joint selection for one beam group.
 *
 * \tparam kUsePDL          Whether to emit the PDL wait/trigger pair
 * \param cum_logprobs      Frontier cumulative logprobs, `n`
 * \param top_logprobs      Per-row candidate logprobs, `n x c`
 * \param top_tokens        Per-row candidate tokens, `n x c` (int64)
 * \param stop_ids          Stop token ids, `num_stop` (may be empty)
 * \param new_cum_logprobs  Survivor scores, `k`
 * \param next_tokens       Survivor tokens, `k`
 * \param parent_idx        Survivor parent rows, `k`
 * \param num_survivors     Scalar int64
 * \param fin_cum_logprobs  Finished scores, `c`
 * \param fin_tokens        Finished tokens, `c`
 * \param fin_parent_idx    Finished parent rows, `c`
 * \param num_finished      Scalar int64
 * \param n                 Frontier row count (1 at prefill, k at decode)
 * \param c                 Candidate count per row (2 * k)
 * \param k                 Beam width (survivor slot count)
 * \param num_out           Candidates to rank (c joint, k final)
 * \param num_stop          Stop id count (0 = plain top-k / final step)
 */
template <bool kUsePDL>
__global__ void joint_select_kernel(const float* __restrict__ cum_logprobs,
                                    const float* __restrict__ top_logprobs,
                                    const int64_t* __restrict__ top_tokens,
                                    const int64_t* __restrict__ stop_ids,
                                    float* __restrict__ new_cum_logprobs,
                                    int64_t* __restrict__ next_tokens,
                                    int64_t* __restrict__ parent_idx,
                                    int64_t* __restrict__ num_survivors,
                                    float* __restrict__ fin_cum_logprobs,
                                    int64_t* __restrict__ fin_tokens,
                                    int64_t* __restrict__ fin_parent_idx,
                                    int64_t* __restrict__ num_finished,
                                    int n,
                                    int c,
                                    int k,
                                    int num_out,
                                    int num_stop) {
  device::PDLWaitPrimary<kUsePDL>();

  __shared__ float cum_s[kJMaxWidth];
  __shared__ int64_t stop_s[kJMaxStop];
  __shared__ JointPair pairs[kJBlockSize * kJMaxOut];
  __shared__ int cnts[kJBlockSize];

  const int tid = threadIdx.x;
  const int N = n * c;

  if (tid < n) {
    cum_s[tid] = cum_logprobs[tid];
  }
  if (tid < num_stop) {
    stop_s[tid] = stop_ids[tid];
  }
  __syncthreads();

  // -- strided scan: per-thread top-num_out list (insertion sort, desc) --
  float lv[kJMaxOut];
  int32_t li[kJMaxOut];
  int cnt = 0;
  for (int i = tid; i < N; i += kJBlockSize) {
    const float score = cum_s[i / c] + top_logprobs[i];
    if (cnt < num_out || score > lv[num_out - 1]) {
      // Ties keep the lower flat index first (deterministic order).
      int p = cnt < num_out ? cnt : num_out - 1;
      while (p > 0 && lv[p - 1] < score) {
        lv[p] = lv[p - 1];
        li[p] = li[p - 1];
        --p;
      }
      lv[p] = score;
      li[p] = i;
      if (cnt < num_out) {
        ++cnt;
      }
    }
  }

  // -- per-thread lists into shared memory --
  JointPair* slot = pairs + tid * kJMaxOut;
  for (int j = 0; j < cnt; ++j) {
    slot[j].score = lv[j];
    slot[j].idx = li[j];
  }
  cnts[tid] = cnt;
  __syncthreads();

  // -- merge tree over variable-length lists --
  // B is read and written by the same thread, so it is staged in local
  // arrays first; A is read-only during the merge.
  float bv[kJMaxOut];
  int32_t bi[kJMaxOut];
  for (int stride = 1; stride < kJBlockSize; stride *= 2) {
    if (tid % (2 * stride) == 0 && tid + stride < kJBlockSize) {
      JointPair* A = pairs + tid * kJMaxOut;
      JointPair* B = A + stride * kJMaxOut;
      const int la = cnts[tid];
      const int lb = cnts[tid + stride];
      for (int j = 0; j < lb; ++j) {
        bv[j] = B[j].score;
        bi[j] = B[j].idx;
      }
      int a = 0;
      int b = 0;
      const int out_len = num_out < la + lb ? num_out : la + lb;
      for (int o = 0; o < out_len; ++o) {
        const bool take_a =
            a < la &&
            (b >= lb || A[a].score > bv[b] ||
             (A[a].score == bv[b] && A[a].idx <= bi[b]));
        if (take_a) {
          B[o] = A[a++];
        } else {
          B[o].score = bv[b];
          B[o].idx = bi[b];
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

  // -- walk: survivors / finished split (thread 0; num_out <= 64) --
  if (tid == 0) {
    int s = 0;
    int f = 0;
    const int out_len = cnts[0];
    for (int o = 0; o < out_len; ++o) {
      const int idx = pairs[o].idx;
      const int parent = idx / c;
      const int64_t tok = top_tokens[idx];
      bool is_stop = false;
      for (int j = 0; j < num_stop; ++j) {
        if (tok == stop_s[j]) {
          is_stop = true;
          break;
        }
      }
      if (is_stop) {
        if (s < k) {
          fin_cum_logprobs[f] = pairs[o].score;
          fin_tokens[f] = tok;
          fin_parent_idx[f] = parent;
          ++f;
        }
      } else if (s < k) {
        new_cum_logprobs[s] = pairs[o].score;
        next_tokens[s] = tok;
        parent_idx[s] = parent;
        ++s;
      }
    }
    // Zero the dump slots (bit-faithful to _scatter_fixed's zero dump slot).
    for (int j = s; j < k; ++j) {
      new_cum_logprobs[j] = 0.f;
      next_tokens[j] = 0;
      parent_idx[j] = 0;
    }
    for (int j = f; j < c; ++j) {
      fin_cum_logprobs[j] = 0.f;
      fin_tokens[j] = 0;
      fin_parent_idx[j] = 0;
    }
    num_survivors[0] = s;
    num_finished[0] = f;
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

/**
 * \brief Validate the tensors and launch `joint_select_kernel`.
 *
 * \tparam kUsePDL Whether to launch with PDL enabled
 * \param cum_logprobs / top_logprobs / top_tokens / stop_ids  Inputs
 * \param new_cum_logprobs / next_tokens / parent_idx /
 *        num_survivors / fin_* / num_finished               Outputs
 * \param k          Beam width (survivor slot count, <= kJMaxWidth)
 * \param num_out    Candidates to rank (2 * k joint, k final)
 * \param num_stop   Stop id count (0 = plain top-k / final step)
 */
template <bool kUsePDL>
void joint_select(tvm::ffi::TensorView cum_logprobs,
                  tvm::ffi::TensorView top_logprobs,
                  tvm::ffi::TensorView top_tokens,
                  tvm::ffi::TensorView stop_ids,
                  tvm::ffi::TensorView new_cum_logprobs,
                  tvm::ffi::TensorView next_tokens,
                  tvm::ffi::TensorView parent_idx,
                  tvm::ffi::TensorView num_survivors,
                  tvm::ffi::TensorView fin_cum_logprobs,
                  tvm::ffi::TensorView fin_tokens,
                  tvm::ffi::TensorView fin_parent_idx,
                  tvm::ffi::TensorView num_finished,
                  int64_t k,
                  int64_t num_out,
                  int64_t num_stop) {
  using namespace host;

  SymbolicSize N_s = {"n"};
  SymbolicSize C_s = {"c"};
  SymbolicSize K_s = {"k"};
  SymbolicSize S_s = {"num_stop"};

  TensorMatcher({N_s}).with_dtype<fp32_t>().verify(cum_logprobs);
  TensorMatcher({N_s, C_s}).with_dtype<fp32_t>().verify(top_logprobs);
  TensorMatcher({N_s, C_s}).with_dtype<int64_t>().verify(top_tokens);
  TensorMatcher({S_s}).with_dtype<int64_t>().verify(stop_ids);
  TensorMatcher({K_s}).with_dtype<fp32_t>().verify(new_cum_logprobs);
  TensorMatcher({K_s}).with_dtype<int64_t>().verify(next_tokens);
  TensorMatcher({K_s}).with_dtype<int64_t>().verify(parent_idx);
  TensorMatcher({1}).with_dtype<int64_t>().verify(num_survivors);
  TensorMatcher({C_s}).with_dtype<fp32_t>().verify(fin_cum_logprobs);
  TensorMatcher({C_s}).with_dtype<int64_t>().verify(fin_tokens);
  TensorMatcher({C_s}).with_dtype<int64_t>().verify(fin_parent_idx);
  TensorMatcher({1}).with_dtype<int64_t>().verify(num_finished);

  const int64_t n = N_s.unwrap();
  const int64_t c = C_s.unwrap();
  const int64_t k_out = K_s.unwrap();
  const int64_t s = S_s.unwrap();

  // Device checks are explicit instead of TensorMatcher.with_device, whose
  // signature differs across installed jit header versions.
  CHECK_HOST(cum_logprobs.device().device_type == kDLCUDA)
      << "joint_select: cum_logprobs must be a CUDA tensor";
  CHECK_HOST(n >= 1 && n <= kJMaxWidth)
      << "joint_select: n must be in [1, " << kJMaxWidth << "], got " << n;
  CHECK_HOST(k >= 1 && k <= kJMaxWidth)
      << "joint_select: k must be in [1, " << kJMaxWidth << "], got " << k;
  CHECK_HOST(k_out == k)
      << "joint_select: survivor output size must equal k, got " << k_out;
  CHECK_HOST(c == 2 * k) << "joint_select: c must be 2 * k, got " << c;
  CHECK_HOST(num_out >= 1 && num_out <= kJMaxOut && num_out <= c)
      << "joint_select: num_out must be in [1, min(" << kJMaxOut
      << ", c)], got " << num_out;
  CHECK_HOST(num_stop == s) << "joint_select: num_stop must match stop_ids";
  CHECK_HOST(num_stop >= 0 && num_stop <= kJMaxStop)
      << "joint_select: num_stop must be in [0, " << kJMaxStop << "], got "
      << num_stop;

  // The launch device comes from the tensor itself; the TensorMatcher chains
  // above stay on shape/dtype checks only.
  LaunchKernel(1, kJBlockSize, cum_logprobs.device())
      .enable_pdl(kUsePDL)(
          joint_select_kernel<kUsePDL>,
          static_cast<const float*>(cum_logprobs.data_ptr()),
          static_cast<const float*>(top_logprobs.data_ptr()),
          static_cast<const int64_t*>(top_tokens.data_ptr()),
          static_cast<const int64_t*>(stop_ids.data_ptr()),
          static_cast<float*>(new_cum_logprobs.data_ptr()),
          static_cast<int64_t*>(next_tokens.data_ptr()),
          static_cast<int64_t*>(parent_idx.data_ptr()),
          static_cast<int64_t*>(num_survivors.data_ptr()),
          static_cast<float*>(fin_cum_logprobs.data_ptr()),
          static_cast<int64_t*>(fin_tokens.data_ptr()),
          static_cast<int64_t*>(fin_parent_idx.data_ptr()),
          static_cast<int64_t*>(num_finished.data_ptr()),
          static_cast<int>(n),
          static_cast<int>(c),
          static_cast<int>(k),
          static_cast<int>(num_out),
          static_cast<int>(num_stop));
}

}  // namespace sglang
