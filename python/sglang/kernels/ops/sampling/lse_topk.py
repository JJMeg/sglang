"""Fused per-row logsumexp + top-k (beam search selection).

Three JIT kernels: per-row segments scan (max + top-k) in parallel, segment
lse partials, and a fixed-order finalize. The torch reference covers k above
the compiled budget and exotic vocabs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Compiled kernel covers k up to this many candidates (k = 2 * beam_width,
# i.e. beam_width <= 32), and vocabs divisible by 4 (float4 path). Callers
# fall back to the torch reference above / outside that.
MAX_K = 64
_SEGMENTS = 64  # must match kMaxSegments in lse_topk.cuh


@cache_once
def _jit_lse_topk_module() -> Module:
    """Compile and cache the JIT lse_topk module (fp32 kernels only)."""
    args = make_cpp_args(False)  # PDL off: these kernels gain nothing from it
    return load_jit(
        "lse_topk",
        *args,
        cuda_files=["sampling/lse_topk.cuh"],
        cuda_wrappers=[
            ("lse_topk_segment_scan", f"lse_topk_segment_scan<{args}>"),
            ("lse_topk_segment_lse", f"lse_topk_segment_lse<{args}>"),
            ("lse_topk_finalize", f"lse_topk_finalize<{args}>"),
        ],
    )


def _torch_reference(
    x: torch.Tensor,
    k: int,
    out_vals: torch.Tensor,
    out_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    lse = torch.logsumexp(x, dim=-1, keepdim=True)
    v, t = torch.topk(x, k, dim=-1)
    out_vals.copy_(v - lse)
    out_indices.copy_(t)
    return out_vals, out_indices


def lse_topk(
    x: torch.Tensor,
    k: int,
    out_vals: Optional[torch.Tensor] = None,
    out_indices: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row lse-normalized top-k.

    Returns ``(vals, indices)`` where ``vals[r, j] = x[r, indices[r, j]] -
    logsumexp(x[r])``, the top-k in descending value order (ties keep the
    lower index first). Every reduction order is fixed per row, so a row's
    output is batch-invariant: it does not depend on the other rows in ``x``.

    Parameters
    ----------
    x      : CUDA tensor ``[num_rows, vocab]`` (any float dtype; cast to fp32)
    k      : number of candidates per row, ``1 <= k <= MAX_K``
    out_vals    : optional pre-allocated ``[num_rows, k]`` fp32 tensor
    out_indices : optional pre-allocated ``[num_rows, k]`` int64 tensor

    Returns
    -------
    ``(vals, indices)`` as above.
    """
    # Tensor invariants are validated in the C++ launchers; the compile-key
    # checks live in `_jit_lse_topk_module`.
    if not x.is_contiguous():
        x = x.contiguous()
    if x.dtype != torch.float32:
        x = x.float()
    rows, vocab = x.shape
    device = x.device
    if out_vals is None:
        out_vals = torch.empty((rows, k), dtype=torch.float32, device=device)
    if out_indices is None:
        out_indices = torch.empty((rows, k), dtype=torch.int64, device=device)
    if k > MAX_K or vocab % 4 != 0:
        return _torch_reference(x, k, out_vals, out_indices)

    seg_len = (vocab + _SEGMENTS - 1) // _SEGMENTS
    seg_len = (seg_len + 3) & ~3  # float4 alignment
    seg_max = torch.empty((rows, _SEGMENTS), dtype=torch.float32, device=device)
    seg_sum = torch.empty((rows, _SEGMENTS), dtype=torch.float32, device=device)
    topk_values = torch.empty(
        (rows, _SEGMENTS, MAX_K), dtype=torch.float32, device=device
    )
    topk_idx = torch.empty(
        (rows, _SEGMENTS, MAX_K), dtype=torch.int32, device=device
    )

    module = _jit_lse_topk_module()
    module.lse_topk_segment_scan(
        x, seg_max, topk_values, topk_idx, k, vocab, seg_len, _SEGMENTS
    )
    module.lse_topk_segment_lse(x, seg_max, seg_sum, vocab, seg_len, _SEGMENTS)
    module.lse_topk_finalize(
        seg_max, seg_sum, topk_values, topk_idx, out_vals, out_indices, k, _SEGMENTS
    )
    return out_vals, out_indices
