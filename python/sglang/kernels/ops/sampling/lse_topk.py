"""Fused per-row logsumexp + top-k (beam search selection)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Compiled kernel covers k up to this many candidates (k = 2 * beam_width,
# i.e. beam_width <= 32). Callers fall back to the torch reference above it.
MAX_K = 64


@cache_once
def _jit_lse_topk_module(dtype: torch.dtype) -> Module:
    """Compile and cache the JIT lse_topk module for a given dtype."""
    # Checks on the compile key live here, not in `lse_topk`: `cache_once`
    # keys on `dtype`, so this runs once per specialisation.
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError(
            f"Unsupported dtype {dtype}. Supported: float16, bfloat16, float32"
        )
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        "lse_topk",
        *args,
        cuda_files=["sampling/lse_topk.cuh"],
        cuda_wrappers=[("lse_topk", f"lse_topk<{args}>")],
    )


def lse_topk(
    x: torch.Tensor,
    k: int,
    out_vals: Optional[torch.Tensor] = None,
    out_indices: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row lse-normalized top-k in a single kernel.

    Returns ``(vals, indices)`` where ``vals[r, j] = x[r, indices[r, j]] -
    logsumexp(x[r])``, the top-k in descending value order (ties keep the
    lower index first). The per-row reduction order is fixed, so a row's
    output is batch-invariant: it does not depend on the other rows in ``x``.

    Parameters
    ----------
    x      : CUDA tensor ``[num_rows, vocab]`` (FP16 / BF16 / FP32)
    k      : number of candidates per row, ``1 <= k <= MAX_K``
    out_vals    : optional pre-allocated ``[num_rows, k]`` fp32 tensor
    out_indices : optional pre-allocated ``[num_rows, k]`` int64 tensor

    Returns
    -------
    ``(vals, indices)`` as above.
    """
    # Tensor invariants are validated in the C++ launcher; the compile-key
    # check lives in `_jit_lse_topk_module`.
    if not x.is_contiguous():
        x = x.contiguous()
    if out_vals is None:
        out_vals = torch.empty((x.shape[0], k), dtype=torch.float32, device=x.device)
    if out_indices is None:
        out_indices = torch.empty((x.shape[0], k), dtype=torch.int64, device=x.device)

    module = _jit_lse_topk_module(x.dtype)
    module.lse_topk(out_vals, out_indices, x, k)
    return out_vals, out_indices
