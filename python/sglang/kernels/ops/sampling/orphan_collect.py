"""Fused orphan-slot collection (beam search KV reclamation)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Compiled kernel covers up to this many slots per entry (beam_width x
# window). Callers fall back to the torch reference above it.
MAX_E = 2048


@cache_once
def _jit_orphan_collect_module() -> Module:
    """Compile and cache the JIT orphan_collect module."""
    args = make_cpp_args(False)  # PDL off: these kernels gain nothing from it
    return load_jit(
        "orphan_collect",
        *args,
        cuda_files=["sampling/orphan_collect.cuh"],
        cuda_wrappers=[("orphan_collect", f"orphan_collect<{args}>")],
    )


def collect_orphan_slots_fused(
    old_mapping: torch.Tensor,
    new_mapping: torch.Tensor,
) -> torch.Tensor:
    """Distinct slots of old_mapping absent from new_mapping.

    One fused kernel (bitonic sort + dedup) plus a single tiny count D2H,
    replacing the torch unique/isin/masked_select chain that synchronizes
    three times per reclaim.

    Parameters
    ----------
    old_mapping : CUDA tensor, any shape, int32, `numel() <= MAX_E`
    new_mapping : CUDA tensor, same shape, int32

    Returns
    -------
    The compacted distinct orphan slots (int64).
    """
    old = old_mapping.contiguous().view(-1)
    new = new_mapping.contiguous().view(-1)
    if old.dtype != torch.int32:
        old = old.to(torch.int32)
    if new.dtype != torch.int32:
        new = new.to(torch.int32)
    e = old.numel()
    out = torch.empty(e, dtype=torch.int64, device=old.device)
    count = torch.empty(1, dtype=torch.int32, device=old.device)

    module = _jit_orphan_collect_module()
    module.orphan_collect(old, new, out, count, e)
    return out[: count.item()]
