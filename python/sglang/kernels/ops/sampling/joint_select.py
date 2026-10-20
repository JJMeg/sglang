"""Fused beam-search joint selection (ranked candidates + stop check)."""

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

# Compiled kernel covers beam_width up to this value (num_out = 2 * width
# must fit the compiled candidate budget). Callers fall back to the torch
# reference above it.
MAX_WIDTH = 32


@cache_once
def _jit_joint_select_module() -> Module:
    """Compile and cache the JIT joint_select module (no dtype variants)."""
    args = make_cpp_args(False)  # PDL off: these kernels gain nothing from it
    return load_jit(
        "joint_select",
        *args,
        cuda_files=["sampling/joint_select.cuh"],
        cuda_wrappers=[("joint_select", f"joint_select<{args}>")],
    )


def joint_select(
    cum_logprobs: torch.Tensor,
    top_logprobs: torch.Tensor,
    top_tokens: torch.Tensor,
    stop_ids: torch.Tensor,
    beam_width: int,
    num_out: int | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Fused joint selection for one beam group, in a single kernel.

    Returns ``(next_tokens [k], parent_idx [k], new_cum_logprobs [k],
    num_survivors [1], fin_tokens [c], fin_parent_idx [c],
    fin_cum_logprobs [c], num_finished [1])`` with the same semantics and
    dump-slot zeroing as `srt.beam_search.joint_select.joint_select`.
    ``c = 2 * beam_width``; the frontier row count is taken from
    ``cum_logprobs`` (1 at the prefill first selection, k at decode).

    Parameters
    ----------
    num_out : candidates to rank; defaults to c (joint). Pass beam_width
        for the length-terminated final step.
    """
    k = beam_width
    c = 2 * k
    if num_out is None:
        num_out = c
    device = cum_logprobs.device
    next_tokens = torch.empty(k, dtype=torch.int64, device=device)
    parent_idx = torch.empty(k, dtype=torch.int64, device=device)
    new_cum_logprobs = torch.empty(k, dtype=torch.float32, device=device)
    num_survivors = torch.empty(1, dtype=torch.int64, device=device)
    fin_tokens = torch.empty(c, dtype=torch.int64, device=device)
    fin_parent_idx = torch.empty(c, dtype=torch.int64, device=device)
    fin_cum_logprobs = torch.empty(c, dtype=torch.float32, device=device)
    num_finished = torch.empty(1, dtype=torch.int64, device=device)

    module = _jit_joint_select_module()
    module.joint_select(
        cum_logprobs,
        top_logprobs,
        top_tokens,
        stop_ids,
        new_cum_logprobs,
        next_tokens,
        parent_idx,
        num_survivors,
        fin_cum_logprobs,
        fin_tokens,
        fin_parent_idx,
        num_finished,
        k,
        num_out,
        stop_ids.numel(),
    )
    return (
        next_tokens,
        parent_idx,
        new_cum_logprobs,
        num_survivors,
        fin_tokens,
        fin_parent_idx,
        fin_cum_logprobs,
        num_finished,
    )


def final_topk(
    cum_logprobs: torch.Tensor,
    top_logprobs: torch.Tensor,
    top_tokens: torch.Tensor,
    beam_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Length-terminated step: the top beam_width candidates overall.

    The same kernel with num_out = beam_width and an empty stop list, i.e.
    `srt.beam_search.joint_select.select_final_topk` semantics. Returns
    ``(tokens [k], parent_idx [k], cum_logprobs [k])``.
    """
    (
        next_tokens,
        parent_idx,
        new_cum_logprobs,
        _,
        _,
        _,
        _,
        _,
    ) = joint_select(
        cum_logprobs,
        top_logprobs,
        top_tokens,
        torch.empty(0, dtype=torch.int64, device=cum_logprobs.device),
        beam_width,
        num_out=beam_width,
    )
    return next_tokens, parent_idx, new_cum_logprobs
