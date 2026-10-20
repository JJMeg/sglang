# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Joint token selection for beam search: pure tensor functions.

Contract (overlap / cuda-graph ready): tensor-in / tensor-out, fixed output
shapes for a given (num_rows, num_candidates, beam_width) signature, no D2H
sync, no data-dependent host branches (the only python branch keys on a
static tensor shape). The caller owns the single sync point per step.

Selection semantics: walk the top num_candidates extensions in descending cumulative-logprob order; stop-token
candidates finish, non-stop candidates survive; a candidate is examined only
while fewer than beam_width survivors precede it in score order.
"""

from __future__ import annotations

import msgspec
import torch

from sglang.kernels.ops.sampling.joint_select import (
    MAX_WIDTH as JOINT_SELECT_MAX_WIDTH,
    final_topk as _final_topk_kernel,
    joint_select as _joint_select_kernel,
)


class SelectResult(msgspec.Struct):
    """Fixed-shape outputs of one expansion step. Valid in [0, num_survivors)
    and [0, num_finished); past that, dump-slot zeros that must not be read."""

    next_tokens: torch.Tensor  # [beam_width] int64
    parent_idx: torch.Tensor  # [beam_width] int64, row index into the input frontier
    new_cum_logprobs: torch.Tensor  # [beam_width] float32
    num_survivors: torch.Tensor  # [] int64; < beam_width means the group must finish
    fin_tokens: torch.Tensor  # [num_candidates] int64
    fin_parent_idx: torch.Tensor  # [num_candidates] int64
    fin_cum_logprobs: torch.Tensor  # [num_candidates] float32
    num_finished: torch.Tensor  # [] int64


class FinalSelect(msgspec.Struct):
    """Top-beam_width candidates of a length-terminated step (all finished)."""

    tokens: torch.Tensor  # [beam_width] int64
    parent_idx: torch.Tensor  # [beam_width] int64
    cum_logprobs: torch.Tensor  # [beam_width] float32


def _scatter_fixed(src: torch.Tensor, slot: torch.Tensor, size: int) -> torch.Tensor:
    # Fixed-shape compaction: element i lands at slot[i]; non-selected elements
    # all target the dump slot `size`, which is sliced away.
    buf = src.new_zeros(size + 1)
    buf.scatter_(0, slot, src)
    return buf[:size]


def _ranked_candidates(cum_logprobs, top_logprobs, top_tokens, num_out):
    num_candidates = top_logprobs.shape[1]
    scores = cum_logprobs.unsqueeze(1) + top_logprobs
    cand_scores, cand_idx = scores.reshape(-1).topk(num_out, sorted=True)
    parent = cand_idx // num_candidates
    tokens = top_tokens.reshape(-1).gather(0, cand_idx)
    return cand_scores, parent, tokens


def joint_select(
    cum_logprobs: torch.Tensor,  # [num_rows] float32, frontier cumulative logprobs
    top_logprobs: torch.Tensor,  # [num_rows, num_candidates] float32
    top_tokens: torch.Tensor,  # [num_rows, num_candidates] int64
    stop_token_ids: torch.Tensor,  # [num_stop] int64, may be empty (ignore_eos)
    beam_width: int,
) -> SelectResult:
    num_candidates = top_logprobs.shape[1]
    k = beam_width

    cand_scores, parent, tokens = _ranked_candidates(
        cum_logprobs, top_logprobs, top_tokens, num_candidates
    )

    if stop_token_ids.numel() > 0:  # static-shape branch, constant under capture
        is_stop = torch.isin(tokens, stop_token_ids)
    else:
        is_stop = torch.zeros_like(tokens, dtype=torch.bool)

    non_stop = ~is_stop
    non_stop_rank = non_stop.long().cumsum(0)  # 1-based at non-stop positions
    survivor = non_stop & (non_stop_rank <= k)
    # Examined while fewer than k survivors strictly precede the candidate.
    examined = (non_stop_rank - non_stop.long()) < k
    finished = is_stop & examined
    fin_rank = finished.long().cumsum(0)

    surv_slot = torch.where(survivor, non_stop_rank - 1, k)
    fin_slot = torch.where(finished, fin_rank - 1, num_candidates)

    return SelectResult(
        next_tokens=_scatter_fixed(tokens, surv_slot, k),
        parent_idx=_scatter_fixed(parent, surv_slot, k),
        new_cum_logprobs=_scatter_fixed(cand_scores, surv_slot, k),
        num_survivors=survivor.long().sum(),
        fin_tokens=_scatter_fixed(tokens, fin_slot, num_candidates),
        fin_parent_idx=_scatter_fixed(parent, fin_slot, num_candidates),
        fin_cum_logprobs=_scatter_fixed(cand_scores, fin_slot, num_candidates),
        num_finished=finished.long().sum(),
    )


def select_final_topk(
    cum_logprobs: torch.Tensor,  # [num_rows] float32
    top_logprobs: torch.Tensor,  # [num_rows, num_candidates] float32
    top_tokens: torch.Tensor,  # [num_rows, num_candidates] int64
    beam_width: int,
) -> FinalSelect:
    """Length-terminated step: the best beam_width extensions all finish. No
    stop check needed -- the caller decides this step from the step counter."""
    cand_scores, parent, tokens = _ranked_candidates(
        cum_logprobs, top_logprobs, top_tokens, beam_width
    )
    return FinalSelect(tokens=tokens, parent_idx=parent, cum_logprobs=cand_scores)


def joint_select_fused(
    cum_logprobs: torch.Tensor,
    top_logprobs: torch.Tensor,
    top_tokens: torch.Tensor,
    stop_token_ids: torch.Tensor,
    beam_width: int,
) -> SelectResult:
    """`joint_select` on the fused JIT kernel (beam_width <= JOINT_SELECT_MAX_WIDTH)."""
    (
        next_tokens,
        parent_idx,
        new_cum_logprobs,
        num_survivors,
        fin_tokens,
        fin_parent_idx,
        fin_cum_logprobs,
        num_finished,
    ) = _joint_select_kernel(
        cum_logprobs, top_logprobs, top_tokens, stop_token_ids, beam_width
    )
    return SelectResult(
        next_tokens=next_tokens,
        parent_idx=parent_idx,
        new_cum_logprobs=new_cum_logprobs,
        num_survivors=num_survivors,
        fin_tokens=fin_tokens,
        fin_parent_idx=fin_parent_idx,
        fin_cum_logprobs=fin_cum_logprobs,
        num_finished=num_finished,
    )


def select_final_topk_fused(
    cum_logprobs: torch.Tensor,
    top_logprobs: torch.Tensor,
    top_tokens: torch.Tensor,
    beam_width: int,
) -> FinalSelect:
    """`select_final_topk` on the fused JIT kernel (beam_width <= JOINT_SELECT_MAX_WIDTH)."""
    tokens, parent_idx, cum_logprobs = _final_topk_kernel(
        cum_logprobs, top_logprobs, top_tokens, beam_width
    )
    return FinalSelect(tokens=tokens, parent_idx=parent_idx, cum_logprobs=cum_logprobs)
