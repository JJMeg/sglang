"""Tests for the fused joint_select JIT kernel (beam search selection)."""

import pytest
import torch

from sglang.kernels.ops.sampling.joint_select import (
    MAX_WIDTH,
    final_topk as fused_final_topk,
    joint_select as fused_joint_select,
)
from sglang.srt.beam_search.joint_select import (
    joint_select as torch_joint_select,
    select_final_topk as torch_final_topk,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def make_inputs(rows, width, vocab, seed):
    """Integer-valued scores, so all rows * 2*width candidate sums are
    distinct and the fused kernel and the torch reference must agree
    exactly. Rows (frontier size) and width (beam width) are independent:
    the prefill first selection has rows=1."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    cum = -torch.arange(rows, dtype=torch.float32, device="cuda") * 10000.0
    top = -(torch.arange(2 * width, dtype=torch.float32, device="cuda") + 100.0)
    top = top.repeat(rows, 1)
    tokens = torch.randint(
        0, vocab, (rows, 2 * width), dtype=torch.int64, device="cuda", generator=g
    )
    return cum, top, tokens


def assert_select_equal(fused, ref):
    nt, p, ncl, ns, ft, fp, fcl, nf = fused
    rnt, rp, rncl, rns, rft, rfp, rfcl, rnf = (
        ref.next_tokens,
        ref.parent_idx,
        ref.new_cum_logprobs,
        ref.num_survivors,
        ref.fin_tokens,
        ref.fin_parent_idx,
        ref.fin_cum_logprobs,
        ref.num_finished,
    )
    for a, b in ((nt, rnt), (p, rp), (ncl, rncl), (ns, rns), (nf, rnf)):
        assert torch.equal(a, b), (a, b)
    for a, b in ((ft, rft), (fp, rfp), (fcl, rfcl)):
        assert torch.equal(a, b), (a, b)


@pytest.mark.parametrize("n", [1, 4, 20, 32])
@pytest.mark.parametrize("seed", [0, 1])
def test_matches_torch_with_stops(n, seed):
    cum, top, tokens = make_inputs(n, n, vocab=1024, seed=seed)
    # Stop ids: two known tokens plus one that never appears.
    stops = torch.tensor(
        [tokens[0, 0].item(), tokens[0, 1].item(), 9999],
        dtype=torch.int64,
        device="cuda",
    )
    fused = fused_joint_select(cum, top, tokens, stops, n)
    ref = torch_joint_select(cum, top, tokens, stops, n)
    assert_select_equal(fused, ref)


@pytest.mark.parametrize("n", [4, 20])
def test_matches_torch_no_stops(n):
    cum, top, tokens = make_inputs(n, n, vocab=1024, seed=7)
    stops = torch.empty(0, dtype=torch.int64, device="cuda")
    fused = fused_joint_select(cum, top, tokens, stops, n)
    ref = torch_joint_select(cum, top, tokens, stops, n)
    assert_select_equal(fused, ref)


def test_all_stop_finishes_everything():
    n = 8
    cum, top, tokens = make_inputs(n, n, vocab=8, seed=3)
    stops = torch.arange(8, dtype=torch.int64, device="cuda")  # all tokens
    fused = fused_joint_select(cum, top, tokens, stops, n)
    ref = torch_joint_select(cum, top, tokens, stops, n)
    assert_select_equal(fused, ref)
    assert fused[3].item() == 0  # num_survivors
    assert fused[7].item() == 2 * n  # num_finished == num_candidates


@pytest.mark.parametrize("n", [1, 20, 32])
def test_final_topk_matches_torch(n):
    cum, top, tokens = make_inputs(n, n, vocab=1024, seed=11)
    t, p, cums = fused_final_topk(cum, top, tokens, n)
    ref = torch_final_topk(cum, top, tokens, n)
    assert torch.equal(t, ref.tokens)
    assert torch.equal(p, ref.parent_idx)
    assert torch.equal(cums, ref.cum_logprobs)


def test_prefill_first_selection_shape():
    # The prefill first selection has one frontier row expanding to k
    # survivors: rows=1, beam_width=20 (the shape that regressed).
    cum, top, tokens = make_inputs(1, 20, vocab=1024, seed=13)
    stops = torch.tensor(
        [tokens[0, 0].item(), 9999], dtype=torch.int64, device="cuda"
    )
    fused = fused_joint_select(cum, top, tokens, stops, 20)
    ref = torch_joint_select(cum, top, tokens, stops, 20)
    assert_select_equal(fused, ref)


def test_final_topk_prefill_shape():
    # max_new_tokens == 1 terminates at prefill: rows=1, beam_width=20.
    cum, top, tokens = make_inputs(1, 20, vocab=1024, seed=14)
    t, p, cums = fused_final_topk(cum, top, tokens, 20)
    ref = torch_final_topk(cum, top, tokens, 20)
    assert torch.equal(t, ref.tokens)
    assert torch.equal(p, ref.parent_idx)
    assert torch.equal(cums, ref.cum_logprobs)


def test_tie_contract_lowest_idx_first_and_deterministic():
    # All scores equal: the deterministic contract is value-desc with the
    # lower flat index first, so survivors are flat indices 0..n-1.
    n = 8
    cum = torch.zeros(n, dtype=torch.float32, device="cuda")
    top = torch.zeros(n, 2 * n, dtype=torch.float32, device="cuda")
    tokens = torch.arange(2 * n * n, dtype=torch.int64, device="cuda").view(
        n, 2 * n
    )
    stops = torch.empty(0, dtype=torch.int64, device="cuda")
    fused = fused_joint_select(cum, top, tokens, stops, n)
    again = fused_joint_select(cum, top, tokens, stops, n)
    for a, b in zip(fused, again):
        assert torch.equal(a, b)
    # Survivors = first n flat candidates, in index order.
    assert fused[3].item() == n
    assert torch.equal(fused[1], torch.arange(n, dtype=torch.int64, device="cuda") // (2 * n))
    assert torch.equal(fused[0], tokens.view(-1)[:n])


def test_width_too_large_raises():
    n = MAX_WIDTH + 1
    cum, top, tokens = make_inputs(1, n, vocab=1024, seed=5)
    stops = torch.empty(0, dtype=torch.int64, device="cuda")
    with pytest.raises(RuntimeError, match="k must be"):
        fused_joint_select(cum, top, tokens, stops, n)


def test_cpu_input_raises():
    cum = torch.zeros(4, dtype=torch.float32)
    top = torch.zeros(4, 8, dtype=torch.float32)
    tokens = torch.zeros(4, 8, dtype=torch.int64)
    stops = torch.empty(0, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="CUDA"):
        fused_joint_select(cum, top, tokens, stops, 4)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
