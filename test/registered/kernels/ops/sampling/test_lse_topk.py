"""Tests for the fused lse_topk JIT kernel (beam search selection)."""

import pytest
import torch

from sglang.kernels.ops.sampling.lse_topk import MAX_K, lse_topk
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def torch_reference(x, k):
    xf = x.float()
    lse = torch.logsumexp(xf, dim=-1, keepdim=True)
    v, t = torch.topk(xf, k, dim=-1)
    return v - lse, t


@pytest.mark.parametrize(
    "rows,vocab,k",
    [(1, 1024, 40), (20, 151936, 40), (2, 512, 1), (2, 65536, 64), (1, 7, 3)],
)
def test_matches_torch_fp32(rows, vocab, k):
    # Random fp32 values are distinct with probability ~1, so the index
    # order must match torch exactly.
    x = torch.randn(rows, vocab, dtype=torch.float32, device="cuda")
    vals, idx = lse_topk(x, k)
    ref_vals, ref_idx = torch_reference(x, k)
    torch.testing.assert_close(vals, ref_vals, rtol=1e-3, atol=1e-5)
    assert torch.equal(idx, ref_idx)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows,vocab,k", [(1, 4096, 40), (20, 151936, 40)])
def test_scores_and_order_fp16_bf16(dtype, rows, vocab, k):
    # fp16/bf16 cast to fp32 creates exact ties, so only the score multiset
    # is comparable to torch; the kernel's own order (value desc, index asc
    # within ties) must hold deterministically.
    x = torch.randn(rows, vocab, dtype=dtype, device="cuda")
    vals, idx = lse_topk(x, k)
    ref_vals, _ = torch_reference(x, k)
    torch.testing.assert_close(vals, ref_vals, rtol=1e-3, atol=1e-5)
    for r in range(rows):
        for j in range(k - 1):
            assert vals[r, j] >= vals[r, j + 1]
            if vals[r, j] == vals[r, j + 1]:
                assert idx[r, j] <= idx[r, j + 1]


def test_batch_invariance():
    # The property that lets the coordinator merge leader + member logits:
    # a row's output is bit-identical alone vs embedded in a larger batch.
    row = torch.randn(1, 65536, dtype=torch.float32, device="cuda")
    batch = torch.cat([torch.randn(3, 65536, device="cuda"), row])
    v_alone, i_alone = lse_topk(row, 40)
    v_batch, i_batch = lse_topk(batch, 40)
    assert torch.equal(v_alone, v_batch[3:4])
    assert torch.equal(i_alone, i_batch[3:4])


def test_suppressed_token_sentinel():
    # Suppressed logits are filled with finfo.min; they must never be
    # selected and must not poison the lse with NaN.
    x = torch.randn(4, 8192, dtype=torch.float32, device="cuda")
    x[:, 0] = torch.finfo(torch.float32).min
    vals, idx = lse_topk(x, 10)
    assert not torch.isnan(vals).any()
    assert not (idx == 0).any()


def test_non_contiguous_input():
    x = torch.randn(2, 8192, dtype=torch.float32, device="cuda")[..., ::2]
    assert not x.is_contiguous()
    vals, idx = lse_topk(x, 10)
    ref_vals, ref_idx = torch_reference(x, 10)
    torch.testing.assert_close(vals, ref_vals, rtol=1e-3, atol=1e-5)
    assert torch.equal(idx, ref_idx)


def test_cpu_input_raises():
    x = torch.randn(2, 128)
    with pytest.raises(RuntimeError, match="CUDA"):
        lse_topk(x, 10)


def test_k_too_large_raises():
    x = torch.randn(2, 128, device="cuda")
    with pytest.raises(RuntimeError, match="max"):
        lse_topk(x, MAX_K + 1)


def test_unsupported_dtype_raises():
    x = torch.randint(0, 10, (2, 128), dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="dtype"):
        lse_topk(x, 10)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
