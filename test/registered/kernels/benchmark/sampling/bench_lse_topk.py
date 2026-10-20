"""Benchmark: fused lse_topk vs the torch cast + logsumexp + topk chain."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.sampling.lse_topk import lse_topk as jit_lse_topk
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large")


def torch_lse_topk(x, k):
    xf = x.float()
    lse = torch.logsumexp(xf, dim=-1, keepdim=True)
    v, t = torch.topk(xf, k, dim=-1)
    return v - lse, t


FN_MAP = {
    "jit": jit_lse_topk,
    "torch": torch_lse_topk,
}


@marker.parametrize(
    "rows,vocab",
    [(1, 151936), (20, 151936), (64, 151936), (20, 262144)],
    [(1, 151936), (20, 151936)],
)
@marker.benchmark("impl", ["jit", "torch"])
def benchmark(rows: int, vocab: int, impl: str):
    x = create_random(rows, vocab)
    k = 40
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(x, k),
        graph_clone_args=(0,),
        memory_args=(x,),
        memory_output=None,
    )


if __name__ == "__main__":
    benchmark.run()
