"""Benchmark: fused joint_select vs the torch ranked-candidates chain."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.sampling.joint_select import joint_select as jit_joint_select
from sglang.srt.beam_search.joint_select import joint_select as torch_joint_select
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large")


def make_state(n):
    cum = -torch.rand(n, dtype=torch.float32, device="cuda") * 10.0
    top = -torch.rand(n, 2 * n, dtype=torch.float32, device="cuda") * 10.0
    tokens = torch.randint(0, 1024, (n, 2 * n), dtype=torch.int64, device="cuda")
    stops = torch.tensor([1, 2, 3], dtype=torch.int64, device="cuda")
    return cum, top, tokens, stops


def jit_fn(n, state):
    return jit_joint_select(*state, n)


def torch_fn(n, state):
    return torch_joint_select(*state, n)


FN_MAP = {"jit": jit_fn, "torch": torch_fn}


@marker.parametrize("n", [1, 4, 20, 32], [20])
@marker.benchmark("impl", ["jit", "torch"])
def benchmark(n: int, impl: str):
    state = make_state(n)
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(n, state),
        graph_clone_args=(),
        memory_args=state,
        memory_output=None,
    )


if __name__ == "__main__":
    benchmark.run()
