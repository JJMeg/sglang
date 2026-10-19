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
"""Wall-clock timing for beam search phases, for the [BEAMPROF] log.

``timed_phase`` accumulates per-phase ms into a per-group ``BeamProfile``;
the coordinator formats it once at group finish. With no profile (profiling
disabled), ``timed_phase`` measures nothing.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator, Optional

import msgspec
from msgspec import field


class PhaseTimer(msgspec.Struct):
    """Accumulator for one phase: sample count, total, and worst case in ms."""

    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0

    def add(self, ms: float) -> None:
        self.count += 1
        self.total_ms += ms
        if ms > self.max_ms:
            self.max_ms = ms


class BeamProfile(msgspec.Struct):
    """Per-group phase accumulators; formatted once at group finish."""

    prefill_select: PhaseTimer = field(default_factory=PhaseTimer)
    spawn: PhaseTimer = field(default_factory=PhaseTimer)
    topk: PhaseTimer = field(default_factory=PhaseTimer)
    joint: PhaseTimer = field(default_factory=PhaseTimer)
    remap: PhaseTimer = field(default_factory=PhaseTimer)
    dag: PhaseTimer = field(default_factory=PhaseTimer)
    reclaim: PhaseTimer = field(default_factory=PhaseTimer)
    finish: PhaseTimer = field(default_factory=PhaseTimer)

    def add(self, phase: str, ms: float) -> None:
        getattr(self, phase).add(ms)

    def total_ms(self) -> float:
        """Sum over phases; may exceed real wall time where phases nest
        (finish wraps the reclaim inside free_member_rows)."""
        return sum(
            getattr(self, name).total_ms
            for name in (
                "prefill_select",
                "spawn",
                "topk",
                "joint",
                "remap",
                "dag",
                "reclaim",
                "finish",
            )
        )

    def format_summary(self) -> str:
        parts = []
        for name in (
            "prefill_select",
            "spawn",
            "topk",
            "joint",
            "remap",
            "dag",
            "reclaim",
            "finish",
        ):
            t = getattr(self, name)
            if t.count == 0:
                continue
            avg = t.total_ms / t.count
            parts.append(
                f"{name}={t.total_ms:.2f}ms"
                f"(n={t.count},avg={avg:.3f}ms,max={t.max_ms:.3f}ms)"
            )
        return " ".join(parts)


@contextmanager
def timed_phase(
    name: str,
    profile: Optional[BeamProfile],
) -> Iterator[None]:
    """Accumulate one phase's ms into ``profile``. A pure no-op when it is
    None (profiling disabled)."""
    if profile is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        ms = (time.perf_counter() - start) * 1000.0
        profile.add(name, ms)
