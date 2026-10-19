"""Unit tests for beam search phase timing (profile.py + BeamGroup wiring).

Guarded contracts:
- [BEAMPROF] log-line format (user-facing, parsed by external tooling).
- Phase-name typos raise instead of silently recording nothing.
- The default (profiling off) has no accumulator and timed_phase no-ops.
- SGLANG_LOG_BEAM_PROFILE defaults off and reads through the Envs API.
"""

import unittest

from sglang.srt.beam_search.beam_group import BeamGroup
from sglang.srt.beam_search.profile import BeamProfile, timed_phase
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PHASES = (
    "prefill_select",
    "spawn",
    "topk",
    "joint",
    "remap",
    "dag",
    "reclaim",
    "finish",
)


class TestBeamProfile(CustomTestCase):
    def test_accumulate_and_summary(self):
        profile = BeamProfile()
        profile.add("topk", 1.0)
        profile.add("topk", 3.0)
        profile.add("reclaim", 5.0)
        self.assertEqual(profile.total_ms(), 9.0)
        summary = profile.format_summary()
        self.assertIn("topk=4.00ms(n=2,avg=2.000ms,max=3.000ms)", summary)
        self.assertIn("reclaim=5.00ms(n=1,avg=5.000ms,max=5.000ms)", summary)
        # Zero-count phases are omitted.
        for name in PHASES:
            if name not in ("topk", "reclaim"):
                self.assertNotIn(f"{name}=", summary)

    def test_unknown_phase_raises(self):
        # A typo'd phase name must not silently record nowhere.
        profile = BeamProfile()
        with self.assertRaises(AttributeError):
            profile.add("nope", 1.0)


class TestTimedPhase(CustomTestCase):
    def test_accumulates_ms(self):
        profile = BeamProfile()
        with timed_phase("joint", profile):
            pass
        self.assertEqual(profile.joint.count, 1)
        self.assertGreater(profile.joint.total_ms, 0.0)

    def test_none_profile_is_noop(self):
        with timed_phase("topk", None):
            pass
        # Nothing to assert beyond not raising; the profile was None.


class TestBeamGroupProfileWiring(CustomTestCase):
    def test_default_profile_none(self):
        # The default has no accumulator: timing must stay fully off.
        group = BeamGroup(beam_width=2, max_new_tokens=5)
        self.assertIsNone(group.profile)

    def test_records_into_group_profile(self):
        group = BeamGroup(beam_width=2, max_new_tokens=5)
        group.profile = BeamProfile()
        with timed_phase("dag", group.profile):
            pass
        self.assertEqual(group.profile.dag.count, 1)


class TestBeamProfileEnvGate(CustomTestCase):
    def test_env_var_default_off_and_override(self):
        self.assertFalse(envs.SGLANG_LOG_BEAM_PROFILE.get())
        with envs.SGLANG_LOG_BEAM_PROFILE.override(True):
            self.assertTrue(envs.SGLANG_LOG_BEAM_PROFILE.get())
        self.assertFalse(envs.SGLANG_LOG_BEAM_PROFILE.get())


if __name__ == "__main__":
    unittest.main()
