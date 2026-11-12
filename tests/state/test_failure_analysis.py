"""Failure analysis replays the saved state policy and explains known deaths."""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import unittest
from unittest.mock import patch

from rl_platformer.agents.ppo import PPO
from rl_platformer.analysis.failures import analyze, classify_death, climbs, replay_state
from rl_platformer.paths import DEFAULT_STATE_POLICY
from rl_platformer.training.state import evaluation_cases


def test_case(seed):
    return next(c for c in evaluation_cases("test", 100, 20) if c["layout_seed"] == seed)


class FailureAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = PPO.load(DEFAULT_STATE_POLICY)

    def test_climb_missed_by_a_few_pixels_after_horizontal_dash(self):
        (sim, frames), = replay_state(self.model, [test_case(14300011)])
        self.assertTrue(sim.deaths)
        death = classify_death(sim, frames)
        self.assertEqual((death["takeoff"], death["outcome"]), ("jump", "under_ledge"))
        missed = [c for c in climbs(sim, frames) if not c["landed"]]
        self.assertEqual(len(missed), 1)
        self.assertTrue(missed[0]["horizontal_dash"])
        self.assertTrue(-5 < missed[0]["clearance_px"] < 0)

    def test_ground_dash_overshoots_a_lower_narrow_platform(self):
        (sim, frames), = replay_state(self.model, [test_case(11400002)])
        death = classify_death(sim, frames)
        self.assertEqual((death["takeoff"], death["outcome"], death["dashes"]), ("ground_dash", "overshoot", "right"))
        self.assertGreater(death["horizontal_miss_px"], 0)
        self.assertFalse(death["braked_in_air"])

    def test_successful_episode_yields_no_failure_row(self):
        cases = [test_case(11100000)]
        failures, climb_rows, trajectories, summary = analyze(replay_state(self.model, cases), cases)
        self.assertEqual((failures, trajectories, summary["deaths"]), ([], {}, 0))

    def test_cli_needs_exactly_one_mode_and_a_new_directory(self):
        from rl_platformer.analysis import run
        for argv in (["--output-dir", "x"], ["--agent", "state", "--figures", "a", "b", "--output-dir", "x"],
                     ["--agent", "state", "--output-dir", "experiments"]):
            with self.subTest(argv=argv), patch("sys.argv", ["analyze_failures.py", *argv]), \
                 patch("sys.stderr"), self.assertRaises(SystemExit) as raised:
                run.main()
            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
