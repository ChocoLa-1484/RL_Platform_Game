"""Saved artifacts remain usable after repository reorganization."""

from pathlib import Path
import tempfile
import unittest

from rl_platformer.paths import ROOT, DEFAULT_STATE_POLICY, BASELINE_STATE_POLICY, resolve_saved_path
from rl_platformer.ui.game import policy_path_for_level


class ProjectLayoutTests(unittest.TestCase):
    def test_historical_absolute_and_relative_model_paths_resolve(self):
        self.assertEqual(resolve_saved_path("experiments/dash_speedrun/policy.json"), DEFAULT_STATE_POLICY)
        self.assertEqual(resolve_saved_path(r"D:\old-checkout\experiments\dash_speedrun\policy.json"), DEFAULT_STATE_POLICY)
        self.assertEqual(resolve_saved_path("experiments/movement/policy.json"), BASELINE_STATE_POLICY)
        self.assertTrue(DEFAULT_STATE_POLICY.is_file())
        self.assertTrue(BASELINE_STATE_POLICY.is_file())

    def test_existing_custom_path_takes_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text("{}", encoding="utf-8")
            self.assertEqual(resolve_saved_path(path), path)
            self.assertEqual(policy_path_for_level(13, custom_path=path), path)

    def test_all_levels_use_the_shared_current_policy(self):
        for level in range(1, 14):
            self.assertEqual(policy_path_for_level(level), DEFAULT_STATE_POLICY)
        self.assertEqual(policy_path_for_level(1, ROOT / "custom"), ROOT / "custom" / "policy.json")


if __name__ == "__main__":
    unittest.main()
