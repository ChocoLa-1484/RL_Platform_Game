import unittest

from rl_platformer.ui.game import FPS, next_level_after_success, record_clear, run_summary


class LevelAdvanceTest(unittest.TestCase):
    def test_next_level_stops_after_the_last_one(self):
        self.assertEqual(next_level_after_success(1, 13), 2)
        self.assertEqual(next_level_after_success(12, 13), 13)
        self.assertIsNone(next_level_after_success(13, 13))

    def test_retrying_a_cleared_level_replaces_its_time(self):
        results = record_clear([], 2, 1.5, 4)
        results = record_clear(results, 1, 1.0, 3)
        results = record_clear(results, 2, 1.25, 5)
        self.assertEqual(results, [(1, 1.0, 3), (2, 1.25, 5)])

    def test_summary_names_a_full_run_and_counts_frames(self):
        full = [(level, 1.0, 2) for level in range(1, 14)]
        title, lines, total = run_summary(full, 13)
        self.assertEqual(title, "ALL 13 LEVELS CLEARED")
        self.assertEqual(len(lines), 13)
        self.assertEqual(total, f"Total 13.00s ({13 * FPS} frames), 26 dashes")
        title, _, _ = run_summary(full[5:], 13)
        self.assertEqual(title, "8 OF 13 LEVELS CLEARED")


if __name__ == "__main__":
    unittest.main()
