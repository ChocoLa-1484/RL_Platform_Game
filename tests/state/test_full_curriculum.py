"""Regression tests for scrolling worlds and held-out sequential layouts."""

from dataclasses import replace
import unittest

from rl_platformer.levels.full_curriculum import (STAGES, generate_level,
                             layout_record, layout_seed)
from rl_platformer.envs.platformer import LEVELS, MAX_STEPS, PLAYER_W, PlatformerEnv, WIDTH
from rl_platformer.levels.procedural import validate_layout
from rl_platformer.training.state import MovementCollector, evaluate, evaluation_cases, layout_seed as movement_seed
from rl_platformer.agents.ppo import PPO


class FullCurriculumTests(unittest.TestCase):
    def test_fixed_routes_are_valid_and_increasing_in_length(self):
        widths = []
        for level in LEVELS[5:10]:
            validate_layout(level)
            widths.append(level.world_width)
            self.assertGreater(level.max_steps, MAX_STEPS)
        self.assertEqual(widths, sorted(set(widths)))
        self.assertEqual(len(LEVELS[9].platforms), 12)
        self.assertEqual(min(p.width for p in LEVELS[9].platforms), 40)

    def test_world_boundary_goal_and_observation_beyond_screen(self):
        env = PlatformerEnv(10)
        env.reset(platform_index=11, start_x=2420)
        self.assertGreater(env.x, WIDTH)
        self.assertAlmostEqual(env.observe()[0], (2420 + PLAYER_W / 2) / 2600)
        while not env.done:
            env.step(2)
        self.assertTrue(env.success)
        self.assertEqual(env.current_platform, 11)
        self.assertLessEqual(env.x, env.level.world_width - PLAYER_W)

    def test_level_specific_timeout(self):
        env = PlatformerEnv(6)
        for _ in range(MAX_STEPS):
            env.step(0)
        self.assertFalse(env.done)
        for _ in range(env.level.max_steps - MAX_STEPS):
            _, _, _, info = env.step(0)
        self.assertTrue(info["truncated"])
        self.assertFalse(info["terminated"])

    def test_scrolling_and_keyboard(self):
        from rl_platformer.ui.game import camera_offset, level_from_key, pygame
        env = PlatformerEnv(1)
        self.assertEqual(camera_offset(env), 0)
        env = PlatformerEnv(10)
        env.reset(platform_index=11)
        self.assertEqual(camera_offset(env), 2600 - WIDTH)
        self.assertEqual(level_from_key(pygame.K_0, 10), 10)
        self.assertEqual(level_from_key(pygame.K_9, 10), 9)
        self.assertIsNone(level_from_key(pygame.K_0, 4))
        self.assertIsNone(level_from_key(pygame.K_COLON, 10))

    def test_generator_geometry_reproducibility_and_split_separation(self):
        hashes = {}
        for split in ("train", "validation", "test"):
            hashes[split] = set()
            for stage in STAGES:
                for index in range(100):
                    seed = layout_seed(stage, split, index)
                    level = generate_level(stage, seed)
                    validate_layout(level)
                    self.assertEqual(level, generate_level(stage, seed))
                    hashes[split].add(layout_record(stage, seed)["layout_hash"])
                    if stage == 4:
                        self.assertTrue(8 <= len(level.platforms) <= 12)
                        self.assertGreater(level.world_width, WIDTH)
        self.assertFalse(hashes["train"] & hashes["validation"])
        self.assertFalse(hashes["train"] & hashes["test"])
        self.assertFalse(hashes["test"] & hashes["validation"])

    def test_collector_training_only_and_evaluation_from_first_platform(self):
        collector = MovementCollector(8, 42, phase=3, platform_start_prob=0.8)
        starts = []
        for _ in range(100):
            collector.reset(collector.envs[0])
            starts.append(collector.envs[0].current_platform)
        self.assertTrue(any(index > 0 for index in starts))
        self.assertIn(0, starts)
        self.assertEqual({s for f, s, _ in collector.seen_layouts if f == "full"}, set(STAGES))
        for family, stage, seed in collector.seen_layouts:
            if family != "fixed":
                self.assertTrue(movement_seed(family, stage, "train", 0) <= seed < movement_seed(family, stage, "train", 1000))
        cases = [c for c in evaluation_cases("test", 5) if c["family"] == "full" and c["stage"] == 4]
        _, rows = evaluate(PPO(42, observation_size=28, action_count=24), cases)
        self.assertEqual([r["start_x"] for r in rows], [c["start_x"] for c in cases])
        self.assertTrue(all(25 <= r["start_x"] <= 50 for r in rows))

    def test_old_world_defaults_remain_compatible(self):
        for level in LEVELS[:5]:
            self.assertEqual(level.world_width, WIDTH)
            self.assertEqual(level.max_steps, MAX_STEPS)
        old = PlatformerEnv(1)
        explicit = PlatformerEnv(replace(LEVELS[0], world_width=WIDTH, max_steps=MAX_STEPS))
        self.assertEqual(old.observe(), explicit.observe())


if __name__ == "__main__":
    unittest.main()
