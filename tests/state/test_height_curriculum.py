"""Climbing geometry, rewards, scrolling and PPO stability regressions."""

import unittest

from rl_platformer.levels import height_curriculum as terrain
from rl_platformer.envs.platformer import HEIGHT, LEVELS, PLAYER_H, PlatformerEnv
from rl_platformer.agents.ppo import PPO, np
from rl_platformer.levels.procedural import validate_layout
from rl_platformer.training.state import MovementCollector, evaluate, evaluation_cases, layout_seed
from rl_platformer.envs.movement import MovementEnv
from rl_platformer.paths import DEFAULT_STATE_POLICY
from rl_platformer.training.common import RolloutCollector


class HeightTests(unittest.TestCase):
    def test_fixed_climbs_and_generated_splits(self):
        for level, ascent in zip(LEVELS[10:], (520, 810, 1240)):
            validate_layout(level)
            self.assertEqual(level.platforms[0].y - level.platforms[-1].y, ascent)
            self.assertGreater(level.world_height, HEIGHT)
        hashes = []
        for split in ("train", "validation", "test"):
            group = set()
            for stage in terrain.STAGES:
                for index in range(100):
                    seed = terrain.layout_seed(stage, split, index)
                    level = terrain.generate_level(stage, seed)
                    validate_layout(level)
                    self.assertEqual(level, terrain.generate_level(stage, seed))
                    self.assertTrue(all(b.y < a.y for a, b in zip(level.platforms, level.platforms[1:])))
                    group.add(terrain.layout_record(stage, seed)["layout_hash"])
            hashes.append(group)
        for i, left in enumerate(hashes):
            for right in hashes[i + 1:]:
                self.assertFalse(left & right)

    def test_world_height_controls_death_not_screen_height(self):
        env = PlatformerEnv(13)
        self.assertGreater(env.y, HEIGHT)
        _, _, done, _ = env.step(0)
        self.assertFalse(done)
        self.assertTrue(env.grounded)
        self.assertAlmostEqual(env.observe()[1], (env.y + PLAYER_H) / env.level.world_height)
        env.y = env.level.world_height + 41
        _, _, done, info = env.step(0)
        self.assertTrue(done)
        self.assertTrue(info["terminated"])
        self.assertTrue(info["death"])

    def test_height_reward_requires_new_highest_landing(self):
        for mode in ("shaped", "sparse"):
            env = PlatformerEnv(11, reward_mode=mode)
            target = env.level.platforms[1]
            for repeat in range(3):
                env.x, env.y = target.x + 5, target.y - PLAYER_H - 1
                env.vx, env.vy, env.grounded = 0, 1, False
                components = env.step(0)[3]["reward_components"]
                self.assertAlmostEqual(components["height"], 1.5 if repeat == 0 and mode == "shaped" else 0)
            env.reset(platform_index=3)
            self.assertEqual(env.step(0)[3]["reward_components"]["height"], 0)
        env = PlatformerEnv(11)
        for _ in range(45):
            self.assertEqual(env.step(3)[3]["reward_components"]["height"], 0)

    def test_camera_tracks_ascent_and_all_levels_are_selectable(self):
        from rl_platformer.ui.game import camera_vertical_offset, level_from_key, pygame, policy_path_for_level
        env = PlatformerEnv(13)
        start = camera_vertical_offset(env)
        self.assertEqual(start, env.level.world_height - HEIGHT)
        env.reset(platform_index=len(env.level.platforms) - 1)
        self.assertTrue(-180 <= camera_vertical_offset(env) < 0)
        self.assertGreaterEqual(env.y - camera_vertical_offset(env), 200)
        self.assertEqual(camera_vertical_offset(PlatformerEnv(1)), 0)
        self.assertEqual(level_from_key(pygame.K_RIGHTBRACKET, 13, 10), 11)
        self.assertEqual(level_from_key(pygame.K_RIGHTBRACKET, 13, 13), 1)
        self.assertEqual(level_from_key(pygame.K_LEFTBRACKET, 13, 1), 13)
        self.assertEqual(policy_path_for_level(13), DEFAULT_STATE_POLICY)

    def test_collector_uses_only_train_pools_and_evaluation_starts_at_bottom(self):
        collector = MovementCollector(2, 42, phase=3, platform_start_prob=0.5)
        starts = []
        for _ in range(100):
            collector.reset(collector.envs[0])
            starts.append(collector.envs[0].current_platform)
        self.assertIn(0, starts)
        self.assertTrue(any(i > 0 for i in starts))
        self.assertEqual({f for f, _, _ in collector.seen_layouts}, {"height", "full", "fixed"})
        for family, stage, seed in collector.seen_layouts:
            if family != "fixed":
                self.assertTrue(layout_seed(family, stage, "train", 0) <= seed < layout_seed(family, stage, "train", 1000))
        cases = [c for c in evaluation_cases("validation", 3) if c["family"] == "height" and c["stage"] == 3]
        _, rows = evaluate(PPO(42, observation_size=28, action_count=24), cases)
        self.assertTrue(all(25 <= r["start_x"] <= 50 for r in rows))
        self.assertEqual([r["layout_seed"] for r in rows], [layout_seed("height", 3, "validation", i) for i in range(3)])

    def test_kl_limit_stops_repeated_epochs_and_validates_threshold(self):
        model = PPO(42)
        collector = RolloutCollector(11, 2, 42)
        samples, _, _ = collector.collect(model, 32)
        diagnostics = model.update(*samples, epochs=8, target_kl=1e-12)
        self.assertEqual(diagnostics["epochs_run"], 1)
        self.assertGreater(diagnostics["rollout_kl"], 1e-12)
        self.assertTrue(all(np.isfinite(v) for v in diagnostics.values()))
        for invalid in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                model.update(*samples, target_kl=invalid)

    def test_saved_climbing_policy_finishes_new_fixed_routes(self):
        model = PPO.load(DEFAULT_STATE_POLICY)
        for level in range(11, 14):
            with self.subTest(level=level):
                env = MovementEnv(level)
                while not env.done:
                    env.step(model.action(env.observe(), greedy=True)[0])
                self.assertTrue(env.success)


if __name__ == "__main__":
    unittest.main()
