from rl_platformer.paths import BASELINE_STATE_POLICY
"""Player/PPO control semantics, dash energy, swept collisions and speed rewards."""

from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from rl_platformer.envs.movement import (DASH_DIRECTIONS, DASH_FRAMES, MOVEMENT_ACTIONS, MOVEMENT_OBSERVATION_SIZE, MOVEMENT_VERSION,
                      MovementEnv, decode_action, movement_action)
from rl_platformer.envs.platformer import LEVELS, PLAYER_H, Level, Platform
from rl_platformer.agents.ppo import PPO, np
from rl_platformer.training.state import MovementCollector, evaluation_cases, expand_policy, geometry_key, selection_score


class MovementTests(unittest.TestCase):
    def test_hold_duration_controls_jump_height_and_no_bunny_hop(self):
        heights = []
        for held in (1, 5, 12, 25):
            env = MovementEnv()
            top = initial = env.y
            for frame in range(80):
                env.step(movement_action(jump=frame < held))
                top = min(top, env.y)
                if env.grounded and frame > 0:
                    break
            heights.append(initial - top)
        self.assertTrue(all(a < b for a, b in zip(heights, heights[1:])))
        self.assertLess(heights[0], 30)
        self.assertGreater(heights[-1], 130)
        env = MovementEnv()
        for _ in range(100):
            env.step(3)
        self.assertEqual(env.jump_count, 1)
        env.step(0)
        env.step(3)
        self.assertEqual(env.jump_count, 2)

    def test_default_facing_and_all_eight_dash_directions(self):
        for dx, dy in DASH_DIRECTIONS[1:]:
            env = MovementEnv()
            env.y = 200
            env.grounded = False
            info = env.step(movement_action(dx, dy, dash=True))[3]
            self.assertTrue(info["dash_started"])
            expected_x, expected_y = dx * 2.4 / np.hypot(dx, dy), dy * 2.4 / np.hypot(dx, dy)
            self.assertAlmostEqual(env.vx, expected_x)
            self.assertAlmostEqual(env.vy, expected_y + 0.48)
            self.assertEqual(env.dash_energy, 0)
        env = MovementEnv()
        env.step(1)
        env.step(movement_action(dash=True))
        self.assertLess(env.vx, -2)
        self.assertEqual(env.facing, -1)
        self.assertEqual(decode_action(movement_action(1, -1, jump=True, dash=True), -1)[1:3], (True, True))

    def test_dash_replaces_incoming_momentum_once(self):
        for dx, dy in DASH_DIRECTIONS[1:]:
            for vx, vy in ((5.6, -11.8), (-13., 12.)):
                env = MovementEnv()
                env.x, env.y, env.grounded = 300., 200., False
                env.vx, env.vy = vx, vy
                action = movement_action(dx, dy, dash=True)
                env.step(action)
                self.assertAlmostEqual(env.vx, dx * 2.4 / np.hypot(dx, dy))
                self.assertAlmostEqual(env.vy, dy * 2.4 / np.hypot(dx, dy) + 0.48)
                first_vx, first_vy = env.vx, env.vy
                self.assertFalse(env.step(action)[3]["dash_started"])
                self.assertAlmostEqual(env.vx, first_vx + dx * 2.4 / np.hypot(dx, dy))
                self.assertAlmostEqual(env.vy, first_vy + dy * 2.4 / np.hypot(dx, dy) + 0.48)
        env = MovementEnv()
        env.step(movement_action(1, -1, jump=True, dash=True))
        self.assertAlmostEqual(env.vy, -2.4 / np.sqrt(2) + 0.48)
        old = MovementEnv(movement_version="agile-v1")
        old.x, old.y, old.grounded, old.vx, old.vy = 300., 200., False, 5.6, -5.
        old.step(movement_action(1, 0, dash=True))
        self.assertAlmostEqual(old.vx, 8.)
        self.assertAlmostEqual(old.vy, -4.52)
        env = MovementEnv()
        env.x, env.y, env.grounded = 300., 200., False
        env.vx, env.vy, env.dash_energy, env.air_dash_used = 7., -3., 0., True
        self.assertFalse(env.step(movement_action(1, 0, dash=True))[3]["dash_started"])
        self.assertAlmostEqual(env.vx, 7. * 0.94)
        self.assertAlmostEqual(env.vy, -3. + 0.48)

    def test_training_dash_search_preserves_start_and_replays_faster_route(self):
        from rl_platformer.training.bootstrap_dash import improve_route
        model = PPO.load(BASELINE_STATE_POLICY)
        start = MovementEnv(1)
        trace, result = improve_route(model, start, rounds=2)
        self.assertEqual(start.steps, 0)
        replay = MovementEnv(1)
        for observation, action in trace:
            self.assertEqual(observation, replay.observe())
            replay.step(action)
        self.assertTrue(replay.success)
        self.assertEqual(replay.steps, result["steps"])
        self.assertEqual(replay.dash_count, result["dashes"])
        self.assertGreater(replay.dash_count, 0)
        self.assertLess(replay.steps, result["baseline"]["steps"])

    def test_uncapped_dash_teacher_and_wider_actor(self):
        from rl_platformer.training.bootstrap_dash import greedy_route, widen_actor
        model = PPO.load(BASELINE_STATE_POLICY)
        model.metadata["action_mask"] = "dash-ready-v1"
        initial = MovementEnv(1)
        wider = widen_actor(model, 64, 42)
        np.testing.assert_allclose(model.forward([initial.observe()])[0], wider.forward([initial.observe()])[0], atol=1e-12)
        trace, report = greedy_route(model, initial)
        for observation, action in trace:
            self.assertEqual(observation, initial.observe())
            initial.step(action)
        self.assertTrue(initial.success)
        self.assertGreater(initial.dash_count, 3)
        self.assertLess(initial.steps, report["baseline"]["steps"])

    def test_air_dash_once_landing_refills_and_held_key_cannot_retrigger(self):
        env = MovementEnv()
        env.step(movement_action(jump=True))
        env.step(movement_action(0, -1, dash=True))
        for _ in range(DASH_FRAMES + 1):
            env.step(0)
        self.assertFalse(env.grounded)
        self.assertFalse(env.step(movement_action(dash=True))[3]["dash_started"])
        self.assertEqual(env.dash_count, 1)
        while not env.grounded:
            env.step(movement_action(0, -1, dash=True))
        self.assertEqual(env.dash_energy, 1)
        self.assertFalse(env.air_dash_used)
        self.assertFalse(env.step(movement_action(0, -1, dash=True))[3]["dash_started"])
        env.step(0)
        self.assertTrue(env.step(movement_action(0, -1, dash=True))[3]["dash_started"])
        env.reset()
        self.assertEqual(env.dash_energy, 1)
        self.assertFalse(env.dash_was_held)
        self.assertFalse(env.jump_was_held)
        self.assertEqual(env.dash_count, 0)

    def test_leaving_ground_during_dash_does_not_grant_second_air_dash(self):
        env = MovementEnv()
        env.reset(start_x=180)
        env.step(movement_action(1, 0, dash=True))
        for _ in range(DASH_FRAMES + 1):
            env.step(2)
        self.assertFalse(env.grounded)
        self.assertEqual(env.dash_energy, 0)
        self.assertFalse(env.step(movement_action(1, -1, dash=True))[3]["dash_started"])

    def test_swept_landing_at_crossing_instead_of_frame_endpoint(self):
        level = Level("collision", (Platform(0, 450, 100), Platform(130, 350, 24), Platform(300, 350, 300)),
                      35, 550, "test", "test")
        env = MovementEnv(level)
        env.x, env.y = 149., 350 - PLAYER_H - 1
        env.vx, env.vy, env.grounded = 13., 10., False
        env.dash_remaining, env.dash_direction = 1, (1., 0.)
        info = env.step(2)[3]
        self.assertEqual(info["landed_platform"], 1)
        self.assertTrue(env.grounded)
        self.assertLess(env.x, 154)
        self.assertEqual(env.dash_energy, 1)

    def test_time_reward_favors_fast_success_not_fast_death(self):
        rewards = []
        for elapsed in (10, 100):
            env = MovementEnv()
            env.reset(platform_index=2, start_x=830)
            env.steps = elapsed
            _, reward, done, info = env.step(0)
            self.assertTrue(done)
            self.assertTrue(env.success)
            rewards.append(reward)
        self.assertGreater(rewards[0], rewards[1])
        env = MovementEnv()
        env.y = env.level.world_height + 50
        _, reward, _, info = env.step(0)
        self.assertLess(reward, 0)
        self.assertEqual(info["reward_components"]["finish_time"], 0)
        faster_failure = {"completion_rate": 0.9, "penalized_time": 0.1, "mean_progress": 1.}
        slower_success = {"completion_rate": 1., "penalized_time": 0.5, "mean_progress": 1.}
        self.assertGreater(selection_score(slower_success), selection_score(faster_failure))
        fast_reliable = {"completion_rate": 0.99, "penalized_time": 0.3, "mean_progress": 1.}
        self.assertGreater(selection_score(fast_reliable, 0.98), selection_score(slower_success, 0.98))
        self.assertGreater(selection_score(slower_success, 0.98), selection_score(faster_failure, 0.98))

    def test_keyboard_up_aims_dash_without_jumping(self):
        from rl_platformer.ui.game import keyboard_movement_action, pygame
        keys = defaultdict(bool, {pygame.K_UP: True, pygame.K_LSHIFT: True})
        self.assertEqual(keyboard_movement_action(keys), movement_action(0, -1, dash=True))
        keys[pygame.K_RIGHT] = keys[pygame.K_SPACE] = True
        self.assertEqual(keyboard_movement_action(keys), movement_action(1, -1, jump=True, dash=True))

    def test_expanded_policy_rollout_and_save_load(self):
        model = expand_policy(PPO(42), 42)
        self.assertEqual(model.action_count, len(MOVEMENT_ACTIONS))
        collector = MovementCollector(2, 42)
        samples, _, _ = collector.collect(model, 32)
        self.assertEqual(samples[0].shape, (64, MOVEMENT_OBSERVATION_SIZE))
        self.assertTrue(np.isfinite(samples[0]).all())
        diagnostics = model.update(*samples, epochs=1, value_coef=0.1)
        self.assertTrue(all(np.isfinite(v) for v in diagnostics.values()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            model.save(path)
            restored = PPO.load(path)
            np.testing.assert_array_equal(model.forward(samples[0])[0], restored.forward(samples[0])[0])

    def test_validation_test_seed_and_geometry_separation(self):
        validation = evaluation_cases("validation", 3)
        testing = evaluation_cases("test", 3)
        # Fixed maps intentionally share geometry; their start streams differ.
        for key in ("layout_seed", "layout_hash"):
            self.assertFalse({r[key] for r in validation if r["family"] != "fixed"} &
                             {r[key] for r in testing if r["family"] != "fixed"})
        level = MovementEnv().level
        self.assertEqual(geometry_key(level), geometry_key(replace(level, name="different seed label")))

    def test_dash_mask_blocks_only_unavailable_dash_actions(self):
        model = expand_policy(PPO(42), 42)
        model.metadata["action_mask"] = "dash-ready-v1"
        env = MovementEnv()
        self.assertGreater(model.forward([env.observe()])[0][0, 6:].sum(), 0)
        env.step(movement_action(0, -1, dash=True))
        probabilities = model.forward([env.observe()])[0][0]
        np.testing.assert_array_equal(probabilities[6:], 0)
        self.assertAlmostEqual(probabilities[:6].sum(), 1.)

    def test_saved_default_policy_completes_all_fixed_levels(self):
        from rl_platformer.ui.game import DEFAULT_MOVEMENT_POLICY
        model = PPO.load(DEFAULT_MOVEMENT_POLICY)
        self.assertEqual(model.metadata["movement_version"], MOVEMENT_VERSION)
        dashes, releases = 0, 0
        for number in range(1, len(LEVELS) + 1):
            with self.subTest(level=number):
                env = MovementEnv(number, movement_version=model.metadata["movement_version"])
                while not env.done:
                    env.step(model.action(env.observe(), greedy=True)[0])
                self.assertTrue(env.success)
                self.assertGreater(env.dash_count, 0)
                dashes += env.dash_count
                releases += env.jump_release_count
        self.assertGreater(dashes, 0)
        self.assertGreater(releases, 0)


if __name__ == "__main__":
    unittest.main()
