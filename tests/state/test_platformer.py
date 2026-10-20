"""Physics, PPO boundary conditions, and saved-policy integration checks."""

from pathlib import Path
import random
import tempfile
import unittest

from rl_platformer.envs.platformer import ACTIONS, LEVELS, MAX_STEPS, OBSERVATION_SIZE, PLAYER_H, PlatformerEnv
from rl_platformer.agents.ppo import LinearPPO, PPO, load_policy, np
from rl_platformer.training.common import RolloutCollector
from rl_platformer.training.state import evaluate, evaluation_cases

from rl_platformer.paths import ROOT


class EnvironmentTests(unittest.TestCase):
    def test_all_levels_have_finite_observations(self):
        for level in range(1, len(LEVELS) + 1):
            env = PlatformerEnv(level)
            rng = random.Random(17)
            while not env.done:
                observation, _, _, info = env.step(rng.randrange(len(ACTIONS)))
                self.assertEqual(len(observation), OBSERVATION_SIZE)
                self.assertTrue(np.isfinite(observation).all())
                self.assertGreaterEqual(info["progress"], 0)
                self.assertLessEqual(info["progress"], 1)

    def test_timeout_and_death_are_distinct(self):
        env = PlatformerEnv()
        for _ in range(MAX_STEPS):
            _, _, _, info = env.step(0)
        self.assertTrue(info["truncated"])
        self.assertFalse(info["terminated"])
        with self.assertRaises(RuntimeError):
            env.step(0)
        env.reset()
        while not env.done:
            _, _, _, info = env.step(2)
        self.assertTrue(info["death"])
        self.assertTrue(info["terminated"])
        self.assertFalse(info["truncated"])

    def test_jump_lands_on_same_platform(self):
        env = PlatformerEnv()
        env.step(3)
        self.assertFalse(env.grounded)
        for _ in range(80):
            env.step(0)
            if env.grounded:
                break
        self.assertTrue(env.grounded)
        self.assertEqual(env.y + PLAYER_H, env.level.platforms[0].y)
        self.assertEqual(env.visited, {0})

    def test_intermediate_reset_has_consistent_state(self):
        env = PlatformerEnv(5)
        env.reset(platform_index=2)
        self.assertEqual(env.current_platform, 2)
        self.assertEqual(env.visited, {0, 1, 2})
        self.assertEqual(env.y + PLAYER_H, env.level.platforms[2].y)
        self.assertTrue(env.grounded)
        self.assertEqual(env.vx, 0)
        with self.assertRaises(ValueError):
            env.reset(start_x=-1)
        with self.assertRaises(ValueError):
            env.reset(platform_index=5)


class LearningTests(unittest.TestCase):
    def test_probability_normalization_and_roundtrip(self):
        model = PPO(42)
        obs = [PlatformerEnv(i).observe() for i in range(1, 6)]
        probabilities, values, _ = model.forward(obs)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        self.assertTrue((probabilities > 0).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            model.save(path, level=1)
            restored = load_policy(path)
            np.testing.assert_array_equal(restored.forward(obs)[0], probabilities)
            np.testing.assert_array_equal(restored.forward(obs)[1], values)

    def test_policy_update_increases_advantaged_action(self):
        model = PPO(42)
        obs = np.asarray([PlatformerEnv().observe()] * 6)
        probabilities, values, _ = model.forward(obs)
        actions = np.arange(6)
        advantages = np.asarray([-1., -1., 5., -1., -1., -1.])
        model.update(obs, actions, probabilities[np.arange(6), actions], advantages,
                     values.copy(), epochs=1, entropy_coef=0)
        self.assertGreater(model.forward(obs)[0][0, 2], probabilities[0, 2])

    def test_gae_bootstraps_timeouts_but_not_deaths_or_resets(self):
        class OneStepEnv:
            def __init__(self, truncated):
                self.truncated = truncated

            def reset(self, start_x):
                return (0.,) * OBSERVATION_SIZE

            def step(self, action):
                return (0.,) * OBSERVATION_SIZE, 1., True, {"truncated": self.truncated, "success": False}

        class ConstantValueModel:
            def act_batch(self, observations):
                n = len(observations)
                return np.zeros(n, dtype=int), np.ones(n), np.full(n, 2.)

            def forward(self, observations):
                return None, np.full(len(observations), 2.), None

        for truncated, expected_return in [(False, 1.), (True, 2.98)]:
            collector = RolloutCollector(1, 1, 42)
            collector.envs = [OneStepEnv(truncated)]
            collector.observations[:] = 0
            samples, completed, _ = collector.collect(ConstantValueModel(), steps=3)
            np.testing.assert_allclose(samples[4], expected_return)
            np.testing.assert_allclose(samples[3], expected_return - 2.)
            self.assertEqual(completed, 3)

    def test_evaluation_uses_saved_start_stream(self):
        class RandomPolicy:
            def __init__(self):
                self.rng = random.Random(124)

            def act_batch(self, observations, greedy):
                return np.asarray([self.rng.randrange(24) for _ in observations]), None, None

        cases = evaluation_cases("test", 1)[:4]
        first, rows = evaluate(RandomPolicy(), cases)
        second, repeated = evaluate(RandomPolicy(), cases)
        self.assertEqual(first, second)
        self.assertEqual(rows, repeated)
        self.assertEqual([r["start_x"] for r in rows], [c["start_x"] for c in cases])

    def test_original_linear_policy_remains_playable(self):
        # The optional legacy checkpoint is no longer shipped. Exercise its
        # actual JSON format with a deterministic fixture instead.
        import json
        legacy = LinearPPO(42)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps({"weights": legacy.weights, "value_weights": legacy.value_weights}), encoding="utf-8")
            restored = load_policy(path)
            env = PlatformerEnv()
            while not env.done:
                self.assertEqual(restored.probabilities(env.observe()), legacy.probabilities(env.observe()))
                env.step(restored.action(env.observe(), greedy=True)[0])


if __name__ == "__main__":
    unittest.main()
