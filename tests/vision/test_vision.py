"""Optional suite: python -B -m unittest discover -s tests/vision -v."""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rl_platformer.agents.vision import VisionAgent, FrameHistory, OBSERVATION_SHAPE, torch, np
from rl_platformer.envs.vision import PixelEnv
from rl_platformer.ui.rendering import draw_agent_view, pygame
from rl_platformer.training.vision import create_run_directory, generalized_advantages, ppo_update


class VisionTests(unittest.TestCase):
    def test_observation_is_only_pixels_and_history_resets(self):
        env = PixelEnv(11)
        initial = env.reset()
        self.assertEqual(initial.shape, OBSERVATION_SHAPE)
        self.assertEqual(initial.dtype, np.uint8)
        np.testing.assert_array_equal(initial[0], initial[-1])
        observation, _, _, _ = env.step(5)
        np.testing.assert_array_equal(observation[:-1], initial[1:])
        self.assertFalse(np.array_equal(observation[-1], initial[-1]))
        reset = env.reset()
        np.testing.assert_array_equal(reset, initial)

    def test_numeric_observation_is_discarded(self):
        clean, poisoned = PixelEnv(3), PixelEnv(3)
        with patch.object(poisoned.simulator, "observe", return_value=(float("nan"),) * 28):
            for action in (2, 5, 5, 2, 12, 2):
                clean_pixels, clean_reward, _, _ = clean.step(action)
                poison_pixels, poison_reward, _, _ = poisoned.step(action)
                np.testing.assert_array_equal(clean_pixels, poison_pixels)
                self.assertEqual(clean_reward, poison_reward)

    def test_hidden_level_metadata_does_not_enter_frame(self):
        env = PixelEnv(13)
        before = env.history.observation()
        env.simulator.level = replace(env.simulator.level, name="SECRET", difficulty="SECRET", challenge="SECRET", max_steps=99999)
        env.simulator.steps = 100
        env.simulator.best_x = 999
        draw_agent_view(env.surface, env.simulator)
        np.testing.assert_array_equal(before, FrameHistory().reset(env.surface))

    def test_energy_is_visible_in_pixels(self):
        env = PixelEnv()
        before = env.history.observation()
        env.simulator.dash_energy = 0
        draw_agent_view(env.surface, env.simulator)
        self.assertFalse(np.array_equal(before[-1], FrameHistory().reset(env.surface)[-1]))

    def test_agent_rejects_state_vectors_and_roundtrips(self):
        model = VisionAgent(device="cpu")
        with self.assertRaises(ValueError):
            model.act_batch(np.zeros((1, 28)))
        pixels = PixelEnv().reset()[None]
        expected = model.act_batch(pixels, greedy=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            model.save(path)
            restored = VisionAgent.load(path, "cpu")
            for a, b in zip(expected, restored.act_batch(pixels, greedy=True)):
                np.testing.assert_array_equal(a, b)

    def test_time_limit_bootstrap_does_not_cross_reset(self):
        # First transition is truncated: reward already includes terminal V=10.
        rewards = np.array([[1 + .9 * 10], [100]], dtype=np.float32)
        values = np.zeros_like(rewards)
        advantages, returns = generalized_advantages(rewards, values, np.array([[True], [False]]), np.array([0.]), gamma=.9)
        self.assertAlmostEqual(advantages[0, 0], 10.)
        self.assertAlmostEqual(returns[1, 0], 100.)

    def test_existing_run_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = create_run_directory(Path(directory) / "run")
            checkpoint = path / "policy.pt"
            checkpoint.write_bytes(b"original")
            with self.assertRaises(FileExistsError):
                create_run_directory(path)
            self.assertEqual(checkpoint.read_bytes(), b"original")

    def test_ppo_update_changes_finite_parameters(self):
        model = VisionAgent(device="cpu")
        env = PixelEnv()
        pixels = np.stack([env.reset(), env.step(5)[0], env.step(5)[0], env.step(2)[0]])
        actions, logs, values = model.act_batch(pixels)
        before = model.network.actor.weight.detach().clone()
        optimizer = torch.optim.Adam(model.network.parameters(), lr=1e-4)
        result = ppo_update(model, optimizer, (pixels, actions, logs, np.array([1., -1., .5, -.5], dtype=np.float32), values + 1), epochs=1)
        self.assertTrue(all(np.isfinite(v) for v in result.values()))
        self.assertFalse(torch.equal(before, model.network.actor.weight))

    def test_cli_rejects_non_positive_time_penalty_before_any_work(self):
        from rl_platformer.training import vision
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value), \
                 patch("sys.argv", ["train_vision.py", "--time-penalty", value]), \
                 patch.object(vision, "create_run_directory") as create, \
                 patch("sys.stderr"), self.assertRaises(SystemExit) as raised:
                vision.main()
            self.assertEqual(raised.exception.code, 2)
            create.assert_not_called()

    def test_custom_test_suite_requires_a_separate_report_directory(self):
        from rl_platformer.training import vision
        run = Path("experiments/vision/cnn_v1")
        for size in (["--test-layouts", "100"], ["--fixed-starts", "20"]):
            with self.subTest(size=size), \
                 patch("sys.argv", ["train_vision.py", "--evaluate-only", str(run), *size]), \
                 patch.object(vision.VisionAgent, "load") as load, \
                 patch("sys.stderr"), self.assertRaises(SystemExit) as raised:
                vision.main()
            self.assertEqual(raised.exception.code, 2)
            load.assert_not_called()

    def test_vision_and_state_share_one_metric_definition(self):
        from rl_platformer.training import state, vision
        env = PixelEnv(1)
        env.reset(35.)
        for _ in range(5):
            env.step(2)
        case = {"family": "fixed", "stage": 1}
        self.assertEqual(vision.episode_row(env, case), state.episode_row(env.simulator, case))
        self.assertIs(vision.summarize, state.summarize)

    def test_playback_rejects_options_that_would_be_ignored(self):
        from rl_platformer.ui import play_vision
        for argv in (["--stage", "3"], ["--layout-seed", "12000123"], ["--family", "height", "--level", "13"]):
            with self.subTest(argv=argv), \
                 patch("sys.argv", ["play_vision.py", "--headless", *argv]), \
                 patch.object(play_vision.VisionAgent, "load") as load, \
                 patch("sys.stderr"), self.assertRaises(SystemExit) as raised:
                play_vision.main()
            self.assertEqual(raised.exception.code, 2)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
