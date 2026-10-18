"""Experimental validity, action ablations, and reward-hacking regressions."""

import random
import tempfile
import unittest
from pathlib import Path

from platformer import MAX_STEPS, PLAYER_H, PlatformerEnv
from ppo import PPO, load_policy, np
from procedural import generate_level, layout_record, layout_seed, validate_layout
from run_experiments import ProceduralCollector, evaluate_procedural, evaluation_cases


class ProceduralTests(unittest.TestCase):
    def test_seeded_geometry_and_disjoint_splits(self):
        hashes = {}
        seeds = {}
        for split in ("train", "validation", "test"):
            hashes[split], seeds[split] = set(), set()
            for stage in (1, 2, 3):
                for index in range(100):
                    seed = layout_seed(stage, split, index)
                    level = generate_level(stage, seed)
                    validate_layout(level)
                    self.assertEqual(level, generate_level(stage, seed))
                    hashes[split].add(layout_record(stage, seed)["layout_hash"])
                    seeds[split].add(seed)
                    if stage == 1:
                        self.assertEqual({p.y for p in level.platforms}, {450})
                    if stage == 3:
                        self.assertLessEqual(min(p.width for p in level.platforms[1:-1]), 75)
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            self.assertFalse(seeds[left] & seeds[right])
            self.assertFalse(hashes[left] & hashes[right])

    def test_collector_uses_training_pool_and_rehearsal(self):
        collector = ProceduralCollector(3, 8, 42, "ppo_shaped")
        for _ in range(100):
            collector.reset(collector.envs[0])
        self.assertEqual({stage for stage, _ in collector.seen_layouts}, {1, 2, 3})
        self.assertTrue(all(100_000 <= seed < 301_000 for _, seed in collector.seen_layouts))
        self.assertTrue(all(env.current_platform == 0 for env in collector.envs))

    def test_auto_jump_matches_explicit_jump_physics(self):
        manual = PlatformerEnv(3)
        automatic = PlatformerEnv(3, auto_jump=True)
        rng = random.Random(32)
        while not manual.done:
            direction = rng.randrange(3)
            expected = manual.step(direction + 3)
            actual = automatic.step(direction)
            self.assertEqual(actual, expected)
        with self.assertRaises(ValueError):
            PlatformerEnv(auto_jump=True).step(3)

    def test_sparse_rewards_are_terminal_only(self):
        shaped = PlatformerEnv(1)
        sparse = PlatformerEnv(1, reward_mode="sparse")
        while not shaped.done:
            _, _, _, shaped_info = shaped.step(2)
            _, reward, _, info = sparse.step(2)
            self.assertEqual(shaped.x, sparse.x)
            self.assertEqual(shaped.y, sparse.y)
            self.assertEqual(shaped_info["death"], info["death"])
            self.assertEqual(reward, -5 if info["death"] else 0)
        sparse.reset()
        for _ in range(MAX_STEPS):
            self.assertEqual(sparse.step(0)[1], 0)

    def test_platform_and_checkpoint_rewards_cannot_be_farmed(self):
        env = PlatformerEnv(1)
        target = env.level.platforms[1]
        # Repeat an identical landing to isolate the first-visit reward rule.
        for landing in range(3):
            env.x, env.y = target.x + 5, target.y - PLAYER_H - 1
            env.vx, env.vy, env.grounded = 0, 1, False
            _, _, _, info = env.step(0)
            self.assertEqual(info["reward_components"]["platform"], 1 if landing == 0 else 0)
            self.assertEqual(info["reward_components"]["checkpoint"], 5 if landing == 0 else 0)

    def test_retracing_ground_does_not_create_progress_reward(self):
        env = PlatformerEnv(1)
        start = env.x
        total = 0.0
        for action in [2] * 8 + [1] * 16 + [2] * 8:
            _, _, _, info = env.step(action)
            total += info["reward_components"]["progress"]
        self.assertAlmostEqual(total, 0.035 * (env.x - start))

    def test_auto_policy_roundtrip_and_collector_update(self):
        model = PPO(42, action_count=3)
        collector = ProceduralCollector(1, 2, 42, "ppo_auto_jump")
        samples, _, _ = collector.collect(model, 32)
        self.assertLess(int(samples[1].max()), 3)
        diagnostics = model.update(*samples, epochs=1)
        self.assertTrue(all(np.isfinite(value) for value in diagnostics.values()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            model.save(path, auto_jump=True)
            restored = load_policy(path)
            self.assertEqual(restored.action_count, 3)
            np.testing.assert_array_equal(model.forward(collector.observations)[0], restored.forward(collector.observations)[0])

    def test_identical_evaluation_cases_for_all_variants(self):
        expected = evaluation_cases(3, "test", 4)
        for variant in ("ppo_shaped", "ppo_auto_jump", "ppo_sparse"):
            _, rows = evaluate_procedural(None, 3, "test", 4, variant)
            self.assertEqual([(r["layout_hash"], r["start_x"]) for r in rows],
                             [(r["layout_hash"], r["start_x"]) for r in expected])

    def test_goal_height_feature_and_old_policy_compatibility(self):
        env = PlatformerEnv(generate_level(2, 200000))
        self.assertAlmostEqual(env.observe()[-1], (env.level.platforms[-1].y - env.y - PLAYER_H) / 540)
        old_model = PPO(42, observation_size=17)
        np.testing.assert_array_equal(old_model.forward([env.observe()])[0], old_model.forward([env.observe()[:17]])[0])


if __name__ == "__main__":
    unittest.main()
