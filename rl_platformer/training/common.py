"""Shared rollout collection, GAE and argument validation."""

import argparse
import random
from rl_platformer.agents.ppo import np
from rl_platformer.envs.platformer import PLAYER_W, PlatformerEnv
from rl_platformer.paths import ROOT, START_RANGE


class RolloutCollector:
    def __init__(self, level, count, seed, platform_start_prob=0.0):
        self.envs = [PlatformerEnv(level) for _ in range(count)]
        self.rng = random.Random(seed)
        self.platform_start_prob = platform_start_prob
        self.observations = np.asarray([self.reset(env) for env in self.envs])

    def reset(self, env):
        if self.platform_start_prob and self.rng.random() < self.platform_start_prob:
            index = self.rng.randrange(1, len(env.level.platforms) - 1)
            platform = env.level.platforms[index]
            return env.reset(self.rng.uniform(platform.x, platform.x + platform.width - PLAYER_W), platform_index=index)
        return env.reset(self.rng.uniform(*START_RANGE))

    def collect(self, model, steps, gamma=0.99, gae_lambda=0.95):
        count, obs_size = self.observations.shape
        observations = np.empty((steps, count, obs_size))
        actions = np.empty((steps, count), dtype=int)
        probabilities = np.empty((steps, count))
        values = np.empty((steps, count))
        rewards = np.empty((steps, count))
        dones = np.zeros((steps, count), dtype=bool)
        completed = successes = 0
        for t in range(steps):
            observations[t] = self.observations
            actions[t], probabilities[t], values[t] = model.act_batch(self.observations)
            for index, env in enumerate(self.envs):
                obs, reward, done, info = env.step(int(actions[t, index]))
                # Time limits bootstrap the final state, but GAE never crosses a reset.
                if info["truncated"]:
                    reward += gamma * model.forward([obs])[1][0]
                rewards[t, index] = reward
                dones[t, index] = done
                if done:
                    completed += 1
                    successes += int(info["success"])
                    obs = self.reset(env)
                self.observations[index] = obs
        next_values = model.forward(self.observations)[1]
        advantages = np.empty_like(rewards)
        gae = np.zeros(count)
        for t in reversed(range(steps)):
            continuation = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_values * continuation - values[t]
            gae = delta + gamma * gae_lambda * continuation * gae
            advantages[t] = gae
            next_values = values[t]
        returns = advantages + values
        return (observations.reshape(-1, obs_size), actions.ravel(), probabilities.ravel(),
                advantages.ravel(), returns.ravel()), completed, successes


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number
