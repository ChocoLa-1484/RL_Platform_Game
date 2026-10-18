"""Train PPO on one level or a curriculum, then evaluate on held-out starts."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

from ppo import LinearPPO, PPO, load_policy, np
from platformer import ACTIONS, LEVELS, PLAYER_W, PlatformerEnv

ROOT = Path(__file__).resolve().parent
START_RANGE = (25.0, 50.0)


def evaluate(model, episodes=100, seed=100912, level=1):
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    # Separate RNG streams give every policy exactly the same starting positions.
    starts = random.Random(seed)
    actions = random.Random(seed + 1)
    results = []
    for _ in range(episodes):
        env = PlatformerEnv(level)
        env.reset(start_x=starts.uniform(*START_RANGE))
        total_reward = 0.0
        while not env.done:
            action = actions.randrange(len(ACTIONS)) if model is None else model.action(env.observe(), greedy=True)[0]
            _, reward, _, info = env.step(action)
            total_reward += reward
        results.append((env.success, env.steps, env.deaths, info["progress"], total_reward, info["truncated"]))
    return {
        "episodes": episodes,
        "successes": sum(r[0] for r in results),
        "completion_rate": sum(r[0] for r in results) / episodes,
        "mean_steps": sum(r[1] for r in results) / episodes,
        "mean_deaths": sum(r[2] for r in results) / episodes,
        "mean_progress": sum(r[3] for r in results) / episodes,
        "mean_reward": sum(r[4] for r in results) / episodes,
        "timeouts": sum(r[5] for r in results),
    }


def evaluate_random(episodes=100, seed=100912, level=1):
    return evaluate(None, episodes, seed, level)


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


def score(result):
    return (result["completion_rate"], result["mean_progress"], result["mean_reward"], -result["mean_steps"])


def evaluate_default_start(model, level):
    env = PlatformerEnv(level)
    while not env.done:
        env.step(model.action(env.observe(), greedy=True)[0])
    return {"success": env.success, "steps": env.steps}


def train_level(args, level, initial_path=None):
    seed = args.seed + level - 1
    model = PPO.load(initial_path, seed=seed) if initial_path else PPO(seed)
    collector = RolloutCollector(level, args.num_envs, seed, args.platform_start_prob)
    policy_path = args.output_dir / f"level{level}_policy.json"
    best_score = (-1.0,)
    best_update = 0
    history = []
    started = time.perf_counter()
    completed_total = success_total = 0
    for iteration in range(args.updates + 1):
        if iteration:
            samples, completed, successes = collector.collect(model, args.rollout_steps)
            completed_total += completed
            success_total += successes
            model.update(*samples, learning_rate=args.learning_rate, entropy_coef=args.entropy)
        if iteration % args.eval_every == 0 or iteration == args.updates:
            result = evaluate(model, episodes=args.validation_episodes, seed=912, level=level)
            history.append({"update": iteration, "environment_steps": iteration * args.rollout_steps * args.num_envs, **result})
            print(f"level {level} | update {iteration:4}/{args.updates} | validation {result['successes']}/{args.validation_episodes} | progress {result['mean_progress']:.0%}", flush=True)
            if score(result) > best_score:
                best_score, best_update = score(result), iteration
                model.save(policy_path, level=level, level_name=LEVELS[level - 1].name,
                           seed=seed, selected_update=iteration, initialized_from=str(initial_path) if initial_path else None)
    best = load_policy(policy_path)
    trained = evaluate(best, args.eval_episodes, seed=100912, level=level)
    random_result = evaluate_random(args.eval_episodes, seed=100912, level=level)
    report = {
        "level": level, "name": LEVELS[level - 1].name, "difficulty": LEVELS[level - 1].difficulty,
        "layout": asdict(LEVELS[level - 1]),
        "algorithm": "NumPy MLP actor-critic PPO with GAE and entropy bonus",
        "seed": seed, "updates": args.updates, "num_envs": args.num_envs,
        "rollout_steps": args.rollout_steps, "learning_rate": args.learning_rate, "entropy": args.entropy,
        "platform_start_probability": args.platform_start_prob,
        "environment_steps": args.updates * args.num_envs * args.rollout_steps,
        "training_episodes": completed_total, "training_successes": success_total,
        "selected_update": best_update, "initialized_from": str(initial_path) if initial_path else None,
        "validation_seed": 912, "validation_episodes": args.validation_episodes,
        "evaluation_seed": 100912, "evaluation_episodes": args.eval_episodes,
        "start_x_range": START_RANGE, "evaluation_mode": "greedy",
        "default_start": evaluate_default_start(best, level),
        "trained_policy": trained, "random_policy": random_result,
        "elapsed_seconds": round(time.perf_counter() - started, 2), "history": history,
        "scope": "Fixed layouts with randomized starting X; not unseen-level generalization.",
    }
    (args.output_dir / f"level{level}_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {policy_path} | held-out test {trained['successes']}/{args.eval_episodes}", flush=True)
    return policy_path, report


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--level", type=int, choices=range(1, len(LEVELS) + 1), default=1)
    selection.add_argument("--curriculum", action="store_true", help="train all levels in order, transferring the previous best weights")
    selection.add_argument("--levels", type=int, nargs="+", choices=range(1, len(LEVELS) + 1), help="train only these levels in the specified order")
    parser.add_argument("--updates", type=positive_int, default=160)
    parser.add_argument("--num-envs", "--episodes-per-update", dest="num_envs", type=positive_int, default=8,
                        help="parallel rollout environments (old flag is an alias)")
    parser.add_argument("--rollout-steps", type=positive_int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.0007)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--platform-start-prob", type=float, default=0.0,
                        help="training-only probability of starting on an intermediate platform; evaluation always starts on platform 0")
    parser.add_argument("--eval-every", type=positive_int, default=10)
    parser.add_argument("--validation-episodes", type=positive_int, default=24)
    parser.add_argument("--eval-episodes", type=positive_int, default=100)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--init-policy", type=Path, help="initialize from a NumPy PPO model (optimizer is reset)")
    parser.add_argument("--evaluate-only", action="store_true", help="evaluate saved models without training")
    args = parser.parse_args()
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or not np.isfinite(args.entropy) or args.entropy < 0:
        parser.error("learning rate must be finite and positive; entropy must be finite and nonnegative")
    if not 0 <= args.platform_start_prob < 1:
        parser.error("platform-start-prob must be in [0, 1)")
    levels = range(1, len(LEVELS) + 1) if args.curriculum else args.levels or [args.level]
    if args.evaluate_only:
        for level in levels:
            path = args.output_dir / f"level{level}_policy.json"
            if not path.is_file():
                parser.error(f"policy missing: {path}; train this level first")
            model = load_policy(path)
            print(json.dumps({"level": level, "trained_policy": evaluate(model, args.eval_episodes, level=level),
                              "random_policy": evaluate_random(args.eval_episodes, level=level),
                              "default_start": evaluate_default_start(model, level)}, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous = args.init_policy
    reports = []
    for level in levels:
        previous, report = train_level(args, level, previous)
        reports.append(report)
    # Refresh the overview after single-level fine-tuning as well as a full curriculum.
    reports = [json.loads(path.read_text(encoding="utf-8"))
               for number in range(1, len(LEVELS) + 1)
               if (path := args.output_dir / f"level{number}_results.json").is_file()]
    (args.output_dir / "curriculum_results.json").write_text(json.dumps({"levels": reports}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
