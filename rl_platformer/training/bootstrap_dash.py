"""Find faster dash demonstrations on training maps, then initialize a PPO actor.

Search runs only during training. The exported model plays without this planner.
"""

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import time
from rl_platformer.paths import BASELINE_STATE_POLICY

from rl_platformer.envs.movement import DASH_DIRECTIONS, MOVEMENT_VERSION, movement_action
from rl_platformer.agents.ppo import PPO, np
from rl_platformer.utils.io import write_csv, write_json
from rl_platformer.training.state import MovementCollector, evaluate, evaluation_cases
from rl_platformer.training.common import positive_int


def rollout(model, env, stride=8):
    trace, states = [], []
    while not env.done:
        if env.dash_energy and not env.dash_was_held and not env.dash_remaining and env.steps % stride == 0:
            states.append((len(trace), deepcopy(env)))
        observation = env.observe()
        action = model.action(observation, greedy=True)[0]
        # Equivalent button state if a legacy unmasked teacher selects an
        # unavailable dash: preserve horizontal movement and the jump button.
        if action >= 6 and not (env.dash_energy and not env.dash_was_held and not env.dash_remaining):
            dx, dy = DASH_DIRECTIONS[(action - 6) % 9]
            action = movement_action(dx, dy, jump=action >= 15)
        trace.append((observation, action))
        env.step(action)
    return trace, states


def improve_route(model, start, rounds=3, stride=8):
    """Insert a dash only when a complete simulated run succeeds sooner."""
    env = deepcopy(start)
    trace, states = rollout(model, env, stride)
    baseline = {"success": env.success, "steps": env.steps, "dashes": env.dash_count}
    search_steps = 0
    for _ in range(rounds):
        candidates = []
        for index, state in states:
            for dx, dy in DASH_DIRECTIONS[1:]:
                branch = deepcopy(state)
                action = movement_action(dx, dy, dash=True)
                branch.step(action)
                candidates.append((branch, index, state, action))
                search_steps += 1
        live = [i for i, (branch, *_) in enumerate(candidates) if not branch.done]
        while live:
            actions = model.act_batch([candidates[i][0].observe() for i in live], greedy=True)[0]
            for i, action in zip(live, actions):
                candidates[i][0].step(int(action))
            search_steps += len(live)
            live = [i for i in live if not candidates[i][0].done]
        successful = [candidate for candidate in candidates if candidate[0].success]
        if not successful:
            break
        best, index, state, action = min(successful, key=lambda candidate: candidate[0].steps)
        if env.success and best.steps >= env.steps:
            break
        env = deepcopy(state)
        prefix = trace[:index] + [(env.observe(), action)]
        env.step(action)
        suffix, states = rollout(model, env, stride)
        states = [(len(prefix) + i, s) for i, s in states]
        trace = prefix + suffix
    return trace, {"baseline": baseline, "success": env.success, "steps": env.steps,
                   "dashes": env.dash_count, "search_environment_steps": search_steps}


def greedy_route(model, start, minimum_saving=5):
    """Replan at every available state, without a cap on demonstration dashes."""
    env = deepcopy(start)
    reference = deepcopy(start)
    rollout(model, reference)
    baseline = {"success": reference.success, "steps": reference.steps, "dashes": reference.dash_count}
    expected_success, expected_steps = reference.success, reference.steps
    trace, search_steps = [], reference.steps
    while not env.done:
        action = model.action(env.observe(), greedy=True)[0]
        if env.dash_energy and not env.dash_was_held and not env.dash_remaining:
            branches = []
            for dx, dy in DASH_DIRECTIONS[1:]:
                branch = deepcopy(env)
                candidate_action = movement_action(dx, dy, dash=True)
                branch.step(candidate_action)
                branches.append((branch, candidate_action))
            search_steps += len(branches)
            live = [i for i, (branch, _) in enumerate(branches) if not branch.done]
            while live:
                actions = model.act_batch([branches[i][0].observe() for i in live], greedy=True)[0]
                for i, candidate_action in zip(live, actions):
                    branches[i][0].step(int(candidate_action))
                search_steps += len(live)
                live = [i for i in live if not branches[i][0].done]
            good = [(branch, candidate_action) for branch, candidate_action in branches if branch.success]
            if good:
                best, candidate_action = min(good, key=lambda item: item[0].steps)
                if not expected_success or best.steps <= expected_steps - minimum_saving:
                    action = candidate_action
                    expected_success, expected_steps = True, best.steps
        trace.append((env.observe(), action))
        env.step(action)
    return trace, {"baseline": baseline, "success": env.success, "steps": env.steps,
                   "dashes": env.dash_count, "search_environment_steps": search_steps}


def widen_actor(model, extra_hidden, seed):
    """Add initially silent features while preserving the transferred policy."""
    if not extra_hidden:
        return model
    old_hidden = len(model.params["b1"])
    wider = PPO(seed, hidden=old_hidden + extra_hidden, observation_size=28, action_count=24)
    wider.params["w1"][:, :old_hidden] = model.params["w1"]
    wider.params["b1"][:old_hidden] = model.params["b1"]
    wider.params["wp"][:] = 0.
    wider.params["wv"][:] = 0.
    wider.params["wp"][:old_hidden] = model.params["wp"]
    wider.params["wv"][:old_hidden] = model.params["wv"]
    wider.params["bp"][:] = model.params["bp"]
    wider.params["bv"][:] = model.params["bv"]
    wider.metadata = model.metadata.copy()
    return wider


def fit_epoch(model, observations, labels, moments, squares, step, dash_weight=4.):
    order = model.rng.permutation(len(labels))
    loss = 0.
    for start in range(0, len(labels), 512):
        ix = order[start:start + 512]
        probs, _, hidden = model.forward(observations[ix])
        weights = np.where(labels[ix] >= 6, dash_weight, 1.)
        loss -= float((weights * np.log(np.maximum(probs[np.arange(len(ix)), labels[ix]], 1e-12))).sum())
        grad = probs.copy()
        grad[np.arange(len(ix)), labels[ix]] -= 1.
        grad *= (weights / weights.sum())[:, None]
        dh = (grad @ model.params["wp"].T) * (1 - hidden * hidden)
        gradients = {"wp": hidden.T @ grad, "bp": grad.sum(axis=0),
                     "w1": observations[ix].T @ dh, "b1": dh.sum(axis=0)}
        step += 1
        for key, gradient in gradients.items():
            moments[key] = 0.9 * moments[key] + 0.1 * gradient
            squares[key] = 0.999 * squares[key] + 0.001 * gradient ** 2
            model.params[key] -= 0.0003 * (moments[key] / (1 - 0.9 ** step)) / (np.sqrt(squares[key] / (1 - 0.999 ** step)) + 1e-8)
    return step, loss / len(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-policy", type=Path, default=BASELINE_STATE_POLICY)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=positive_int, default=160)
    parser.add_argument("--rounds", type=positive_int, default=3)
    parser.add_argument("--stride", type=positive_int, default=8)
    parser.add_argument("--epochs", type=positive_int, default=60)
    parser.add_argument("--search-mode", choices=("insert", "greedy"), default="greedy")
    parser.add_argument("--extra-hidden", type=int, default=64)
    parser.add_argument("--dash-weight", type=float, default=2.)
    args = parser.parse_args()
    if args.extra_hidden < 0 or not np.isfinite(args.dash_weight) or args.dash_weight <= 0:
        parser.error("extra hidden units must be nonnegative and dash weight positive and finite")
    if args.output_dir.exists():
        parser.error("choose a new output directory")
    teacher = PPO.load(args.init_policy, seed=args.seed)
    if teacher.action_count != 24 or len(teacher.params["w1"]) != 28:
        parser.error("dash demonstrations require an existing 28-state, 24-action actor")
    if args.search_mode == "greedy":
        teacher.metadata["action_mask"] = "dash-ready-v1"
    args.output_dir.mkdir(parents=True)
    collector = MovementCollector(1, args.seed, phase=3, platform_start_prob=0.)
    observations, labels, routes = [], [], []
    started = time.perf_counter()
    for episode in range(args.episodes):
        env = collector.envs[0]
        collector.reset(env)
        trace, report = greedy_route(teacher, env) if args.search_mode == "greedy" else improve_route(teacher, env, args.rounds, args.stride)
        routes.append({"episode": episode, "level": env.level.name, "start_x": env.x, **report})
        if report["success"]:
            observations.extend(row[0] for row in trace)
            labels.extend(row[1] for row in trace)
        if (episode + 1) % 10 == 0:
            print(f"search {episode + 1}/{args.episodes}: steps {report['baseline']['steps']} -> {report['steps']}, dashes {report['dashes']}", flush=True)
    x, y = np.asarray(observations), np.asarray(labels)
    if not len(y):
        raise RuntimeError("no successful training demonstrations found")
    np.savez_compressed(args.output_dir / "demonstrations.npz", observations=x, actions=y)
    model = widen_actor(PPO.load(args.init_policy, seed=args.seed), args.extra_hidden, args.seed)
    model.metadata.update(movement_version=MOVEMENT_VERSION, action_mask="dash-ready-v1",
                          initialization="training-only dash rollout search", time_penalty=0.06)
    moments = {k: np.zeros_like(model.params[k]) for k in ("w1", "b1", "wp", "bp")}
    squares = {k: np.zeros_like(v) for k, v in moments.items()}
    step, history = 0, []
    for epoch in range(1, args.epochs + 1):
        step, loss = fit_epoch(model, x, y, moments, squares, step, dash_weight=args.dash_weight)
        if epoch % 5 == 0 or epoch == args.epochs:
            summary, _ = evaluate(model, evaluation_cases("validation", 12))
            history.append({"epoch": epoch, "loss": loss, **summary})
            print(f"imitation {epoch}: success={summary['successes']}/123 time={summary['mean_success_seconds']} dashes={summary['mean_dashes']}", flush=True)
    # This is an initialization for PPO, not a test-selected gameplay policy.
    model.save(args.output_dir / "policy.json")
    write_csv(args.output_dir / "learning_curve.csv", history)
    write_json(args.output_dir / "manifest.json", {
        "movement_version": MOVEMENT_VERSION, "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "initial_policy_sha256": hashlib.sha256(args.init_policy.read_bytes()).hexdigest(),
        "training_layouts": sorted(collector.seen_layouts), "samples": len(y), "dash_samples": int((y >= 6).sum()),
        "routes": routes, "elapsed_seconds": time.perf_counter() - started,
        "scope": "Training-pool search and weighted imitation only; no test cases used, no search at playback. Search steps are additional to PPO steps."})


if __name__ == "__main__":
    main()
