"""Train variable-jump/dash PPO, selecting completion first and elapsed time second."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

from rl_platformer.levels import full_curriculum
from rl_platformer.levels import height_curriculum
from rl_platformer.envs.movement import DASH_DIRECTIONS, MOVEMENT_ACTIONS, MOVEMENT_OBSERVATION_SIZE, MOVEMENT_VERSION, MOVEMENT_VERSIONS, MovementEnv
from rl_platformer.envs.platformer import FPS, LEVELS
from rl_platformer.agents.ppo import PPO, np
from rl_platformer.utils.io import write_csv, write_json
from rl_platformer.training.common import ROOT, START_RANGE, RolloutCollector, positive_int

GROUPS = [("full", s) for s in range(1, 5)] + [("height", s) for s in range(1, 4)]
TRAIN_LAYOUTS = 1000


def layout_seed(family, stage, split, index):
    if family not in ("full", "height") or (family, stage) not in GROUPS or split not in ("train", "validation", "test") or not 0 <= index < 100_000:
        raise ValueError("invalid movement layout request")
    return (9_000_000 if family == "full" else 12_000_000) + ("train", "validation", "test").index(split) * 1_000_000 + stage * 100_000 + index


def generated_level(family, stage, seed):
    return (full_curriculum if family == "full" else height_curriculum).generate_level(stage, seed)


def case_level(case):
    return LEVELS[case["stage"] - 1] if case["family"] == "fixed" else generated_level(case["family"], case["stage"], case["layout_seed"])


def geometry_key(level):
    # Exclude display names/seed labels: identical geometry must hash identically.
    geometry = {"platforms": [asdict(p) for p in level.platforms], "start_x": level.start_x,
                "goal_x": level.goal_x, "world_width": level.world_width, "world_height": level.world_height}
    return hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()


def audit_layout_splits(validation, testing):
    training = {geometry_key(generated_level(f, s, layout_seed(f, s, "train", i)))
                for f, s in GROUPS for i in range(TRAIN_LAYOUTS)}
    groups = [training] + [{geometry_key(case_level(c)) for c in cases if c["family"] != "fixed"}
                           for cases in (validation, testing)]
    collisions = [len(groups[i] & groups[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
    return {"unique_training_geometries": len(training), "unique_validation_geometries": len(groups[1]),
            "unique_test_geometries": len(groups[2]), "cross_split_collisions": collisions,
            "collision_order": ["train-validation", "train-test", "validation-test"],
            "test_cases_overlapping_training_or_validation": [
                c for c in testing if c["family"] != "fixed" and geometry_key(case_level(c)) in (groups[0] | groups[1])],
            "scope": "Current training pool only; inherited teacher exposure is not audited. Different seeds can produce identical geometry.",
            "fixed_maps": "Intentionally trained; excluded from unseen-geometry claims."}


def evaluation_cases(split, layouts, fixed_starts=3):
    cases = []
    for family, stage in GROUPS + [("fixed", n) for n in range(1, len(LEVELS) + 1)]:
        for index in range(fixed_starts if family == "fixed" else layouts):
            seed = (20_000_000 if split == "validation" else 21_000_000) + stage * 1000 + index if family == "fixed" else layout_seed(family, stage, split, index)
            case = {"family": family, "stage": stage, "layout_seed": seed,
                    "start_x": random.Random(seed + 31_000_000).uniform(*START_RANGE)}
            case["layout_hash"] = hashlib.sha256(json.dumps(asdict(case_level(case)), sort_keys=True).encode()).hexdigest()
            cases.append(case)
    return cases


def evaluate(model, cases, movement_version=MOVEMENT_VERSION, allow_dash=True):
    envs = [MovementEnv(case_level(case), movement_version=movement_version) for case in cases]
    for env, case in zip(envs, cases):
        env.reset(case["start_x"])
    live = list(range(len(envs)))
    while live:
        observations = [envs[i].observe() for i in live]
        actions = model.act_batch(observations, greedy=True)[0] if allow_dash else model.forward(observations)[0][:, :6].argmax(axis=1)
        for i, action in zip(live, actions):
            envs[i].step(int(action))
        live = [i for i in live if not envs[i].done]
    rows = [{**case, "success": env.success, "steps": env.steps, "seconds": env.steps / FPS,
             "max_steps": env.level.max_steps, "progress": min(1., (env.best_x + 24) / env.level.goal_x),
             "dashes": env.dash_count, "jump_releases": env.jump_release_count,
             "death": bool(env.deaths), "timeout": not env.success and not env.deaths}
            for env, case in zip(envs, cases)]
    successful = [r for r in rows if r["success"]]
    summary = {"episodes": len(rows), "successes": len(successful), "completion_rate": len(successful) / len(rows),
               "mean_success_seconds": float(np.mean([r["seconds"] for r in successful])) if successful else None,
               "mean_progress": float(np.mean([r["progress"] for r in rows])),
               "penalized_time": float(np.mean([r["steps"] / r["max_steps"] if r["success"] else 1. for r in rows])),
               "mean_dashes": float(np.mean([r["dashes"] for r in rows])),
               "mean_jump_releases": float(np.mean([r["jump_releases"] for r in rows]))}
    return summary, rows


def selection_score(summary, minimum_completion=None):
    if minimum_completion is None:
        return summary["completion_rate"], -summary["penalized_time"], summary["mean_progress"]
    # Once reliability reaches the requested floor, minimize time including
    # failed runs. Below that floor, recovering completion remains primary.
    if summary["completion_rate"] >= minimum_completion:
        return 1., -summary["penalized_time"], summary["completion_rate"], summary["mean_progress"]
    return 0., summary["completion_rate"], -summary["penalized_time"], summary["mean_progress"]


class MovementCollector(RolloutCollector):
    def __init__(self, count, seed, phase=3, platform_start_prob=0.25, time_penalty=0.04):
        self.rng = random.Random(seed)
        self.phase = phase
        self.platform_start_prob = platform_start_prob
        self.seen_layouts = set()
        self.envs = [MovementEnv(time_penalty=time_penalty) for _ in range(count)]
        self.observations = np.asarray([self.reset(env) for env in self.envs])

    def reset(self, env):
        if self.rng.random() < 0.3:
            number = self.rng.randint(1, (5, 10, len(LEVELS))[self.phase - 1])
            env.level = LEVELS[number - 1]
            self.seen_layouts.add(("fixed", number, 0))
        else:
            groups = [("full", 1), ("full", 2), ("height", 1)] if self.phase == 1 else GROUPS[:-1] if self.phase == 2 else GROUPS
            family, stage = self.rng.choice(groups)
            seed = layout_seed(family, stage, "train", self.rng.randrange(TRAIN_LAYOUTS))
            env.level = generated_level(family, stage, seed)
            self.seen_layouts.add((family, stage, seed))
        return super().reset(env)


def expand_policy(teacher, seed):
    if teacher.action_count == len(MOVEMENT_ACTIONS) and teacher.params["w1"].shape[0] == MOVEMENT_OBSERVATION_SIZE:
        return teacher
    if teacher.action_count != 6 or teacher.params["w1"].shape[0] not in (17, 18):
        raise ValueError("initialize from a six-action legacy PPO or an agile PPO")
    model = PPO(seed, hidden=len(teacher.params["b1"]), observation_size=MOVEMENT_OBSERVATION_SIZE,
                action_count=len(MOVEMENT_ACTIONS))
    model.params["w1"][:] = 0.
    model.params["w1"][:len(teacher.params["w1"])] = teacher.params["w1"]
    for key in ("b1", "wv", "bv"):
        model.params[key][:] = teacher.params[key]
    model.params["wp"][:, :6] = teacher.params["wp"]
    model.params["bp"][:6] = teacher.params["bp"]
    model.params["wp"][:, 6:] = 0.
    model.params["bp"][6:] = -5.
    return model


def bootstrap_controls(model, teacher, seed, episodes=240, epochs=20):
    """Imitate legacy routes with held-jump semantics on training layouts only.

    No controller or teacher is used during PPO rollout or final playback.
    """
    collector = MovementCollector(1, seed, phase=3, platform_start_prob=0.)
    observations, labels = [], []
    env = collector.envs[0]
    for _ in range(episodes):
        collector.reset(env)
        while not env.done:
            observation = env.observe()
            action = teacher.action(observation[:18], greedy=True)[0]
            # Old jumps ignored the button in flight. Preserve the old full
            # arc for demonstrations, and release before the next takeoff.
            hold = (env.jump_active and env.vy < 0) or (env.grounded and action >= 3 and not env.jump_was_held)
            action = action % 3 + (3 if hold else 0)
            observations.append(observation)
            labels.append(action)
            env.step(action)
    x, labels = np.asarray(observations), np.asarray(labels)
    steps = 0
    moments = {k: np.zeros_like(model.params[k]) for k in ("w1", "b1", "wp", "bp")}
    squares = {k: np.zeros_like(v) for k, v in moments.items()}
    for _ in range(epochs):
        order = model.rng.permutation(len(labels))
        for start in range(0, len(labels), 512):
            ix = order[start:start + 512]
            probs, _, hidden = model.forward(x[ix])
            grad = probs.copy()
            grad[np.arange(len(ix)), labels[ix]] -= 1
            grad /= len(ix)
            dh = (grad @ model.params["wp"].T) * (1 - hidden * hidden)
            gradients = {"wp": hidden.T @ grad, "bp": grad.sum(axis=0), "w1": x[ix].T @ dh, "b1": dh.sum(axis=0)}
            steps += 1
            for key, gradient in gradients.items():
                moments[key] = 0.9 * moments[key] + 0.1 * gradient
                squares[key] = 0.999 * squares[key] + 0.001 * gradient ** 2
                model.params[key] -= 0.0005 * (moments[key] / (1 - 0.9 ** steps)) / (np.sqrt(squares[key] / (1 - 0.999 ** steps)) + 1e-8)
    return {"episodes": episodes, "samples": len(labels), "epochs": epochs,
            "training_layouts": sorted(collector.seen_layouts)}


def seed_dash_exploration(model):
    # Related movement logits give dashes a small, state-dependent probability.
    # A constant logit would disappear next to a confident transferred policy.
    for action in range(6, len(MOVEMENT_ACTIONS)):
        dx, _ = DASH_DIRECTIONS[(action - 6) % 9]
        base = (1 if dx < 0 else 2) + (3 if action >= 15 else 0)
        model.params["wp"][:, action] = model.params["wp"][:, base]
        model.params["bp"][action] = model.params["bp"][base] - 3.


def summarize_groups(rows):
    summaries = []
    for family, stage in GROUPS + [("fixed", n) for n in range(1, len(LEVELS) + 1)]:
        group = [r for r in rows if r["family"] == family and r["stage"] == stage]
        successful = [r for r in group if r["success"]]
        summaries.append({"family": family, "stage": stage, "episodes": len(group), "successes": len(successful),
                          "mean_success_seconds": float(np.mean([r["seconds"] for r in successful])) if successful else None,
                          "mean_dashes": float(np.mean([r["dashes"] for r in group]))})
    return summaries


def train(args):
    teacher = PPO.load(args.init_policy, seed=args.seed)
    model = expand_policy(teacher, args.seed)
    use_mask = args.mask_dash or args.refresh_dash or model.metadata.get("action_mask") == "dash-ready-v1"
    if use_mask:
        model.metadata["action_mask"] = "dash-ready-v1"
    validation = evaluation_cases("validation", args.validation_layouts, 3)
    testing = evaluation_cases("test", args.test_layouts, 20)
    manifest = {"movement_version": MOVEMENT_VERSION, "actions": MOVEMENT_ACTIONS,
                "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "initial_policy_sha256": hashlib.sha256(args.init_policy.read_bytes()).hexdigest(),
                "validation_cases": validation, "test_cases": testing, "numpy_version": np.__version__,
                "selection": "completion rate, then failure-penalized normalized time; validation only" if args.minimum_completion is None else
                             f"reach completion >= {args.minimum_completion}, then minimize failure-penalized normalized time; validation only",
                "geometry_audit": audit_layout_splits(validation, testing)}
    write_json(args.output_dir / "manifest.json", manifest)
    started = time.perf_counter()
    warmup = bootstrap_controls(model, teacher, args.seed, args.bootstrap_episodes, args.bootstrap_epochs) if teacher.action_count == 6 else None
    if args.refresh_dash or (teacher.action_count == 6 and args.dash_initialization == "explore"):
        seed_dash_exploration(model)
    elif teacher.action_count == 6:
        model.params["bp"][6:] = -3.
    model.metadata = {"movement_version": MOVEMENT_VERSION, "reward_mode": "speedrun", "training_seed": args.seed,
                      "time_penalty": args.time_penalty}
    if use_mask:
        model.metadata["action_mask"] = "dash-ready-v1"
    model.save(args.output_dir / "initial_policy.json")
    history, reports, used = [], [], set()
    steps_per_update = args.num_envs * args.rollout_steps
    for phase_index, phase in enumerate(args.phases):
        collector = MovementCollector(args.num_envs, args.seed * 100 + phase, phase, args.platform_start_prob, args.time_penalty)
        best, selected = (-1.,), 0
        checkpoint = args.output_dir / f"phase{phase}_policy.json"
        diagnostics = {"entropy": 0., "approx_kl": 0., "clip_fraction": 0., "value_mse": 0., "epochs_run": 0, "rollout_kl": 0.}
        for iteration in range(args.updates_per_phase + 1):
            fraction = iteration / args.updates_per_phase
            collector.platform_start_prob = args.platform_start_prob * (1 - fraction)
            if iteration:
                samples, _, _ = collector.collect(model, args.rollout_steps, gamma=args.gamma)
                diagnostics = model.update(*samples, learning_rate=args.learning_rate * (1 - 0.5 * fraction),
                                           entropy_coef=args.entropy, target_kl=0.02, value_coef=0.1)
            if iteration % args.eval_every == 0 or iteration == args.updates_per_phase:
                summary, _ = evaluate(model, validation)
                score = selection_score(summary, args.minimum_completion)
                if score > best:
                    best, selected = score, iteration
                    model.save(checkpoint, selected_phase=phase, selected_update=iteration)
                history.append({"phase": phase, "update": iteration,
                                "environment_steps": (phase_index * args.updates_per_phase + iteration) * steps_per_update,
                                **summary, **diagnostics})
                write_csv(args.output_dir / "learning_curve.csv", history)
                seconds = summary["mean_success_seconds"]
                print(f"agile phase={phase} update={iteration}/{args.updates_per_phase} "
                      f"success={summary['successes']}/{summary['episodes']} time={seconds} dashes={summary['mean_dashes']:.1f}", flush=True)
        used.update(collector.seen_layouts)
        reports.append({"phase": phase, "selected_update": selected, "score": best})
        model = PPO.load(checkpoint, seed=args.seed + phase)
    model.save(args.output_dir / "policy.json")
    # Test outcomes are inspected only after the final checkpoint is frozen.
    trained, rows = evaluate(model, testing)
    baseline, before = evaluate(PPO.load(args.output_dir / "initial_policy.json"), testing)
    paired = [(old, new) for old, new in zip(before, rows) if old["success"] and new["success"]]
    report = {"movement_version": MOVEMENT_VERSION, "environment_steps": len(args.phases) * args.updates_per_phase * steps_per_update,
              "elapsed_seconds": round(time.perf_counter() - started, 2), "bootstrap": warmup,
              "phase_training": reports, "training_layouts": sorted(used), "trained": trained, "baseline": baseline,
              "trained_groups": summarize_groups(rows), "baseline_groups": summarize_groups(before),
              "paired_successes": len(paired),
              "paired_before_seconds": float(np.mean([a["seconds"] for a, b in paired])) if paired else None,
              "paired_after_seconds": float(np.mean([b["seconds"] for a, b in paired])) if paired else None,
              "scope": "One seed; baseline is the initialized policy on the same new physics, after control imitation or dash reseeding when requested. Fixed maps are trained; procedural test seeds are held out, but geometry collisions are reported in the manifest."}
    write_json(args.output_dir / "results.json", report)
    write_csv(args.output_dir / "test_episodes.csv", rows)
    write_csv(args.output_dir / "baseline_episodes.csv", before)
    print(json.dumps({"trained": trained, "baseline": baseline, "paired_before_seconds": report["paired_before_seconds"],
                      "paired_after_seconds": report["paired_after_seconds"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-policy", type=Path, default=ROOT / "experiments" / "state" / "dash_demonstrations_v2" / "policy.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "state" / "dash_speedrun")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--updates-per-phase", type=positive_int, default=1600)
    parser.add_argument("--phases", type=int, nargs="+", choices=(1, 2, 3), default=[2, 3])
    parser.add_argument("--num-envs", type=positive_int, default=8)
    parser.add_argument("--rollout-steps", type=positive_int, default=256)
    parser.add_argument("--eval-every", type=positive_int, default=40)
    parser.add_argument("--validation-layouts", type=positive_int, default=12)
    parser.add_argument("--test-layouts", type=positive_int, default=100)
    parser.add_argument("--bootstrap-episodes", type=positive_int, default=240)
    parser.add_argument("--bootstrap-epochs", type=positive_int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--platform-start-prob", type=float, default=0.20)
    parser.add_argument("--time-penalty", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--minimum-completion", type=float, default=0.98,
                        help="after reaching this validation completion floor, select by failure-penalized time")
    parser.add_argument("--refresh-dash", action="store_true", help="reseed dash exploration when continuing an agile model")
    parser.add_argument("--mask-dash", action="store_true", help="mask dash actions while energy or button state prevents a fresh dash")
    parser.add_argument("--dash-initialization", choices=("conservative", "explore"), default="conservative",
                        help="initial exploration of dash actions when adapting a six-action model")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.evaluate_only:
        if not all((args.output_dir / name).is_file() for name in ("manifest.json", "results.json", "policy.json", "initial_policy.json")):
            parser.error("no completed movement experiment at --output-dir")
        manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
        saved = json.loads((args.output_dir / "results.json").read_text(encoding="utf-8"))
        cases = evaluation_cases("test", manifest["parameters"]["test_layouts"], 20)
        version = manifest["movement_version"]
        if version not in MOVEMENT_VERSIONS or manifest["test_cases"] != cases:
            parser.error("physics version or test geometry differs from the saved manifest")
        trained, _ = evaluate(PPO.load(args.output_dir / "policy.json"), cases, movement_version=version)
        baseline, _ = evaluate(PPO.load(args.output_dir / "initial_policy.json"), cases, movement_version=version)
        print(json.dumps({"matches_saved_results": trained == saved["trained"] and baseline == saved["baseline"], "trained": trained}, indent=2))
        return
    if not args.init_policy.is_file():
        parser.error("initial policy does not exist")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or not np.isfinite(args.entropy) or args.entropy < 0:
        parser.error("learning rate must be positive and entropy nonnegative, both finite")
    if not 0 <= args.platform_start_prob < 1 or max(args.validation_layouts, args.test_layouts) >= 100_000:
        parser.error("invalid platform start probability or layout count")
    if not 0 < args.gamma <= 1 or not np.isfinite(args.time_penalty) or args.time_penalty <= 0:
        parser.error("gamma must be in (0, 1] and time penalty positive and finite")
    if len(args.phases) != len(set(args.phases)):
        parser.error("phases must not repeat, because each phase has one checkpoint")
    if args.minimum_completion is not None and not 0 < args.minimum_completion <= 1:
        parser.error("minimum completion must be in (0, 1]")
    if (args.output_dir / "manifest.json").exists():
        parser.error("experiment exists; choose a new output directory")
    train(args)


if __name__ == "__main__":
    main()
