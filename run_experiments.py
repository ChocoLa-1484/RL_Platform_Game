"""Reproducible procedural PPO, auto-jump, and sparse-reward experiments."""

import argparse
import csv
import json
from pathlib import Path
import random
import time

from platformer import FPS, PlatformerEnv
from ppo import PPO, np
from procedural import GENERATOR_VERSION, TRAIN_LAYOUTS, generate_level, layout_record, layout_seed
from train_ppo import ROOT, START_RANGE, RolloutCollector, positive_int

VARIANTS = {
    "ppo_shaped": {"reward_mode": "shaped", "auto_jump": False},
    "ppo_auto_jump": {"reward_mode": "shaped", "auto_jump": True},
    "ppo_sparse": {"reward_mode": "sparse", "auto_jump": False},
}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluation_cases(stage, split, count):
    return [{**layout_record(stage, seed := layout_seed(stage, split, index)),
             "start_x": random.Random(seed + 9_000_000).uniform(*START_RANGE)}
            for index in range(count)]


def evaluate_procedural(model, stage, split, count, variant="ppo_shaped", *, cases=None, level_factory=generate_level):
    config = VARIANTS[variant]
    rows = []
    for case in evaluation_cases(stage, split, count) if cases is None else cases:
        seed = case["layout_seed"]
        env = PlatformerEnv(level_factory(stage, seed), **config)
        env.reset(start_x=case["start_x"])
        rng = random.Random(seed + 8_000_000)
        total_reward = 0.0
        while not env.done:
            action = rng.randrange(3 if config["auto_jump"] else 6) if model is None else model.action(env.observe(), greedy=True)[0]
            _, reward, _, info = env.step(action)
            total_reward += reward
        rows.append({"stage": stage, "split": split, "layout_seed": seed, "layout_hash": case["layout_hash"],
                     "start_x": case["start_x"], "success": env.success, "steps": env.steps,
                     "death": env.deaths, "timeout": bool(info["truncated"]), "progress": info["progress"],
                     "platform_coverage": len(env.visited) / len(env.level.platforms),
                     "reward": total_reward, "jumps": env.jump_count,
                     "mean_landing_speed": float(np.mean(env.landing_speeds)) if env.landing_speeds else 0.0})
    successful = [r["steps"] for r in rows if r["success"]]
    rate = sum(r["success"] for r in rows) / count
    # Wilson interval describes layout sampling for one trained model, not training-seed uncertainty.
    z = 1.96
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    radius = z * np.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denominator
    summary = {
        "stage": stage, "split": split, "episodes": count,
        "successes": sum(r["success"] for r in rows), "completion_rate": rate,
        "completion_wilson95": [max(0.0, float(center - radius)), min(1.0, float(center + radius))],
        "death_count": sum(r["death"] for r in rows), "death_rate": float(np.mean([r["death"] for r in rows])),
        "timeout_count": sum(r["timeout"] for r in rows),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "mean_success_steps": float(np.mean(successful)) if successful else None,
        "mean_success_seconds": float(np.mean(successful)) / FPS if successful else None,
        "mean_progress": float(np.mean([r["progress"] for r in rows])),
        "mean_platform_coverage": float(np.mean([r["platform_coverage"] for r in rows])),
        "mean_reward": float(np.mean([r["reward"] for r in rows])),
        "mean_jumps": float(np.mean([r["jumps"] for r in rows])),
        "mean_landing_speed": float(np.mean([r["mean_landing_speed"] for r in rows])),
    }
    return summary, rows


class ProceduralCollector(RolloutCollector):
    def __init__(self, stage, count, seed, variant, rehearsal=0.25):
        self.stage = stage
        self.rehearsal = rehearsal
        self.rng = random.Random(seed)
        self.platform_start_prob = 0.0
        self.seen_layouts = set()
        self.envs = [PlatformerEnv(1, **VARIANTS[variant]) for _ in range(count)]
        self.observations = np.asarray([self.reset(env) for env in self.envs])

    def reset(self, env):
        stage = self.stage
        if stage > 1 and self.rng.random() < self.rehearsal:
            stage = self.rng.randrange(1, stage)
        seed = layout_seed(stage, "train", self.rng.randrange(TRAIN_LAYOUTS))
        self.seen_layouts.add((stage, seed))
        env.level = generate_level(stage, seed)
        env.level_number = stage
        return env.reset(start_x=self.rng.uniform(*START_RANGE))


def train_run(args, variant, training_seed):
    config = VARIANTS[variant]
    directory = args.output_dir / variant / f"seed{training_seed}"
    model = PPO(training_seed, action_count=3 if config["auto_jump"] else 6)
    history, stage_reports = [], []
    used_layouts = set()
    start_time = time.perf_counter()
    steps_per_update = args.num_envs * args.rollout_steps
    for stage in (1, 2, 3):
        collector = ProceduralCollector(stage, args.num_envs, training_seed * 100 + stage, variant, args.rehearsal)
        best_score = (-1.,)
        checkpoint = directory / f"stage{stage}_policy.json"
        first_80 = None
        selected_update = 0
        completed_total = successes_total = 0
        diagnostics = {"entropy": 0., "approx_kl": 0., "clip_fraction": 0., "value_mse": 0.}
        for iteration in range(args.updates_per_stage + 1):
            if iteration:
                samples, completed, successes = collector.collect(model, args.rollout_steps)
                completed_total += completed
                successes_total += successes
                diagnostics = model.update(*samples, learning_rate=args.learning_rate, entropy_coef=args.entropy)
            if iteration % args.eval_every == 0 or iteration == args.updates_per_stage:
                validation = [evaluate_procedural(model, s, "validation", args.validation_layouts, variant)[0]
                              for s in range(1, stage + 1)]
                completion = float(np.mean([v["completion_rate"] for v in validation]))
                progress = float(np.mean([v["mean_progress"] for v in validation]))
                coverage = float(np.mean([v["mean_platform_coverage"] for v in validation]))
                mean_steps = float(np.mean([v["mean_steps"] for v in validation]))
                total_steps = ((stage - 1) * args.updates_per_stage + iteration) * steps_per_update
                history.append({"stage": stage, "update": iteration, "environment_steps": total_steps,
                                "stage_environment_steps": iteration * steps_per_update,
                                "active_stage_completion": validation[-1]["completion_rate"],
                                "seen_stages_completion": completion, "seen_stages_progress": progress,
                                "training_episodes": completed_total, "training_successes": successes_total, **diagnostics})
                if validation[-1]["completion_rate"] >= 0.8 and first_80 is None:
                    first_80 = iteration * steps_per_update
                # Only validation layouts select checkpoints; test layouts are untouched here.
                score = (completion, coverage, progress, -mean_steps)
                if score > best_score:
                    best_score, selected_update = score, iteration
                    model.save(checkpoint, variant=variant, training_seed=training_seed, training_stage=stage,
                               selected_update=iteration, environment_steps=total_steps,
                               generator=GENERATOR_VERSION, **config)
                print(f"{variant} seed={training_seed} stage={stage} update={iteration}/{args.updates_per_stage} "
                      f"current={validation[-1]['completion_rate']:.0%} seen={completion:.0%}", flush=True)
        used_layouts.update(collector.seen_layouts)
        stage_reports.append({"stage": stage, "selected_update": selected_update,
                              "stage_steps_to_first_80pct_validation": first_80,
                              "training_episodes": completed_total, "training_successes": successes_total})
        model = PPO.load(checkpoint, seed=training_seed + stage)
    # Freeze the single final policy before any test evaluations.
    model.save(directory / "policy.json")
    test_summaries, episode_rows = [], []
    for stage in (1, 2, 3):
        summary, rows = evaluate_procedural(model, stage, "test", args.test_layouts, variant)
        test_summaries.append(summary)
        episode_rows.extend(rows)
    result = {
        "variant": variant, "training_seed": training_seed, "config": config,
        "environment_steps": 3 * args.updates_per_stage * steps_per_update,
        "elapsed_seconds": round(time.perf_counter() - start_time, 2),
        "stage_training": stage_reports, "test": test_summaries,
        "used_training_layouts": [{"stage": s, "layout_seed": seed} for s, seed in sorted(used_layouts)],
        "checkpoint_selection": "mean completion, platform coverage, progress, then speed over validation stages seen so far",
    }
    write_json(directory / "results.json", result)
    write_csv(directory / "learning_curve.csv", history)
    write_csv(directory / "test_episodes.csv", episode_rows)
    print(f"TEST {variant} seed={training_seed}: " + ", ".join(f"S{r['stage']} {r['successes']}/{r['episodes']}" for r in test_summaries), flush=True)
    return result


def aggregate_runs(runs):
    groups = []
    for variant in dict.fromkeys(run["variant"] for run in runs):
        subset = [run for run in runs if run["variant"] == variant]
        for stage in (1, 2, 3):
            metrics = [run["test"][stage - 1] for run in subset]
            entry = {"variant": variant, "stage": stage, "training_seeds": [r["training_seed"] for r in subset]}
            for key in ("completion_rate", "death_rate", "mean_progress", "mean_platform_coverage", "mean_steps"):
                values = [m[key] for m in metrics]
                entry[key + "_mean"] = float(np.mean(values))
                entry[key + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            groups.append(entry)
    return groups


def select_demo_policies(directory, runs):
    """Choose one demo per variant using validation only, retaining all test runs."""
    selections = {}
    for variant in dict.fromkeys(run["variant"] for run in runs):
        candidates = []
        for run in runs:
            if run["variant"] != variant:
                continue
            folder = directory / variant / f"seed{run['training_seed']}"
            with (folder / "learning_curve.csv").open(encoding="utf-8") as stream:
                selected = next(row for row in csv.DictReader(stream)
                                if int(row["stage"]) == 3 and int(row["update"]) == run["stage_training"][-1]["selected_update"])
            score = (float(selected["seen_stages_completion"]), float(selected["seen_stages_progress"]))
            candidates.append((score, run["training_seed"], folder / "policy.json"))
        score, seed, source = max(candidates, key=lambda candidate: candidate[0])
        policy = PPO.load(source)
        policy.save(directory / variant / "policy.json", demo_selection="validation only")
        selections[variant] = {"training_seed": seed, "source": str(source.relative_to(directory)),
                               "validation_completion": score[0], "selection": "validation only; not held-out test outcomes"}
    write_json(directory / "selected_policies.json", selections)
    return selections


def write_summary(directory, runs, random_baseline):
    groups = aggregate_runs(runs)
    write_json(directory / "summary.json", {"aggregate": groups, "random_baseline": random_baseline, "runs": runs})
    lines = ["# Procedural platformer experiment", "",
             "Mean ± sample standard deviation across training seeds. Each stage uses the same held-out layouts and starts for every model.", "",
             "| Variant | Stage | Completion | Death rate | Progress | Platform coverage |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for g in groups:
        std = g["completion_rate_std"]
        uncertainty = f" ± {std:.1%}" if std is not None else " (one seed)"
        lines.append(f"| {g['variant']} | {g['stage']} | {g['completion_rate_mean']:.1%}{uncertainty} | "
                     f"{g['death_rate_mean']:.1%} | {g['mean_progress_mean']:.1%} | {g['mean_platform_coverage_mean']:.1%} |")
    lines += ["", "Sparse reward is goal +20 / death -5 only; shaped return values are not directly comparable to sparse returns.",
              "All variants receive the same interaction budget, curriculum, rehearsal probability, and starting-position distribution.",
              "Auto-jump has three learned directional actions and automatically jumps whenever grounded.",
              "No training-only intermediate-platform starts, demonstration trajectories, or movement controllers are used.",
              "Generalization is measured within this generator's parameter ranges; it does not establish generalization to arbitrary games.",
              "", "See manifest.json for parameters and layout hashes, learning_curve.csv for learning efficiency and PPO diagnostics,",
              "and each run's test_episodes.csv for individual outcomes. Missing 80% milestones mean the threshold was not reached."]
    (directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    select_demo_policies(directory, runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--updates-per-stage", type=positive_int, default=200)
    parser.add_argument("--num-envs", type=positive_int, default=8)
    parser.add_argument("--rollout-steps", type=positive_int, default=256)
    parser.add_argument("--eval-every", type=positive_int, default=20)
    parser.add_argument("--validation-layouts", type=positive_int, default=24)
    parser.add_argument("--test-layouts", type=positive_int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--entropy", type=float, default=0.02)
    parser.add_argument("--rehearsal", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "easy_pdf")
    parser.add_argument("--evaluate-only", action="store_true", help="replay every completed run on the manifest's held-out cases; do not train or select again")
    args = parser.parse_args()
    if args.evaluate_only:
        manifest_path = args.output_dir / "manifest.json"
        if not manifest_path.is_file() or not (args.output_dir / "completed_runs.json").is_file():
            parser.error("no experiment manifest or completed runs at --output-dir")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["generator"] != GENERATOR_VERSION:
            parser.error("generator version differs from the saved experiment")
        count = manifest["parameters"]["test_layouts"]
        expected_cases = [case for stage in (1, 2, 3) for case in evaluation_cases(stage, "test", count)]
        if expected_cases != manifest["evaluation_cases"]["test"]:
            parser.error("generated test cases differ from the saved manifest; do not compare different layouts")
        runs = json.loads((args.output_dir / "completed_runs.json").read_text(encoding="utf-8"))
        for run in runs:
            model = PPO.load(args.output_dir / run["variant"] / f"seed{run['training_seed']}" / "policy.json")
            metrics = [evaluate_procedural(model, stage, "test", count, run["variant"])[0] for stage in (1, 2, 3)]
            print(json.dumps({"variant": run["variant"], "training_seed": run["training_seed"],
                              "matches_saved_results": metrics == run["test"], "test": metrics}, indent=2))
        return
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or not np.isfinite(args.entropy) or args.entropy < 0 or not 0 <= args.rehearsal < 1:
        parser.error("invalid learning rate, entropy, or rehearsal probability")
    if max(args.validation_layouts, args.test_layouts) >= 100_000:
        parser.error("layout count must be less than 100000 to keep seed splits disjoint")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants):
        parser.error("seeds and variants must not contain duplicates")
    # A run is an immutable experiment; never mix old reports with a new configuration.
    if (args.output_dir / "manifest.json").exists():
        parser.error("experiment already exists; choose a new --output-dir")
    cases = {split: [case for s in (1, 2, 3) for case in evaluation_cases(s, split, count)]
             for split, count in (("validation", args.validation_layouts), ("test", args.test_layouts))}
    train_hashes = {layout_record(s, layout_seed(s, "train", i))["layout_hash"] for s in (1, 2, 3) for i in range(TRAIN_LAYOUTS)}
    validation_hashes = {case["layout_hash"] for case in cases["validation"]}
    test_hashes = {case["layout_hash"] for case in cases["test"]}
    if train_hashes & validation_hashes or train_hashes & test_hashes or validation_hashes & test_hashes:
        parser.error("geometry collision between splits; choose a different generator version before running")
    manifest = {"generator": GENERATOR_VERSION, "numpy_version": np.__version__,
                "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "train_layout_seeds": {str(s): [layout_seed(s, "train", i) for i in range(TRAIN_LAYOUTS)] for s in (1, 2, 3)},
                "evaluation_cases": cases, "evaluation_policy": "greedy", "evaluation_start_platform": 0}
    write_json(args.output_dir / "manifest.json", manifest)
    runs = []
    for variant in args.variants:
        for seed in args.seeds:
            runs.append(train_run(args, variant, seed))
            write_json(args.output_dir / "completed_runs.json", runs)
    random_baseline = [evaluate_procedural(None, stage, "test", args.test_layouts)[0] for stage in (1, 2, 3)]
    write_summary(args.output_dir, runs, random_baseline)
    print(f"Saved experiment report: {args.output_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
