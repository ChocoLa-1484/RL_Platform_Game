"""Train and evaluate the full document's sequential-platform curriculum."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from full_curriculum import (FullCollector, GENERATOR_VERSION, STAGES, TRAIN_LAYOUTS,
                             evaluation_cases, generate_level, layout_record, layout_seed)
from ppo import PPO, np
from platformer import LEVELS
from procedural import layout_record as old_record, layout_seed as old_seed
from run_experiments import evaluate_procedural, write_csv, write_json
from train_ppo import ROOT, positive_int


def evaluate(model, stage, split, count):
    return evaluate_procedural(model, stage, split, count,
                               cases=evaluation_cases(stage, split, count), level_factory=generate_level)


def prepare_manifest(args):
    cases = {split: [case for stage in STAGES for case in evaluation_cases(stage, split, count)]
             for split, count in (("validation", args.validation_layouts), ("test", args.test_layouts))}
    train = [layout_record(stage, layout_seed(stage, "train", i)) for stage in STAGES for i in range(TRAIN_LAYOUTS)]
    hashes = [set(case["layout_hash"] for case in group) for group in (train, cases["validation"], cases["test"])]
    if any(hashes[i] & hashes[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("geometry collision between train, validation and test splits")
    if args.init_policy:
        # Prior exposure includes fixed maps and all old procedural splits, even
        # though only old training maps changed the warm-start model's weights.
        prior = {old_record(s, old_seed(s, split, i))["layout_hash"]
                 for s in (1, 2, 3) for split, count in (("train", 1000), ("validation", 24), ("test", 100))
                 for i in range(count)}
        for level in LEVELS:
            geometry = {"platforms": [asdict(p) for p in level.platforms], "start_x": level.start_x, "goal_x": level.goal_x}
            prior.add(hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest())
        if prior & (hashes[1] | hashes[2]):
            raise ValueError("evaluation geometry overlaps the prior project's layouts")
    return {"generator": GENERATOR_VERSION, "numpy_version": np.__version__,
            "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "initial_policy_sha256": hashlib.sha256(args.init_policy.read_bytes()).hexdigest() if args.init_policy else None,
            "train_layout_seeds": {str(s): [layout_seed(s, "train", i) for i in range(TRAIN_LAYOUTS)] for s in STAGES},
            "evaluation_cases": cases, "evaluation_policy": "greedy", "evaluation_start_platform": 0,
            "stage_5": "Frozen final model on unseen test layouts of stages 1-4; never trained or selected on test outcomes."}


def train(args):
    model = PPO.load(args.init_policy, seed=args.seed) if args.init_policy else PPO(args.seed)
    if model.action_count != 6:
        raise ValueError("full curriculum requires a six-action PPO model")
    manifest = prepare_manifest(args)
    write_json(args.output_dir / "manifest.json", manifest)
    if args.init_policy:
        model.save(args.output_dir / "initial_policy.json")
    history, stage_reports, used_layouts = [], [], set()
    started = time.perf_counter()
    steps_per_update = args.num_envs * args.rollout_steps
    for stage in STAGES:
        collector = FullCollector(stage, args.num_envs, args.seed * 100 + stage,
                                  args.rehearsal, args.platform_start_prob)
        best_score, selected_update, first_80 = (-1.,), 0, None
        checkpoint = args.output_dir / f"stage{stage}_policy.json"
        completed_total = successes_total = 0
        for iteration in range(args.updates_per_stage + 1):
            if iteration:
                samples, completed, successes = collector.collect(model, args.rollout_steps)
                completed_total += completed
                successes_total += successes
                model.update(*samples, learning_rate=args.learning_rate, entropy_coef=args.entropy)
            if iteration % args.eval_every == 0 or iteration == args.updates_per_stage:
                validation = [evaluate(model, s, "validation", args.validation_layouts)[0] for s in range(1, stage + 1)]
                completion = float(np.mean([v["completion_rate"] for v in validation]))
                progress = float(np.mean([v["mean_progress"] for v in validation]))
                coverage = float(np.mean([v["mean_platform_coverage"] for v in validation]))
                mean_steps = float(np.mean([v["mean_steps"] for v in validation]))
                total_steps = ((stage - 1) * args.updates_per_stage + iteration) * steps_per_update
                history.append({"stage": stage, "update": iteration, "environment_steps": total_steps,
                                "current_stage_completion": validation[-1]["completion_rate"],
                                "seen_stages_completion": completion, "seen_stages_progress": progress,
                                "training_episodes": completed_total, "training_successes": successes_total})
                if validation[-1]["completion_rate"] >= 0.8 and first_80 is None:
                    first_80 = iteration * steps_per_update
                score = (completion, coverage, progress, -mean_steps)
                if score > best_score:
                    best_score, selected_update = score, iteration
                    model.save(checkpoint, generator=GENERATOR_VERSION, training_seed=args.seed,
                               training_stage=stage, selected_update=iteration, environment_steps=total_steps,
                               reward_mode="shaped", auto_jump=False)
                print(f"full stage={stage} update={iteration}/{args.updates_per_stage} "
                      f"current={validation[-1]['completion_rate']:.0%} seen={completion:.0%}", flush=True)
        used_layouts.update(collector.seen_layouts)
        stage_reports.append({"stage": stage, "selected_update": selected_update,
                              "steps_to_first_80pct_validation": first_80,
                              "training_episodes": completed_total, "training_successes": successes_total})
        model = PPO.load(checkpoint, seed=args.seed + stage)
    # Both weights and checkpoint choice are frozen before evaluating Stage 5.
    model.save(args.output_dir / "policy.json")
    summaries, rows, baseline = [], [], []
    for stage in STAGES:
        summary, episodes = evaluate(model, stage, "test", args.test_layouts)
        summaries.append(summary)
        rows.extend(episodes)
        baseline.append(evaluate(None, stage, "test", args.test_layouts)[0])
        print(f"TEST stage {stage}: {summary['successes']}/{args.test_layouts}", flush=True)
    result = {"generator": GENERATOR_VERSION, "training_seed": args.seed,
              "environment_steps": 4 * args.updates_per_stage * steps_per_update,
              "elapsed_seconds": round(time.perf_counter() - started, 2),
              "stage_training": stage_reports, "test": summaries, "random_baseline": baseline,
              "used_training_layouts": [{"stage": s, "layout_seed": seed} for s, seed in sorted(used_layouts)],
              "checkpoint_selection": "mean completion, coverage, progress, speed over validation stages seen so far",
              "scope": "One training seed; held-out layouts from this generator, not arbitrary games.",
              "warm_start": args.init_policy is not None}
    write_json(args.output_dir / "results.json", result)
    write_csv(args.output_dir / "learning_curve.csv", history)
    write_csv(args.output_dir / "test_episodes.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-policy", type=Path, help="optional six-action PPO warm start; omit to train from scratch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--updates-per-stage", type=positive_int, default=300)
    parser.add_argument("--num-envs", type=positive_int, default=8)
    parser.add_argument("--rollout-steps", type=positive_int, default=256)
    parser.add_argument("--eval-every", type=positive_int, default=20)
    parser.add_argument("--validation-layouts", type=positive_int, default=24)
    parser.add_argument("--test-layouts", type=positive_int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--entropy", type=float, default=0.02)
    parser.add_argument("--rehearsal", type=float, default=0.25)
    parser.add_argument("--platform-start-prob", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "full_version")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.evaluate_only:
        if not all((args.output_dir / name).is_file() for name in ("manifest.json", "results.json", "policy.json")):
            parser.error("no completed experiment at --output-dir")
        manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
        saved = json.loads((args.output_dir / "results.json").read_text(encoding="utf-8"))
        count = manifest["parameters"]["test_layouts"]
        cases = [case for s in STAGES for case in evaluation_cases(s, "test", count)]
        if manifest["generator"] != GENERATOR_VERSION or cases != manifest["evaluation_cases"]["test"]:
            parser.error("test layouts differ from the saved manifest")
        model = PPO.load(args.output_dir / "policy.json")
        summaries = [evaluate(model, s, "test", count)[0] for s in STAGES]
        print(json.dumps({"matches_saved_results": summaries == saved["test"], "test": summaries}, indent=2))
        return
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or not np.isfinite(args.entropy) or args.entropy < 0:
        parser.error("learning rate must be positive and entropy nonnegative, both finite")
    if not 0 <= args.rehearsal < 1 or not 0 <= args.platform_start_prob < 1:
        parser.error("rehearsal and platform-start-prob must be in [0, 1)")
    if max(args.validation_layouts, args.test_layouts) >= 100_000:
        parser.error("layout count must be less than 100000")
    if (args.output_dir / "manifest.json").exists():
        parser.error("experiment exists; choose a new --output-dir")
    train(args)


if __name__ == "__main__":
    main()
