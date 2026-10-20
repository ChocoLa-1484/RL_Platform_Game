"""Train an independent screenshot CNN, with live preview enabled by default."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import time

from rl_platformer.agents.vision import VisionAgent, OBSERVATION_SHAPE, ppo_update, torch, np
from rl_platformer.envs.vision import PixelEnv, LivePreview, StopTraining
from rl_platformer.envs.movement import MOVEMENT_ACTIONS, MOVEMENT_VERSION
from rl_platformer.envs.platformer import LEVELS, FPS
from rl_platformer.agents.ppo import PPO
from rl_platformer.training.state import (GROUPS, TRAIN_LAYOUTS, layout_seed, generated_level, case_level,
                            evaluation_cases, evaluate as evaluate_state, selection_score,
                            summarize_groups, audit_layout_splits)
from rl_platformer.training.common import positive_int
from rl_platformer.utils.io import write_json, write_csv

from rl_platformer.paths import ROOT, resolve_saved_path
BASELINE = ROOT / "experiments" / "state" / "dash_speedrun" / "policy.json"
REWARD_SCALE = .05


def create_run_directory(requested=None):
    destination = Path(requested) if requested else ROOT / "experiments" / "vision" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # Even an empty existing directory is refused: every run is independent.
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def reset_training(env, rng, phase):
    if rng.random() < .3:
        level = LEVELS[rng.randrange((5, 10, len(LEVELS))[phase - 1])]
    else:
        groups = [("full", 1), ("full", 2), ("height", 1)] if phase == 1 else GROUPS[:-1] if phase == 2 else GROUPS
        family, stage = rng.choice(groups)
        level = generated_level(family, stage, layout_seed(family, stage, "train", rng.randrange(TRAIN_LAYOUTS)))
    return env.reset(rng.uniform(25., 50.), level)


def episode_row(env, case):
    sim = env.simulator
    return {**case, "success": sim.success, "steps": sim.steps, "seconds": sim.steps / FPS,
            "max_steps": sim.level.max_steps, "progress": min(1., (sim.best_x + 24) / sim.level.goal_x),
            "dashes": sim.dash_count, "jump_releases": sim.jump_release_count,
            "death": bool(sim.deaths), "timeout": not sim.success and not sim.deaths}


def summary(rows):
    good = [r for r in rows if r["success"]]
    return {"episodes": len(rows), "successes": len(good), "completion_rate": len(good) / len(rows),
            "mean_success_seconds": float(np.mean([r["seconds"] for r in good])) if good else None,
            "mean_progress": float(np.mean([r["progress"] for r in rows])),
            "penalized_time": float(np.mean([r["steps"] / r["max_steps"] if r["success"] else 1. for r in rows])),
            "mean_dashes": float(np.mean([r["dashes"] for r in rows])),
            "mean_jump_releases": float(np.mean([r["jump_releases"] for r in rows]))}


def evaluate_pixels(model, cases, preview=None, label="VALIDATION - greedy CNN", batch_size=12):
    rows = []
    for start in range(0, len(cases), batch_size):
        chunk = cases[start:start + batch_size]
        envs = [PixelEnv(case_level(case)) for case in chunk]
        observations = [env.reset(case["start_x"]) for env, case in zip(envs, chunk)]
        live = list(range(len(envs)))
        while live:
            actions = model.act_batch(np.stack([observations[i] for i in live]), greedy=True)[0]
            for i, action in zip(live, actions):
                observations[i], _, _, _ = envs[i].step(action)
            if preview:
                preview.show(envs[live[0]], label)
            live = [i for i in live if not envs[i].simulator.done]
        rows.extend(episode_row(env, case) for env, case in zip(envs, chunk))
    return summary(rows), rows


def generalized_advantages(rewards, values, dones, next_values, gamma=.995, gae_lambda=.95):
    # Truncations must have their terminal value added to rewards beforehand.
    advantages = np.empty_like(rewards)
    gae = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(len(rewards))):
        alive = 1. - dones[t]
        delta = rewards[t] + gamma * next_values * alive - values[t]
        gae = delta + gamma * gae_lambda * alive * gae
        advantages[t] = gae
        next_values = values[t]
    return advantages, advantages + values


class PixelCollector:
    def __init__(self, count, seed, phase, time_penalty, preview):
        self.rng, self.phase = random.Random(seed), phase
        self.envs = [PixelEnv(time_penalty=time_penalty) for _ in range(count)]
        self.observations = np.stack([reset_training(e, self.rng, phase) for e in self.envs])
        self.preview = preview

    def collect(self, model, steps, update, gamma=.995):
        count = len(self.envs)
        pixels = np.empty((steps, count, *OBSERVATION_SHAPE), dtype=np.uint8)
        actions = np.empty((steps, count), dtype=np.int64)
        logs, values, rewards = [np.empty((steps, count), dtype=np.float32) for _ in range(3)]
        dones = np.zeros((steps, count), dtype=bool)
        episodes = wins = 0
        for t in range(steps):
            pixels[t] = self.observations
            actions[t], logs[t], values[t] = model.act_batch(self.observations)
            for i, env in enumerate(self.envs):
                obs, reward, done, info = env.step(actions[t, i])
                reward *= REWARD_SCALE
                if info["truncated"]:
                    reward += gamma * model.act_batch(obs[None], greedy=True)[2][0]
                rewards[t, i], dones[t, i] = reward, done
                if i == 0:
                    self.preview.show(env, f"PPO TRAINING | update {update} | frame {t + 1}/{steps} | sampled CNN actions")
                if done:
                    episodes += 1
                    wins += int(info["success"])
                    obs = reset_training(env, self.rng, self.phase)
                self.observations[i] = obs
        next_values = model.act_batch(self.observations, greedy=True)[2]
        advantages, returns = generalized_advantages(rewards, values, dones, next_values, gamma)
        return (pixels.reshape(-1, *OBSERVATION_SHAPE), actions.ravel(), logs.ravel(), advantages.ravel(), returns.ravel()), episodes, wins


class Demonstrations:
    def __init__(self, capacity, seed):
        self.pixels = np.empty((capacity, *OBSERVATION_SHAPE), dtype=np.uint8)
        self.probs = np.empty((capacity, 24), dtype=np.float32)
        self.values = np.empty(capacity, dtype=np.float32)
        self.size, self.seen = 0, 0
        self.rng = np.random.default_rng(seed)

    def add(self, pixels, probs, value):
        # Reservoir sampling retains examples from every demonstration stage.
        self.seen += 1
        index = self.size if self.size < len(self.pixels) else self.rng.integers(self.seen)
        if index < len(self.pixels):
            self.pixels[index], self.probs[index], self.values[index] = pixels, probs, value
            self.size = min(self.size + 1, len(self.pixels))


def collect_demonstrations(data, teacher, student, episodes, rng, phase, preview, student_probability=0.):
    env = PixelEnv()
    for episode in range(episodes):
        observation = reset_training(env, rng, phase)
        while not env.simulator.done:
            # Privileged teacher is training-only; student input is still pixels.
            probs, values, _ = teacher.forward([env.simulator.observe()])
            data.add(observation, probs[0], values[0] * REWARD_SCALE)
            use_student = rng.random() < student_probability
            action = student.action(observation) if use_student else int(probs[0].argmax())
            observation, _, _, _ = env.step(action)
            source = "mixed student/teacher" if student_probability else "teacher demonstrations"
            preview.show(env, f"WARMUP: {source} | episode {episode + 1}/{episodes} | samples {data.size}")
        if (episode + 1) % 20 == 0:
            print(f"demonstrations {episode + 1}/{episodes}, retained={data.size}, seen={data.seen}", flush=True)
    return env


def imitate(model, data, epochs, preview, env, stage, validate):
    optimizer = torch.optim.Adam(model.network.parameters(), lr=3e-4)
    for epoch in range(1, epochs + 1):
        losses = []
        order = data.rng.permutation(data.size)
        for start in range(0, data.size, 128):
            ix = order[start:start + 128]
            logits, values = model.network(model.tensor(data.pixels[ix]))
            targets = torch.as_tensor(data.probs[ix], device=model.device)
            labels = targets.argmax(dim=-1)
            weights = torch.where(labels >= 6, 1.5, 1.)
            hard = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
            soft = -(targets * logits.log_softmax(dim=-1)).sum(dim=-1)
            loss = ((.5 * hard + .5 * soft) * weights).mean()
            loss += .05 * (values - torch.as_tensor(data.values[ix], device=model.device)).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.network.parameters(), 1.)
            optimizer.step()
            losses.append(loss.item())
            preview.show(env, f"{stage}: CNN OPTIMIZATION | epoch {epoch}/{epochs} | last rollout frame (paused during updates)")
        print(f"{stage} epoch={epoch}/{epochs} loss={np.mean(losses):.4f}", flush=True)
        if epoch % 5 == 0 or epoch == epochs:
            validate(f"{stage}-{epoch}")


def compare_rows(visual, baseline):
    paired = [(v, b) for v, b in zip(visual, baseline) if v["success"] and b["success"]]
    return {"episodes": len(paired),
            "vision_seconds": float(np.mean([v["seconds"] for v, _ in paired])) if paired else None,
            "state_seconds": float(np.mean([b["seconds"] for _, b in paired])) if paired else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="new directory only; default is a timestamp under experiments/vision")
    parser.add_argument("--init-policy", type=Path, help="continue a vision checkpoint in a NEW run (fresh optimizer)")
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--from-scratch", action="store_true", help="pure pixel PPO without teacher warmup")
    parser.add_argument("--bootstrap-episodes", type=positive_int, default=160)
    parser.add_argument("--bootstrap-epochs", type=positive_int, default=20)
    parser.add_argument("--dagger-rounds", type=int, default=2)
    parser.add_argument("--dagger-episodes", type=positive_int, default=80)
    parser.add_argument("--dagger-epochs", type=positive_int, default=10)
    parser.add_argument("--demo-capacity", type=positive_int, default=24000)
    parser.add_argument("--updates", type=positive_int, default=200)
    parser.add_argument("--num-envs", type=positive_int, default=8)
    parser.add_argument("--rollout-steps", type=positive_int, default=128)
    parser.add_argument("--eval-every", type=positive_int, default=10)
    parser.add_argument("--validation-layouts", type=positive_int, default=2)
    parser.add_argument("--test-layouts", type=positive_int, default=10)
    parser.add_argument("--fixed-starts", type=positive_int, default=3)
    parser.add_argument("--phase", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--entropy", type=float, default=.005)
    parser.add_argument("--time-penalty", type=float, default=.10)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--realtime", action="store_true", help="limit previewed environment to 60 frames/sec")
    parser.add_argument("--evaluate-only", type=Path, metavar="RUN", help="re-evaluate a run against its original cases; write nothing")
    parser.add_argument("--report-dir", type=Path, help="with --evaluate-only: save comparison in a NEW separate directory")
    args = parser.parse_args()
    if args.dagger_rounds < 0 or any(not np.isfinite(v) or v < 0 for v in (args.learning_rate, args.entropy, args.time_penalty)) or args.learning_rate == 0:
        parser.error("invalid learning-rate, entropy, time-penalty or dagger-rounds")
    if args.report_dir and not args.evaluate_only:
        parser.error("--report-dir requires --evaluate-only")
    if args.no_preview:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
    if args.evaluate_only:
        manifest = json.loads((args.evaluate_only / "manifest.json").read_text(encoding="utf-8"))
        model = VisionAgent.load(args.evaluate_only / "policy.pt", args.device)
        baseline_path = resolve_saved_path(manifest["parameters"]["baseline"])
        if hashlib.sha256(baseline_path.read_bytes()).hexdigest() != manifest["baseline_sha256"]:
            parser.error("baseline hash changed; restore the original baseline for a valid comparison")
        try:
            report_dir = create_run_directory(args.report_dir) if args.report_dir else None
        except FileExistsError:
            parser.error("report directory already exists; choose a new one")
        preview = LivePreview(not args.no_preview, args.realtime)
        try:
            visual, rows = evaluate_pixels(model, manifest["test_cases"], preview, "TEST - saved CNN")
            state, baseline_rows = evaluate_state(PPO.load(baseline_path), manifest["test_cases"])
            result = {"vision": visual, "state_baseline": state, "paired_success_time": compare_rows(rows, baseline_rows),
                      "groups": summarize_groups(rows), "selected_stage": model.metadata.get("selected_stage"),
                      "source_run": str(args.evaluate_only.resolve()), "baseline_unchanged": True}
            saved_path = args.evaluate_only / "results.json"
            if saved_path.is_file():
                saved = json.loads(saved_path.read_text(encoding="utf-8"))
                result["matches_saved_results"] = visual == saved["vision"] and state == saved["state_baseline"]
            if report_dir:
                write_json(report_dir / "results.json", result)
                write_json(report_dir / "manifest.json", {"source_run": str(args.evaluate_only.resolve()),
                           "policy_sha256": hashlib.sha256((args.evaluate_only / "policy.pt").read_bytes()).hexdigest(),
                           "baseline_sha256": manifest["baseline_sha256"], "test_cases": manifest["test_cases"],
                           "geometry_audit": manifest["geometry_audit"]})
                write_csv(report_dir / "vision_episodes.csv", rows)
                write_csv(report_dir / "state_baseline_episodes.csv", baseline_rows)
            print(json.dumps(result, indent=2))
        finally:
            preview.close()
        return
    # Validate dependencies and inputs before allocating any new result folder.
    baseline = PPO.load(args.baseline)
    if baseline.metadata.get("movement_version") != MOVEMENT_VERSION or baseline.action_count != 24:
        parser.error("baseline must use the current agile-v2 movement and 24 actions")
    model = VisionAgent.load(args.init_policy, args.device, args.seed) if args.init_policy else VisionAgent(args.seed, args.device)
    try:
        output = create_run_directory(args.output_dir)
    except FileExistsError:
        parser.error("output directory already exists; choose a new one to preserve comparisons")
    validation = evaluation_cases("validation", args.validation_layouts, 1)
    testing = evaluation_cases("test", args.test_layouts, args.fixed_starts)
    baseline_hash = hashlib.sha256(args.baseline.read_bytes()).hexdigest()
    manifest = {"format": "vision-training-v1", "movement_version": MOVEMENT_VERSION,
                "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "baseline_sha256": baseline_hash, "torch_version": str(torch.__version__), "device": str(model.device),
                "observation": model.metadata, "actions": MOVEMENT_ACTIONS, "reward_scale": REWARD_SCALE,
                "validation_cases": validation, "test_cases": testing, "geometry_audit": audit_layout_splits(validation, testing),
                "teacher_usage": "none" if args.from_scratch or args.init_policy else "training-only behavior cloning and DAgger; no teacher in PPO, validation or playback",
                "selection": "validation completion, then failure-penalized time; test evaluated only after selection",
                "init_policy_sha256": hashlib.sha256(args.init_policy.read_bytes()).hexdigest() if args.init_policy else None}
    write_json(output / "manifest.json", manifest)
    model.save(output / "initial_policy.pt")
    model.save(output / "policy.pt", selected_stage="initial")
    preview = LivePreview(not args.no_preview, args.realtime)
    history, best, selected = [], (-1.,), "initial"
    started = time.monotonic()
    environment_steps = 0
    print(f"VISION RUN: {output.resolve()} | device={model.device}", flush=True)

    def validate(stage):
        nonlocal best, selected
        result, _ = evaluate_pixels(model, validation, preview)
        row = {"stage": stage, "environment_steps": environment_steps, **result}
        history.append(row)
        if selection_score(result) > best:
            best, selected = selection_score(result), stage
            model.save(output / "policy.pt", selected_stage=stage)
        model.save(output / "last_policy.pt", stage=stage)
        write_csv(output / "learning_curve.csv", history)
        preview.snapshot(output / "training_preview.png")
        write_json(output / "status.json", {"status": "training", "stage": stage, "selected_stage": selected, **result})
        print(f"{stage} | validation {result['successes']}/{result['episodes']} | time={result['mean_success_seconds']} | dashes={result['mean_dashes']:.2f}", flush=True)

    try:
        if not args.from_scratch and not args.init_policy:
            data = Demonstrations(args.demo_capacity, args.seed)
            rng = random.Random(args.seed)
            env = collect_demonstrations(data, baseline, model, args.bootstrap_episodes, rng, args.phase, preview)
            imitate(model, data, args.bootstrap_epochs, preview, env, "BC", validate)
            for round_index in range(args.dagger_rounds):
                env = collect_demonstrations(data, baseline, model, args.dagger_episodes, rng, args.phase, preview, student_probability=.5 + .25 * round_index / max(1, args.dagger_rounds - 1))
                imitate(model, data, args.dagger_epochs, preview, env, f"DAgger-{round_index + 1}", validate)
            write_json(output / "demonstrations.json", {"seen_frames": data.seen, "retained_frames": data.size,
                       "teacher": str(args.baseline), "teacher_sha256": baseline_hash, "scope": "train layouts only; pixel observations and teacher action/value labels"})
            del data
            model = VisionAgent.load(output / "policy.pt", args.device, args.seed)
        else:
            validate("initial")
        optimizer = torch.optim.Adam(model.network.parameters(), lr=args.learning_rate)
        collector = PixelCollector(args.num_envs, args.seed + 1, args.phase, args.time_penalty, preview)
        for update in range(1, args.updates + 1):
            samples, completed, wins = collector.collect(model, args.rollout_steps, update)
            environment_steps += args.num_envs * args.rollout_steps
            diagnostics = ppo_update(model, optimizer, samples, entropy_coef=args.entropy)
            del samples
            if update % args.eval_every == 0 or update == args.updates:
                print(f"PPO {update}/{args.updates} | episodes {wins}/{completed} | {diagnostics}", flush=True)
                validate(f"PPO-{update}")
        model = VisionAgent.load(output / "policy.pt", args.device, args.seed)
        visual, rows = evaluate_pixels(model, testing, preview, "FINAL TEST - selected CNN")
        state, baseline_rows = evaluate_state(baseline, testing)
        baseline_unchanged = hashlib.sha256(args.baseline.read_bytes()).hexdigest() == baseline_hash
        result = {"vision": visual, "state_baseline": state, "paired_success_time": compare_rows(rows, baseline_rows),
                  "groups": summarize_groups(rows), "selected_stage": selected, "environment_steps": environment_steps,
                  "elapsed_seconds": round(time.monotonic() - started, 2), "baseline_unchanged": baseline_unchanged,
                  "scope": "CNN receives screenshots only. Teacher uses state during optional warmup. Reward/termination come from simulator. Fixed layouts are trained; different seeds can share geometry, see manifest audit."}
        write_json(output / "results.json", result)
        write_csv(output / "vision_episodes.csv", rows)
        write_csv(output / "state_baseline_episodes.csv", baseline_rows)
        write_json(output / "status.json", {"status": "complete", **result})
        print(json.dumps(result, indent=2), flush=True)
    except (StopTraining, KeyboardInterrupt):
        model.save(output / "last_policy.pt", stage="interrupted")
        preview.snapshot(output / "training_preview.png")
        write_json(output / "status.json", {"status": "interrupted", "selected_stage": selected, "environment_steps": environment_steps})
        print(f"Stopped safely; selected and latest checkpoints saved separately in {output}", flush=True)
    finally:
        preview.close()


if __name__ == "__main__":
    main()
