"""Failure analysis of saved policies on the frozen 960-case test suite (writes a NEW directory)."""

import argparse
import csv
import json
from pathlib import Path

from rl_platformer.paths import HW3_STATE_POLICY, ROOT
from rl_platformer.training.state import evaluation_cases
from rl_platformer.utils.io import write_csv, write_json

DEFAULT_VISION_POLICY = ROOT / "experiments" / "vision" / "20261020_134343_080697" / "policy.pt"
SAVED_EPISODES = {"state": ROOT / "experiments" / "state" / "dash_speedrun" / "test_episodes.csv",
                  "vision": ROOT / "experiments" / "vision" / "eval_960_hw3_suite" / "vision_episodes.csv"}


def saved_failures(path):
    with Path(path).open(encoding="utf-8") as stream:
        return sorted((r["family"], int(r["stage"]), int(r["layout_seed"]))
                      for r in csv.DictReader(stream) if r["success"] != "True")


def run_analysis(args):
    from rl_platformer.analysis.failures import analyze, replay_state, replay_vision

    cases = evaluation_cases("test", args.test_layouts, args.fixed_starts)
    if args.agent == "state":
        from rl_platformer.agents.ppo import PPO
        policy = args.policy or HW3_STATE_POLICY
        episodes = replay_state(PPO.load(policy), cases)
    else:
        from rl_platformer.agents.vision import VisionAgent
        policy = args.policy or DEFAULT_VISION_POLICY
        episodes = replay_vision(VisionAgent.load(policy, args.device), cases)
    failures, climb_rows, trajectories, summary = analyze(episodes, cases)
    reference = args.saved_episodes or (SAVED_EPISODES[args.agent] if args.policy is None else None)
    if reference and args.test_layouts == 100 and args.fixed_starts == 20:
        summary["matches_saved_failures"] = saved_failures(reference) == sorted(
            (r["family"], r["stage"], r["layout_seed"]) for r in failures)
        summary["saved_episodes"] = str(Path(reference).resolve())
    summary.update({"agent": args.agent, "policy": str(Path(policy).resolve()),
                    "test_layouts": args.test_layouts, "fixed_starts": args.fixed_starts})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "summary.json", summary)
    write_csv(args.output_dir / "failures.csv", failures)
    write_csv(args.output_dir / "climbs.csv", climb_rows)
    write_json(args.output_dir / "trajectories.json", trajectories)
    print(json.dumps({k: summary[k] for k in ("failures", "deaths", "outcome", "takeoff", "climbs")}, indent=2))
    if "matches_saved_failures" in summary:
        print("matches saved failures:", summary["matches_saved_failures"])


def pick_examples(state_failures, vision_failures):
    """Deterministic, typical examples: the state climb missed by the smallest margin on a height map,
    and the CNN 'short' death in the first two gaps whose miss is closest to the median miss."""
    climbs = [r for r in state_failures if r.get("outcome") == "under_ledge" and r["family"] == "height"]
    state = min(climbs, key=lambda r: (abs(float(r["apex_above_target_px"])), int(r["layout_seed"])))
    shorts = [r for r in vision_failures if r.get("outcome") == "short"]
    median = sorted(float(r["horizontal_miss_px"]) for r in shorts)[len(shorts) // 2]
    early = [r for r in shorts if int(r["source_platform"]) <= 1]
    vision = min(early, key=lambda r: (abs(float(r["horizontal_miss_px"]) - median), int(r["layout_seed"])))
    return state, vision


def run_figures(args):
    from rl_platformer.analysis import figures

    state_dir, vision_dir = args.figures
    state, state_climbs, state_traj, state_failures = figures.load(state_dir)
    vision, _, vision_traj, vision_failures = figures.load(vision_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    figures.outcome_shares(state, vision, args.output_dir / "fig_failure_outcomes.png")
    figures.climb_clearance(state_climbs, args.output_dir / "fig_climb_clearance.png")
    s, v = pick_examples(state_failures, vision_failures)
    figures.example_trajectories(
        state_traj[s["layout_seed"]], s, vision_traj[v["layout_seed"]], v,
        f"height {s['stage']}, seed {s['layout_seed']}: misses the ledge on arrival",
        f"{v['family']} {v['stage']}, seed {v['layout_seed']}: {float(v['horizontal_miss_px']):.0f} px short",
        args.output_dir / "fig_failure_examples.png")
    write_json(args.output_dir / "figures.json", {"state_analysis": str(Path(state_dir).resolve()),
                                                  "vision_analysis": str(Path(vision_dir).resolve()),
                                                  "state_example": s, "vision_example": v})
    print("figures written to", args.output_dir.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=("state", "vision"), help="analyze this agent's failures")
    parser.add_argument("--policy", type=Path, help="default: the HW3 state policy (experiments/state/dash_speedrun) or the saved CNN run")
    parser.add_argument("--saved-episodes", type=Path, help="episode CSV whose failures must match the replay")
    parser.add_argument("--test-layouts", type=int, default=100)
    parser.add_argument("--fixed-starts", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--figures", nargs=2, type=Path, metavar=("STATE_DIR", "VISION_DIR"),
                        help="draw report figures from two analysis directories instead")
    parser.add_argument("--output-dir", type=Path, required=True, help="must not exist")
    args = parser.parse_args()
    if bool(args.agent) == bool(args.figures):
        parser.error("give exactly one of --agent or --figures")
    if args.output_dir.exists():
        parser.error("output directory already exists; choose a new one")
    run_figures(args) if args.figures else run_analysis(args)
