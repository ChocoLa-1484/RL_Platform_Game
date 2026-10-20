"""Compare frozen movement policies and a dash-disabled ablation on identical cases."""

import argparse
import hashlib
from pathlib import Path
from rl_platformer.paths import BASELINE_STATE_POLICY

from rl_platformer.envs.movement import MOVEMENT_VERSION, MovementEnv
from rl_platformer.envs.platformer import FPS, LEVELS
from rl_platformer.agents.ppo import PPO, np
from rl_platformer.utils.io import write_json
from rl_platformer.training.state import audit_layout_splits, evaluate, evaluation_cases


def paired_times(before, after):
    paired = [(a, b) for a, b in zip(before, after) if a["success"] and b["success"]]
    if not paired:
        return {"episodes": 0, "before_seconds": None, "after_seconds": None, "time_saved_fraction": None}
    old = float(np.mean([a["seconds"] for a, _ in paired]))
    new = float(np.mean([b["seconds"] for _, b in paired]))
    return {"episodes": len(paired), "before_seconds": old, "after_seconds": new,
            "time_saved_fraction": 1 - new / old}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--baseline", type=Path, default=BASELINE_STATE_POLICY)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cases = evaluation_cases("test", 100, 20)
    model, baseline = PPO.load(args.policy), PPO.load(args.baseline)
    after, after_rows = evaluate(model, cases)
    before, before_rows = evaluate(baseline, cases)
    disabled, disabled_rows = evaluate(model, cases, allow_dash=False)
    fixed = []
    for number in range(1, len(LEVELS) + 1):
        row = {"level": number}
        for name, actor in (("baseline", baseline), ("trained", model)):
            env = MovementEnv(number)
            while not env.done:
                env.step(actor.action(env.observe(), greedy=True)[0])
            row[name] = {"success": env.success, "steps": env.steps, "seconds": env.steps / FPS, "dashes": env.dash_count}
        fixed.append(row)
    report = {"movement_version": MOVEMENT_VERSION,
              "policy": str(args.policy), "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
              "baseline_policy": str(args.baseline), "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
              "baseline": before, "trained": after, "dash_disabled": disabled,
              "paired_baseline_to_trained": paired_times(before_rows, after_rows),
              "paired_disabled_to_enabled": paired_times(disabled_rows, after_rows),
              "fixed_default_starts": fixed,
              "geometry_audit": audit_layout_splits(evaluation_cases("validation", 12), cases),
              "scope": "Frozen policies on the same v2 physics and test cases. Disabling dash masks actions 6-23 without retraining. Fixed maps are trained; one procedural test geometry matches the training pool. No test-based model selection."}
    write_json(args.output, report)
    print({key: report[key] for key in ("baseline", "trained", "dash_disabled", "paired_baseline_to_trained")})


if __name__ == "__main__":
    main()
