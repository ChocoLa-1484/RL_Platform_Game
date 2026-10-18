"""Export the full curriculum's recorded learning curve and fixed level maps."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from platformer import LEVELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=Path(__file__).resolve().parent / "experiments" / "full_version")
    args = parser.parse_args()
    result = json.loads((args.experiment_dir / "results.json").read_text(encoding="utf-8"))
    with (args.experiment_dir / "learning_curve.csv").open(encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    colors = ("#3c88bd", "#309e86", "#cf9835", "#af5b87")
    for stage, color in zip(range(1, 5), colors):
        rows = [r for r in history if int(r["stage"]) == stage]
        axes[0].plot([int(r["environment_steps"]) / 1e6 for r in rows],
                     [float(r["current_stage_completion"]) * 100 for r in rows],
                     color=color, marker=".", label=f"Stage {stage}")
    axes[0].set(xlabel="Additional training steps (millions)", ylabel="Validation completion (%)",
                title="Active-stage validation during training", ylim=(-3, 105))
    axes[0].legend(ncol=2, loc="lower left")
    axes[0].grid(alpha=0.18)
    test = result["test"]
    rates = [r["completion_rate"] * 100 for r in test]
    intervals = [[max(0, rates[i] - r["completion_wilson95"][0] * 100) for i, r in enumerate(test)],
                 [max(0, r["completion_wilson95"][1] * 100 - rates[i]) for i, r in enumerate(test)]]
    axes[1].bar(range(1, 5), rates, color=colors, width=0.6, yerr=intervals, capsize=5)
    for stage, rate in enumerate(rates, 1):
        axes[1].text(stage, test[stage - 1]["completion_wilson95"][1] * 100 + 3, f"{rate:.0f}/100", ha="center")
    axes[1].set(xticks=range(1, 5), xticklabels=["S1: flat", "S2: heights", "S3: narrow", "S4: long"],
                ylabel="Unseen-layout completion (%)", ylim=(0, 114),
                title="Frozen model: 100 test layouts per stage")
    fig.suptitle("Full curriculum | one warm-start run (seed 42) | bars: 95% Wilson intervals")
    fig.savefig(args.experiment_dir / "comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(5, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    for ax, level in zip(axes, LEVELS[5:]):
        for index, platform in enumerate(level.platforms):
            ax.add_patch(Rectangle((platform.x, platform.y), platform.width, 540 - platform.y,
                                   facecolor="#309e86" if index else "#3c88bd"))
        ax.plot(level.goal_x, level.platforms[-1].y - 25, marker="v", color="#cf9835")
        ax.set(xlim=(-20, 2640), ylim=(540, 240), yticks=[300, 450], ylabel="Y (px)")
        ax.set_title(f"{level.name}  |  {level.world_width:g} px  |  {len(level.platforms)} platforms", loc="left", fontsize=11)
        ax.grid(axis="x", alpha=0.15)
    axes[-1].set_xlabel("World X (px); viewport width = 900 px")
    fig.savefig(args.experiment_dir / "level_maps.png", dpi=160)
    plt.close(fig)
    print(f"Saved comparison.png and level_maps.png in {args.experiment_dir}")


if __name__ == "__main__":
    main()
