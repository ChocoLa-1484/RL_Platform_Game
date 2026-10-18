"""Create a static research figure from the saved experiment reports (no training)."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {"ppo_shaped": "PPO + shaping", "ppo_auto_jump": "Auto-jump PPO", "ppo_sparse": "Sparse-reward PPO"}
COLORS = {"ppo_shaped": "#168c81", "ppo_auto_jump": "#d28630", "ppo_sparse": "#7a63b8"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=Path(__file__).resolve().parent / "experiments" / "easy_pdf")
    args = parser.parse_args()
    report = json.loads((args.experiment_dir / "summary.json").read_text(encoding="utf-8"))
    variants = list(dict.fromkeys(run["variant"] for run in report["runs"]))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), constrained_layout=True)
    fig.suptitle("Precision platformer: procedural layouts and PPO ablations", fontsize=18, weight="bold")
    for stage in (1, 2, 3):
        ax = axes[0, stage - 1]
        for variant in variants:
            curves = []
            for run in report["runs"]:
                if run["variant"] != variant:
                    continue
                path = args.experiment_dir / variant / f"seed{run['training_seed']}" / "learning_curve.csv"
                with path.open(encoding="utf-8") as stream:
                    rows = [row for row in csv.DictReader(stream) if int(row["stage"]) == stage]
                x = np.array([float(row["stage_environment_steps"]) / 1000 for row in rows])
                curves.append([float(row["active_stage_completion"]) * 100 for row in rows])
            mean = np.mean(curves, axis=0)
            std = np.std(curves, axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
            ax.plot(x, mean, color=COLORS[variant], label=LABELS[variant], linewidth=2)
            ax.fill_between(x, np.maximum(0, mean - std), np.minimum(100, mean + std), color=COLORS[variant], alpha=0.12)
        ax.set(title=f"Stage {stage}: validation learning curve", xlabel="Environment steps in this stage (thousands)",
               ylabel="Completion (%)", ylim=(-3, 103))
        ax.grid(alpha=0.2)
    for ax, metric, title in zip(axes[1], ("completion_rate", "death_rate", "mean_platform_coverage"),
                                 ("Held-out completion", "Held-out deaths", "Held-out platform coverage")):
        width = 0.24
        for index, variant in enumerate(variants):
            groups = [g for g in report["aggregate"] if g["variant"] == variant]
            values = [g[metric + "_mean"] * 100 for g in groups]
            centers = np.arange(1, 4) + (index - (len(variants) - 1) / 2) * width
            ax.bar(centers, values, width, color=COLORS[variant], label=LABELS[variant])
            variant_runs = [run for run in report["runs"] if run["variant"] == variant]
            for run_index, run in enumerate(variant_runs):
                offset = (run_index - (len(variant_runs) - 1) / 2) * 0.035
                ax.scatter(centers + offset, [m[metric] * 100 for m in run["test"]],
                           s=22, facecolors="white", edgecolors="#222222", linewidths=0.8, zorder=3)
        ax.set(title=title, xticks=[1, 2, 3], xticklabels=["Stage 1", "Stage 2", "Stage 3"], ylabel="Percent", ylim=(-3, 105))
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False,
               title="Lines/bars: means. Bands: ±1 sample SD (clipped to 0–100%). Dots: individual training seeds.")
    destination = args.experiment_dir / "comparison.png"
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    print(destination)


if __name__ == "__main__":
    main()
