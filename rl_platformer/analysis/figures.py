"""Report figures for the failure analysis (requires matplotlib; see requirements/analysis.txt).

Colors follow a validated categorical palette: slot 1 (blue) is always the
state agent and slot 2 (orange) always the CNN, in every figure.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from rl_platformer.envs.platformer import PLAYER_H, PLAYER_W  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
AGENT_COLORS = {"state": "#2a78d6", "vision": "#eb6834"}
AGENT_NAMES = {"state": "State PPO", "vision": "Pixel CNN"}
OUTCOME_NAMES = {"overshoot": "Overshot next\nplatform", "short": "Fell short of\nnext platform",
                 "under_ledge": "Below ledge\n(climb missed)"}


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "xtick.color": MUTED,
        "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True,
        "legend.frameon": False, "font.family": "DejaVu Sans"})


def load(directory):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with (directory / "climbs.csv").open(encoding="utf-8") as stream:
        climbs = list(csv.DictReader(stream))
    trajectories = json.loads((directory / "trajectories.json").read_text(encoding="utf-8"))
    with (directory / "failures.csv").open(encoding="utf-8") as stream:
        failures = list(csv.DictReader(stream))
    return summary, climbs, trajectories, failures


def outcome_shares(state, vision, path):
    """Fig. A: share of deaths by where the player passed the next platform, per agent."""
    _style()
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    outcomes, width = list(OUTCOME_NAMES), .38
    for offset, (agent, summary) in zip((-width / 2 - .01, width / 2 + .01), (("state", state), ("vision", vision))):
        shares = [100 * summary["outcome"][o] / summary["deaths"] for o in outcomes]
        bars = ax.bar([i + offset for i in range(len(outcomes))], shares, width, color=AGENT_COLORS[agent],
                      label=f"{AGENT_NAMES[agent]} ({summary['deaths']} deaths)")
        for bar, share, outcome in zip(bars, shares, outcomes):
            ax.text(bar.get_x() + bar.get_width() / 2, share + 1.2, f"{summary['outcome'][outcome]}",
                    ha="center", va="bottom", fontsize=7, color=INK_2)
    ax.set_xticks(range(len(outcomes)), [OUTCOME_NAMES[o] for o in outcomes], fontsize=7.5)
    ax.set_ylabel("Share of deaths (%)")
    ax.set_ylim(0, 70)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def climb_clearance(climbs, path, agent="state"):
    """Fig. B: feet clearance above the ledge after a horizontal-dash climb; landed vs missed."""
    _style()
    dashed = [r for r in climbs if r["horizontal_dash"] == "True"]
    landed = [float(r["clearance_px"]) for r in dashed if r["landed"] == "True"]
    missed = [float(r["clearance_px"]) for r in dashed if r["landed"] != "True"]
    bins = list(range(-16, 33))
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(3.4, 2.9), sharex=True,
                                      gridspec_kw={"height_ratios": [2.2, 1], "hspace": .15})
    top.hist(landed, bins=bins, color=AGENT_COLORS[agent], edgecolor=SURFACE, linewidth=.6)
    bottom.hist(missed, bins=bins, color=INK_2, edgecolor=SURFACE, linewidth=.6)
    for ax, label in ((top, f"Landed (n={len(landed):,})"), (bottom, f"Missed (n={len(missed)})")):
        ax.axvline(0, color=INK, linewidth=1, linestyle=(0, (3, 2)))
        ax.text(.98, .9, label, transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK_2)
        ax.set_ylabel("Climbs")
        ax.grid(axis="x", visible=False)
    under3 = sum(c < 3 for c in landed) / len(landed)
    top.text(.98, .72, f"median {sorted(landed)[len(landed) // 2]:.0f} px\n{under3:.0%} under 3 px",
             transform=top.transAxes, ha="right", va="top", fontsize=7, color=MUTED)
    bottom.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=3))
    bottom.set_xlabel("Feet clearance above ledge on arrival (px)")
    bottom.set_xlim(-16, 32)
    fig.subplots_adjust(left=.16, right=.97, top=.97, bottom=.15)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _critical_frame(record, row):
    """Frame where the miss is decided: arrival over the ledge (climb) or passing the target's top (short)."""
    frames, target = record["frames"], record["platforms"][int(row["target_platform"])]
    takeoff = int(row["takeoff_frame"])
    flight = range(takeoff, len(frames))
    if row["outcome"] == "under_ledge":
        return next(i for i in flight if frames[i]["x"] + PLAYER_W > target[0])
    return next((i for i in flight if frames[i]["vy"] > 0 and frames[i]["y"] + PLAYER_H > target[1] + 1), len(frames) - 1)


def _window(record, row):
    platforms = record["platforms"]
    source, target = platforms[int(row["source_platform"])], platforms[int(row["target_platform"])]
    frames = record["frames"][max(0, int(row["takeoff_frame"]) - 10):]
    left, right = source[0] + source[2] - 90, target[0] + target[2] + 50
    feet = [f["y"] + PLAYER_H for f in frames if left <= f["x"] <= right]
    return left, right, min(feet + [target[1], source[1]]) - 45, max(target[1], source[1]) + 60


def _trajectory(ax, record, row, agent, title, note, size):
    platforms, frames = record["platforms"], record["frames"]
    target = platforms[int(row["target_platform"])]
    left, right, top, bottom = _window(record, row)
    width, height = size
    cx, cy = (left + right) / 2, (top + bottom) / 2
    left, right, top, bottom = cx - width / 2, cx + width / 2, cy - height / 2, cy + height / 2
    for x, y, w in platforms:
        ax.add_patch(Rectangle((x, y), w, 9, color=INK_2, linewidth=0))
    lo = max(0, int(row["takeoff_frame"]) - 14)
    path = [(f["x"] + PLAYER_W / 2, f["y"] + PLAYER_H) for f in frames[lo:]]
    path = [pt for pt in path if pt[1] <= target[1] + 45]
    ax.plot([q[0] for q in path], [q[1] for q in path], color=AGENT_COLORS[agent], linewidth=2, label="feet path")
    dashes = [(f["x"] + PLAYER_W / 2, f["y"] + PLAYER_H) for f in frames[lo:]
              if f["dash_started"] and f["y"] + PLAYER_H <= target[1] + 45]
    ax.scatter([d[0] for d in dashes], [d[1] for d in dashes], s=40, color=AGENT_COLORS[agent],
               edgecolor=SURFACE, linewidth=1.5, zorder=4, label="dash start")
    key = frames[_critical_frame(record, row)]
    ax.add_patch(Rectangle((key["x"], key["y"]), PLAYER_W, PLAYER_H, fill=False,
                           edgecolor=AGENT_COLORS[agent], linewidth=1.2, zorder=3))
    if row["outcome"] == "under_ledge":
        gap = target[1] - (key["y"] + PLAYER_H)
        ax.annotate(f"arrives {abs(gap):.1f} px below\nthe ledge top, passes under", xy=(target[0], target[1] + 2),
                    xytext=(target[0] + target[2] / 2, target[1] + 42), fontsize=7, color=INK, ha="center",
                    va="top", arrowprops={"arrowstyle": "-|>", "color": INK, "linewidth": .8})
    else:
        y = target[1] - 7
        ax.annotate("", xy=(target[0], y), xytext=(key["x"] + PLAYER_W, y),
                    arrowprops={"arrowstyle": "<|-|>", "color": INK, "linewidth": .8, "shrinkA": 0, "shrinkB": 0})
        ax.text(target[0] + 4, y - 8, f"{float(row['horizontal_miss_px']):.0f} px short",
                fontsize=7, color=INK, ha="left", va="bottom")
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8, color=INK, loc="left")
    ax.text(.01, .02, note, transform=ax.transAxes, fontsize=6.5, color=MUTED, va="bottom")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def example_trajectories(state_record, state_row, vision_record, vision_row, state_note, vision_note, path):
    """Fig. C: one typical death per agent at the same scale; line = feet path, dots = dash starts,
    outline = player at the frame that decides the miss, bars = platforms."""
    _style()
    windows = [_window(state_record, state_row), _window(vision_record, vision_row)]
    size = (max(w[1] - w[0] for w in windows), max(w[3] - w[2] for w in windows))
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.0))  # stacked to fit one paper column
    _trajectory(axes[0], state_record, state_row, "state", "State PPO: horizontal dash at ledge height", state_note, size)
    _trajectory(axes[1], vision_record, vision_row, "vision", "Pixel CNN: low jump + dash falls short", vision_note, size)
    axes[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, .93), handletextpad=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
