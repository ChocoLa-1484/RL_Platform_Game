"""Replay saved policies on the frozen test suite and explain each death.

Every episode is replayed greedily with exactly the evaluation cases, so the
set of failures matches the saved episode CSVs. For each death we look at the
last take-off (including any dash in the run-up on the source platform) and
where the player fell relative to the next platform. For every rightward
climb to a higher platform we also record the feet clearance above that
platform's top at the first frame the player overlaps it horizontally.
"""

from collections import Counter

import numpy as np

from rl_platformer.envs.movement import MovementEnv, decode_action
from rl_platformer.envs.platformer import PLAYER_H, PLAYER_W
from rl_platformer.training.state import case_level

TAKEOFFS = ("jump", "ground_dash", "walk_off")
OUTCOMES = ("overshoot", "short", "under_ledge")
DIRECTIONS = {(0, -1): "up", (1, -1): "up-right", (1, 0): "right", (1, 1): "down-right", (0, 1): "down",
              (-1, 1): "down-left", (-1, 0): "left", (-1, -1): "up-left"}


def _sign(value):
    return 0 if abs(value) < 1e-9 else (1 if value > 0 else -1)


def step_and_record(sim, step, action):
    """Advance one frame through ``step`` and return the frame record of the simulator ``sim``."""
    was_grounded, platform, facing, dashes = sim.grounded, sim.current_platform, sim.facing, sim.dash_count
    step(int(action))
    _, jump, _, direction = decode_action(int(action), facing)
    started = sim.dash_count > dashes
    return {"step": sim.steps, "x": sim.x, "y": sim.y, "vx": sim.vx, "vy": sim.vy,
            "grounded": sim.grounded, "was_grounded": was_grounded, "platform": platform,
            "action": int(action), "jump": bool(jump), "dash_started": started,
            "dash_dir": DIRECTIONS[(_sign(direction[0]), _sign(direction[1]))] if started else None}


def replay_state(model, cases):
    """Greedy replay of a NumPy state policy; returns [(simulator, frames)]."""
    episodes = []
    for case in cases:
        env = MovementEnv(case_level(case))
        env.reset(case["start_x"])
        frames = []
        while not env.done:
            action = model.act_batch([env.observe()], greedy=True)[0][0]
            frames.append(step_and_record(env, env.step, action))
        episodes.append((env, frames))
    return episodes


def replay_vision(model, cases, batch_size=12):
    """Greedy replay of a pixel policy, batched like evaluate_pixels; returns [(simulator, frames)]."""
    from rl_platformer.envs.vision import PixelEnv

    episodes = []
    for start in range(0, len(cases), batch_size):
        chunk = cases[start:start + batch_size]
        envs = [PixelEnv(case_level(case)) for case in chunk]
        observations = [env.reset(case["start_x"]) for env, case in zip(envs, chunk)]
        frames = [[] for _ in envs]
        live = list(range(len(envs)))
        while live:
            actions = model.act_batch(np.stack([observations[i] for i in live]), greedy=True)[0]
            for i, action in zip(live, actions):
                result = {}

                def step(a, i=i):
                    result["obs"] = envs[i].step(a)[0]
                frames[i].append(step_and_record(envs[i].simulator, step, action))
                observations[i] = result["obs"]
            live = [i for i in live if not envs[i].simulator.done]
        episodes.extend((env.simulator, f) for env, f in zip(envs, frames))
    return episodes


def _last_takeoff(frames):
    takeoff = max((i for i, f in enumerate(frames) if f["was_grounded"]), default=0)
    source = frames[takeoff]["platform"] or 0
    start = takeoff
    while start > 0 and frames[start - 1]["was_grounded"] and frames[start - 1]["platform"] == source:
        start -= 1
    return takeoff, start, source


def classify_death(sim, frames):
    """Explain a death by the last take-off and where the player passed the next platform's top."""
    platforms = sim.level.platforms
    takeoff, runup_start, source = _last_takeoff(frames)
    flight, approach = frames[takeoff:], frames[runup_start:]
    if flight[0]["jump"] and flight[0]["was_grounded"]:
        how = "jump"
    elif any(f["dash_started"] for f in frames[runup_start:takeoff + 1]):
        how = "ground_dash"
    else:
        how = "walk_off"
    target_index = min(source + 1, len(platforms) - 1)
    target, origin = platforms[target_index], platforms[source]
    crossing = next((f for f in flight if f["vy"] > 0 and f["y"] + PLAYER_H > target.y + 1), flight[-1])
    if crossing["x"] + PLAYER_W < target.x:
        outcome, miss = "short", target.x - (crossing["x"] + PLAYER_W)
    elif crossing["x"] > target.x + target.width:
        outcome, miss = "overshoot", crossing["x"] - (target.x + target.width)
    else:
        outcome, miss = "under_ledge", 0.
    toward = 1 if target.x >= origin.x else -1
    apex_feet = min(f["y"] for f in flight) + PLAYER_H
    return {"source_platform": source, "target_platform": target_index, "platform_count": len(platforms),
            "route_position": source / max(1, len(platforms) - 1),
            "takeoff": how, "outcome": outcome, "horizontal_miss_px": round(float(miss), 2),
            "dashes": "+".join(f["dash_dir"] for f in approach if f["dash_started"]) or "none",
            "dash_left_at_death": bool(sim.dash_energy >= 1 and not sim.air_dash_used),
            "braked_in_air": any(decode_action(f["action"], 1)[0] == -toward for f in flight),
            "gap_px": target.x - (origin.x + origin.width), "rise_px": origin.y - target.y,
            "target_width_px": target.width, "apex_above_target_px": round(float(target.y - apex_feet), 2),
            "vx_at_crossing": round(float(crossing["vx"]), 2), "takeoff_frame": takeoff}


def climbs(sim, frames):
    """Each airborne attempt to reach a higher next platform, with feet clearance on arrival."""
    platforms = sim.level.platforms
    records, i = [], 0
    while i < len(frames):
        if not frames[i]["was_grounded"] or frames[i]["grounded"]:
            i += 1
            continue
        source = frames[i]["platform"]
        j = i
        while j + 1 < len(frames) and not frames[j + 1]["was_grounded"]:
            j += 1
        k = i
        while k > 0 and frames[k - 1]["was_grounded"] and frames[k - 1]["platform"] == source:
            k -= 1
        if source is not None and source + 1 < len(platforms) and platforms[source + 1].y < platforms[source].y:
            target = platforms[source + 1]
            arrive = next((f for f in frames[i:j + 1] if f["x"] + PLAYER_W > target.x), None)
            if arrive is not None:
                records.append({"source_platform": source,
                                "clearance_px": round(float(target.y - (arrive["y"] + PLAYER_H)), 2),
                                "horizontal_dash": any(f["dash_started"] and f["dash_dir"] in ("left", "right")
                                                       for f in frames[k:j + 1]),
                                "landed": bool(frames[j]["grounded"])})
        i = j + 1
    return records


def analyze(episodes, cases):
    """Rows for failed episodes, climb records, compact trajectories of deaths, and a summary."""
    failures, climb_rows, trajectories = [], [], {}
    for case, (sim, frames) in zip(cases, episodes):
        key = {"family": case["family"], "stage": case["stage"], "layout_seed": case["layout_seed"]}
        climb_rows.extend({**key, "episode_success": sim.success, **c} for c in climbs(sim, frames))
        if sim.success:
            continue
        row = {**key, "death": bool(sim.deaths), "timeout": not sim.deaths, "steps": sim.steps}
        if sim.deaths:
            row.update(classify_death(sim, frames))
            trajectories[str(case["layout_seed"])] = {
                "case": key, "platforms": [[p.x, p.y, p.width] for p in sim.level.platforms],
                "frames": [{"x": round(f["x"], 2), "y": round(f["y"], 2), "vy": round(f["vy"], 2),
                            **{k: f[k] for k in ("grounded", "dash_started", "dash_dir")}} for f in frames]}
        failures.append(row)
    return failures, climb_rows, trajectories, summarize(failures, climb_rows, len(cases))


def summarize(failures, climb_rows, episodes):
    deaths = [r for r in failures if r["death"]]

    def median(values):
        return round(float(np.median(values)), 2) if values else None

    dashed = [c for c in climb_rows if c["horizontal_dash"]]
    landed = [c["clearance_px"] for c in dashed if c["landed"]]
    return {
        "episodes": episodes, "failures": len(failures), "deaths": len(deaths),
        "timeouts": sum(r["timeout"] for r in failures),
        "takeoff": {k: sum(r["takeoff"] == k for r in deaths) for k in TAKEOFFS},
        "outcome": {k: sum(r["outcome"] == k for r in deaths) for k in OUTCOMES},
        "takeoff_outcome": {f"{a}/{b}": n for (a, b), n in Counter((r["takeoff"], r["outcome"]) for r in deaths).most_common()},
        "median_horizontal_miss_px": {k: median([r["horizontal_miss_px"] for r in deaths if r["outcome"] == k])
                                      for k in ("short", "overshoot")},
        "died_in_first_two_gaps": sum(r["source_platform"] <= 1 for r in deaths),
        "median_route_position": median([r["route_position"] for r in deaths]),
        "dash_left_at_death": sum(r["dash_left_at_death"] for r in deaths),
        "braked_in_air": sum(r["braked_in_air"] for r in deaths),
        "climbs": {"total": len(climb_rows), "with_horizontal_dash": len(dashed),
                   "landed_after_horizontal_dash": len(landed),
                   "missed_after_horizontal_dash": len(dashed) - len(landed),
                   "landed_clearance_median_px": median(landed),
                   "landed_clearance_under_3px": round(float(np.mean(np.array(landed) < 3)), 4) if landed else None,
                   "landed_clearance_under_5px": round(float(np.mean(np.array(landed) < 5)), 4) if landed else None,
                   "missed_clearance_px": sorted(c["clearance_px"] for c in dashed if not c["landed"])},
        "scope": "Greedy replay of the saved policy on the frozen test cases. Categories describe the final "
                 "take-off and where the player passed the next platform's top; clearance is meaningful for "
                 "horizontal-dash approaches, where the player is no longer rising.",
    }
