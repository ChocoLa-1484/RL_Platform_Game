"""Reproducible climbing layouts with separate training/validation/test pools."""

from dataclasses import asdict
import hashlib
import json
import random

from rl_platformer.envs.platformer import ascent_level
from rl_platformer.levels.procedural import validate_layout
from rl_platformer.paths import START_RANGE

GENERATOR_VERSION = "height-v1"
STAGES = (1, 2, 3)
TRAIN_LAYOUTS = 1000
SPLIT_OFFSETS = {"train": 6_000_000, "validation": 7_000_000, "test": 8_000_000}


def layout_seed(stage, split, index):
    if stage not in STAGES or split not in SPLIT_OFFSETS or not 0 <= index < 100_000:
        raise ValueError("invalid stage, split, or layout index")
    return SPLIT_OFFSETS[split] + stage * 100_000 + index


def generate_level(stage, seed):
    if stage not in STAGES:
        raise ValueError("stage must be 1, 2, or 3")
    rng = random.Random(seed)
    count_range, width_range, gap_range, rise_range = {
        1: ((6, 8), (90, 130), (65, 100), (40, 80)),
        2: ((8, 10), (65, 95), (80, 115), (65, 105)),
        3: ((10, 12), (50, 80), (90, 130), (80, 125)),
    }[stage]
    count = rng.randint(*count_range)
    widths = [rng.randint(160, 190)] + [rng.randint(*width_range) for _ in range(count - 2)] + [220]
    gaps = [rng.randint(*gap_range) for _ in range(count - 1)]
    rises = [rng.randint(*rise_range) for _ in range(count - 1)]
    level = ascent_level(f"Climb {stage} - {('Foothills', 'High ridge', 'Summit')[stage - 1]}",
                         widths, gaps, rises, ("Climber", "Mountaineer", "Summit")[stage - 1],
                         f"{sum(rises)}px ascent | seed {seed} | {GENERATOR_VERSION}")
    validate_layout(level)
    return level


def layout_record(stage, seed):
    level = generate_level(stage, seed)
    geometry = {"platforms": [asdict(p) for p in level.platforms],
                "start_x": level.start_x, "goal_x": level.goal_x}
    return {"stage": stage, "layout_seed": seed, "generator": GENERATOR_VERSION,
            "layout_hash": hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest(),
            **geometry, "world_width": level.world_width, "world_height": level.world_height,
            "max_steps": level.max_steps}


def evaluation_cases(stage, split, count):
    return [{**layout_record(stage, seed := layout_seed(stage, split, i)),
             "start_x": random.Random(seed + 9_000_000).uniform(*START_RANGE)} for i in range(count)]
