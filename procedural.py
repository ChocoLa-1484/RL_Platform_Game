"""Seeded three-stage layouts and disjoint experimental splits from easy.pdf."""

from dataclasses import asdict
import hashlib
import json
import random

from platformer import Level, Platform, PLAYER_W, WIDTH

GENERATOR_VERSION = "precision-v1"
SPLIT_OFFSETS = {"train": 0, "validation": 1_000_000, "test": 2_000_000}
TRAIN_LAYOUTS = 1000
STAGE_NAMES = {1: "Wide and flat", 2: "Changing heights", 3: "Precision landings"}


def layout_seed(stage, split, index):
    if stage not in STAGE_NAMES or split not in SPLIT_OFFSETS or not 0 <= index < 100_000:
        raise ValueError("invalid stage, split, or layout index")
    return SPLIT_OFFSETS[split] + stage * 100_000 + index


def descending_jump_distance(rise):
    """Discrete physics upper bound, not an agent controller or a path solution."""
    y, vy = 0.0, -11.8
    previous = y
    for step in range(1, 100):
        vy = min(13.0, vy + 0.48)
        y += vy
        if vy >= 0 and previous <= -rise <= y:
            return step * 5.6
        previous = y
    return 0.0


def validate_layout(level):
    if not level.platforms[0].x <= level.start_x <= level.platforms[0].x + level.platforms[0].width - PLAYER_W:
        raise ValueError("invalid start")
    if not level.platforms[-1].x <= level.goal_x <= level.platforms[-1].x + level.platforms[-1].width:
        raise ValueError("goal must be on the final platform")
    for platform in level.platforms:
        if platform.width < PLAYER_W or platform.x < 0 or platform.x + platform.width > WIDTH or not 250 <= platform.y <= 450:
            raise ValueError("invalid platform geometry")
    for left, right in zip(level.platforms, level.platforms[1:]):
        gap = right.x - left.x - left.width
        reach = descending_jump_distance(left.y - right.y)
        if gap <= 0 or reach <= 0 or gap + PLAYER_W > reach:
            raise ValueError("gap exceeds conservative single-jump distance")


def generate_level(stage, seed):
    if stage not in STAGE_NAMES:
        raise ValueError("stage must be 1, 2, or 3")
    rng = random.Random(seed)
    if stage == 1:
        widths = [rng.randint(180, 220), rng.randint(235, 310)]
        gap_range, height_range = (45, 80), (0, 0)
    elif stage == 2:
        widths = [rng.randint(160, 180), rng.randint(110, 150), rng.randint(110, 150)]
        gap_range, height_range = (60, 90), (-55, 55)
    else:
        widths = [rng.randint(140, 160)] + [rng.randint(45, 75) for _ in range(3)]
        gap_range, height_range = (75, 105), (-85, 70)
    platforms = []
    x, y = 0, 450
    for width in widths:
        platforms.append(Platform(x, y, width))
        x += width + rng.randint(*gap_range)
        y = max(270 if stage == 3 else 335, min(450, y + rng.randint(*height_range)))
    platforms.append(Platform(x, y, WIDTH - x))
    level = Level(f"Stage {stage} - {STAGE_NAMES[stage]}", tuple(platforms), 35, 850,
                  ("Beginner", "Intermediate", "Advanced")[stage - 1],
                  f"Procedural layout {seed} | {GENERATOR_VERSION}")
    validate_layout(level)
    return level


def layout_record(stage, seed):
    level = generate_level(stage, seed)
    geometry = {"platforms": [asdict(p) for p in level.platforms], "start_x": level.start_x, "goal_x": level.goal_x}
    return {"stage": stage, "layout_seed": seed, "generator": GENERATOR_VERSION,
            "layout_hash": hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest(), **geometry}
