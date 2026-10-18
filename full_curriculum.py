"""Full_version curriculum: four terrain stages, then untouched-layout testing.

Stages 1-3 retain the easy.pdf generator. Stage 4 adds long, sequential routes.
The document's Stage 5 is the held-out evaluation, never a training stage.
"""

from dataclasses import asdict
import hashlib
import json
import random

from platformer import Level, Platform, PlatformerEnv
from ppo import np
from procedural import generate_level as easy_level, validate_layout
from train_ppo import RolloutCollector, START_RANGE

GENERATOR_VERSION = "full-v1"
STAGES = (1, 2, 3, 4)
TRAIN_LAYOUTS = 1000
SPLIT_OFFSETS = {"train": 3_000_000, "validation": 4_000_000, "test": 5_000_000}


def layout_seed(stage, split, index):
    if stage not in STAGES or split not in SPLIT_OFFSETS or not 0 <= index < 100_000:
        raise ValueError("invalid stage, split, or layout index")
    return SPLIT_OFFSETS[split] + stage * 100_000 + index


def generate_level(stage, seed):
    if stage not in STAGES:
        raise ValueError("stage must be 1, 2, 3, or 4")
    if stage < 4:
        return easy_level(stage, seed)
    rng = random.Random(seed)
    count = rng.randint(8, 12)
    platforms = [Platform(0, 450, rng.randint(150, 180))]
    for index in range(1, count):
        previous = platforms[-1]
        y = max(270, min(450, previous.y + rng.randint(-100, 90)))
        x = previous.x + previous.width + rng.randint(100, 170)
        width = rng.randint(40, 80) if index < count - 1 else 200
        platforms.append(Platform(x, y, width))
    world_width = platforms[-1].x + platforms[-1].width
    level = Level("Stage 4 - Sequential precision", tuple(platforms), 35, world_width - 50,
                  "Extreme", f"8-12 platforms | seed {seed} | {GENERATOR_VERSION}",
                  world_width, 200 + count * 80)
    validate_layout(level)
    return level


def layout_record(stage, seed):
    level = generate_level(stage, seed)
    # Keep the hash schema compatible with precision-v1 to detect reused geometry.
    geometry = {"platforms": [asdict(p) for p in level.platforms],
                "start_x": level.start_x, "goal_x": level.goal_x}
    return {"stage": stage, "layout_seed": seed, "generator": GENERATOR_VERSION,
            "layout_hash": hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest(),
            **geometry, "world_width": level.world_width, "max_steps": level.max_steps}


def evaluation_cases(stage, split, count):
    return [{**layout_record(stage, seed := layout_seed(stage, split, index)),
             "start_x": random.Random(seed + 9_000_000).uniform(*START_RANGE)}
            for index in range(count)]


class FullCollector(RolloutCollector):
    def __init__(self, stage, count, seed, rehearsal=0.25, platform_start_prob=0.25):
        self.stage = stage
        self.rehearsal = rehearsal
        self.platform_start_prob = platform_start_prob
        self.rng = random.Random(seed)
        self.seen_layouts = set()
        self.envs = [PlatformerEnv(1) for _ in range(count)]
        self.observations = np.asarray([self.reset(env) for env in self.envs])

    def reset(self, env):
        stage = self.stage
        if stage > 1 and self.rng.random() < self.rehearsal:
            stage = self.rng.randrange(1, stage)
        seed = layout_seed(stage, "train", self.rng.randrange(TRAIN_LAYOUTS))
        self.seen_layouts.add((stage, seed))
        env.level = generate_level(stage, seed)
        env.level_number = stage
        return super().reset(env)
