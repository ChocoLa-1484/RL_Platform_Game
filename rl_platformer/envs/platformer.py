"""Shared, deterministic platformer physics for the game and the RL agent."""

from dataclasses import dataclass
import math

WIDTH, HEIGHT = 900, 540
PLAYER_W, PLAYER_H = 24, 32
GROUND_Y = 450
MAX_STEPS = 420
ACTIONS = ("None", "Left", "Right", "Jump", "Left+Jump", "Right+Jump")
OBSERVATION_SIZE = 18
FPS = 60


@dataclass(frozen=True)
class Platform:
    x: float
    y: float
    width: float


@dataclass(frozen=True)
class Level:
    name: str
    platforms: tuple[Platform, ...]
    start_x: float
    goal_x: float
    difficulty: str
    challenge: str
    world_width: float = WIDTH
    max_steps: int = MAX_STEPS
    world_height: float = HEIGHT


def ascent_level(name, widths, gaps, rises, difficulty, challenge):
    """Build an ascending route with room above the summit and below the start."""
    if len(gaps) != len(widths) - 1 or len(rises) != len(gaps):
        raise ValueError("one gap and rise are required per crossing")
    world_height = max(HEIGHT, sum(rises) + 310)
    x, y = 0, world_height - 90
    platforms = [Platform(x, y, widths[0])]
    for width, gap, rise in zip(widths[1:], gaps, rises):
        x += platforms[-1].width + gap
        y -= rise
        platforms.append(Platform(x, y, width))
    world_width = max(WIDTH, x + widths[-1])
    return Level(name, tuple(platforms), 35, x + widths[-1] - 50,
                 difficulty, challenge, world_width, 200 + len(widths) * 90, world_height)


LEVELS = (
    Level("1 - Wide platforms", (Platform(0, 450, 205), Platform(265, 450, 315), Platform(645, 450, 255)), 35, 850,
          "Beginner", "Learn to move and jump across small gaps."),
    Level("2 - Changing heights", (Platform(0, 450, 190), Platform(260, 405, 175), Platform(510, 440, 160), Platform(750, 400, 150)), 35, 850,
          "Easy", "Adjust jump timing for platforms at different heights."),
    Level("3 - Precision landings", (Platform(0, 450, 175), Platform(245, 415, 95), Platform(410, 385, 85), Platform(565, 425, 95), Platform(730, 395, 170)), 35, 850,
          "Medium", "Land on narrower platforms without overshooting."),
    Level("4 - High jumps", (Platform(0, 450, 150), Platform(235, 355, 85), Platform(425, 420, 75), Platform(620, 325, 80), Platform(805, 400, 95)), 35, 850,
          "Hard", "Combine longer gaps with steep uphill jumps."),
    Level("5 - Tiny footholds", (Platform(0, 450, 140), Platform(240, 340, 50), Platform(415, 395, 46), Platform(585, 290, 44), Platform(780, 375, 120)), 35, 850,
          "Expert", "Time takeoffs and control speed on tiny footholds."),
    Level("6 - Long crossings", (Platform(0, 450, 170), Platform(310, 415, 90), Platform(540, 450, 75), Platform(760, 350, 80), Platform(1000, 420, 200)), 35, 1150,
          "Expert+", "Build speed for long gaps, then brake before the next jump.", 1200, 560),
    Level("7 - Narrow staircase", (Platform(0, 450, 150), Platform(270, 370, 60), Platform(440, 310, 50), Platform(620, 380, 48), Platform(810, 290, 50), Platform(1000, 370, 52), Platform(1200, 330, 250)), 35, 1400,
          "Master", "Chain precise landings across alternating heights.", 1450, 700),
    Level("8 - Rhythm changes", (Platform(0, 450, 160), Platform(300, 420, 55), Platform(450, 340, 44), Platform(650, 430, 48), Platform(860, 340, 44), Platform(1050, 390, 48), Platform(1250, 300, 48), Platform(1450, 390, 250)), 35, 1650,
          "Master+", "Adapt takeoff timing to uneven spacing and tiny footholds.", 1700, 800),
    Level("9 - Endurance route", (Platform(0, 450, 160), Platform(300, 390, 60), Platform(490, 430, 58), Platform(680, 340, 52), Platform(890, 410, 56), Platform(1110, 320, 50), Platform(1320, 400, 48), Platform(1530, 310, 46), Platform(1750, 390, 52), Platform(1970, 340, 230)), 35, 2150,
          "Extreme", "Maintain control through a long sequence of jumps.", 2200, 980),
    Level("10 - Precision marathon", (Platform(0, 450, 160), Platform(300, 350, 50), Platform(490, 430, 42), Platform(690, 335, 40), Platform(900, 420, 44), Platform(1110, 320, 40), Platform(1320, 400, 42), Platform(1530, 310, 40), Platform(1740, 410, 42), Platform(1960, 320, 40), Platform(2180, 400, 44), Platform(2400, 330, 200)), 35, 2550,
          "Extreme+", "Combine long gaps, 40-pixel footholds and twelve platforms.", 2600, 1200),
    ascent_level("11 - Mountain steps", (180, 100, 95, 90, 85, 90, 220),
                 (85, 90, 95, 95, 100, 100), (75, 85, 90, 80, 90, 100),
                 "Climber", "Climb 520 pixels through seven elevated platforms."),
    ascent_level("12 - High ridge", (170, 80, 75, 70, 70, 65, 70, 65, 220),
                 (95, 100, 105, 110, 105, 115, 110, 115), (90, 100, 105, 95, 110, 100, 110, 100),
                 "Mountaineer", "Climb 810 pixels with precise uphill takeoffs."),
    ascent_level("13 - Summit marathon", (170, 75, 65, 60, 60, 55, 60, 55, 60, 55, 60, 220),
                 (100, 105, 110, 115, 110, 115, 105, 115, 110, 115, 110),
                 (100, 110, 115, 105, 120, 110, 115, 105, 120, 115, 125),
                 "Summit", "Climb 1240 pixels across twelve platforms; brake before landing."),
)


class PlatformerEnv:
    def __init__(self, level: int | Level = 1, reward_mode="shaped", auto_jump=False):
        if not isinstance(level, Level) and not 1 <= level <= len(LEVELS):
            raise ValueError(f"level must be between 1 and {len(LEVELS)}")
        if reward_mode not in ("shaped", "sparse"):
            raise ValueError("reward_mode must be shaped or sparse")
        self.level_number = 0 if isinstance(level, Level) else level
        self.level = level if isinstance(level, Level) else LEVELS[level - 1]
        self.reward_mode = reward_mode
        self.auto_jump = auto_jump
        self.reset()

    def reset(self, start_x: float | None = None, platform_index: int = 0):
        if not 0 <= platform_index < len(self.level.platforms):
            raise ValueError("invalid starting platform")
        platform = self.level.platforms[platform_index]
        default_x = self.level.start_x if platform_index == 0 else platform.x + (platform.width - PLAYER_W) / 2
        x = float(default_x if start_x is None else start_x)
        if not math.isfinite(x) or not platform.x <= x <= platform.x + platform.width - PLAYER_W:
            raise ValueError("starting position must fit on the starting platform")
        self.x = x
        self.y = platform.y - PLAYER_H
        self.vx = 0.0
        self.vy = 0.0
        self.grounded = True
        self.steps = 0
        self.visited = set(range(platform_index + 1))
        self.current_platform = platform_index
        self.done = False
        self.success = False
        self.deaths = 0
        self.best_x = self.x
        self.jump_count = 0
        self.landing_speeds = []
        self.highest_landing_y = platform.y
        return self.observe()

    def observe(self):
        center = self.x + PLAYER_W / 2
        supporting = next((p for p in self.level.platforms if p.x <= center <= p.x + p.width and abs(self.y + PLAYER_H - p.y) < 7), None)
        if supporting is None:
            upcoming = next((p for p in self.level.platforms if p.x > center), self.level.platforms[-1])
            edge_distance = (upcoming.x - center) / 180
        else:
            edge_distance = (supporting.x + supporting.width - center) / 180
        next_platform = next((p for p in self.level.platforms if p.x > center), self.level.platforms[-1])
        gap = (next_platform.x - (supporting.x + supporting.width)) / 150 if supporting else (next_platform.x - center) / 150
        target_index = min(self.current_platform + 1, len(self.level.platforms) - 1)
        target = self.level.platforms[target_index]
        after_target = self.level.platforms[min(target_index + 1, len(self.level.platforms) - 1)]
        launch = self.level.platforms[self.current_platform]
        # The first nine fields remain compatible with the original linear policy.
        return (
            center / self.level.world_width,
            (self.y + PLAYER_H) / self.level.world_height,
            self.vx / 6,
            self.vy / 13,
            (self.level.goal_x - center) / self.level.world_width,
            1.0 if self.grounded else 0.0,
            max(-1.5, min(1.5, edge_distance)),
            max(-1.5, min(1.5, gap)),
            (next_platform.y - (self.y + PLAYER_H)) / 120,
            (target.x - center) / 300,
            (target.x + target.width - center) / 300,
            (target.y - (self.y + PLAYER_H)) / 150,
            target.width / 200,
            (launch.x + launch.width - center) / 300,
            (after_target.x - center) / 500,
            (after_target.y - (self.y + PLAYER_H)) / 150,
            after_target.width / 200,
            (self.level.platforms[-1].y - (self.y + PLAYER_H)) / self.level.world_height,
        )

    def step(self, action: int):
        if self.done:
            raise RuntimeError("reset the environment after an episode ends")
        if not 0 <= action < (3 if self.auto_jump else len(ACTIONS)):
            raise ValueError("invalid action")
        if self.auto_jump:
            action += 3
        old_x, old_bottom = self.x, self.y + PLAYER_H
        was_grounded = self.grounded
        direction = -1 if action in (1, 4) else 1 if action in (2, 5) else 0
        if direction:
            self.vx = max(-5.6, min(5.6, self.vx + direction * (0.72 if self.grounded else 0.48)))
        else:
            self.vx *= 0.76 if self.grounded else 0.94
        if action in (3, 4, 5) and self.grounded:
            self.vy = -11.8
            self.grounded = False
            self.jump_count += 1
        self.x = max(0.0, min(self.level.world_width - PLAYER_W, self.x + self.vx))
        self.vy = min(13.0, self.vy + 0.48)
        self.y += self.vy
        self.grounded = False
        landed = None
        if self.vy >= 0:
            for index, p in enumerate(self.level.platforms):
                if old_bottom <= p.y + 0.001 and self.y + PLAYER_H >= p.y and self.x + PLAYER_W > p.x and self.x < p.x + p.width:
                    self.y = p.y - PLAYER_H
                    self.vy = 0.0
                    self.grounded = True
                    landed = index
                    self.current_platform = index
                    if not was_grounded:
                        self.landing_speeds.append(abs(self.vx))
                    break
        self.steps += 1
        self.best_x = max(self.best_x, self.x)
        components = {"progress": 0.0, "step": 0.0, "stall": 0.0,
                      "platform": 0.0, "checkpoint": 0.0, "goal": 0.0, "death": 0.0,
                      "height": 0.0}
        if self.reward_mode == "shaped":
            components["progress"] = 0.035 * (self.x - old_x)
            components["step"] = -0.01
            if abs(self.x - old_x) < 0.2:
                components["stall"] = -0.1
        if landed is not None and landed not in self.visited:
            self.visited.add(landed)
            if self.reward_mode == "shaped":
                components["platform"] = 1.0
                if landed == len(self.level.platforms) // 2:
                    components["checkpoint"] = 5.0
        if landed is not None:
            if self.reward_mode == "shaped" and self.level.world_height > HEIGHT:
                components["height"] = 0.02 * max(0, self.highest_landing_y - self.level.platforms[landed].y)
            self.highest_landing_y = min(self.highest_landing_y, self.level.platforms[landed].y)
        if self.x + PLAYER_W >= self.level.goal_x and self.grounded and self.current_platform == len(self.level.platforms) - 1:
            self.done = self.success = True
            components["goal"] = 20.0
        elif self.y > self.level.world_height + 40:
            self.done = True
            self.deaths = 1
            components["death"] = -5.0
        elif self.steps >= self.level.max_steps:
            self.done = True
        return self.observe(), sum(components.values()), self.done, {
            "success": self.success, "death": bool(self.deaths),
            "progress": min(1.0, (self.best_x + PLAYER_W) / self.level.goal_x),
            "terminated": self.success or bool(self.deaths),
            "truncated": self.done and not self.success and not self.deaths,
            "reward_components": components,
            "landed_platform": landed if not was_grounded else None,
            "jumps": self.jump_count,
        }
