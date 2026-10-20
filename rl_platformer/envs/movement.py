"""Variable jump and eight-direction acceleration dash, shared by player and PPO."""

import math

from rl_platformer.envs.platformer import ACTIONS, FPS, PLAYER_H, PLAYER_W, PlatformerEnv

MOVEMENT_VERSION = "agile-v2"
MOVEMENT_VERSIONS = ("agile-v1", MOVEMENT_VERSION)
DASH_FRAMES = 8
DASH_ACCELERATION = 2.4
DASH_MAX_SPEED = 14.0
JUMP_RELEASE_SPEED = 4.0
# Screen-space directions: negative Y is up. Index 0 uses the current facing.
DASH_DIRECTIONS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (1, -1), (-1, 1), (1, 1))
DASH_NAMES = ("Facing", "Left", "Right", "Up", "Down", "UpLeft", "UpRight", "DownLeft", "DownRight")
MOVEMENT_ACTIONS = ACTIONS + tuple(f"Dash{name}" for name in DASH_NAMES) + tuple(f"Jump+Dash{name}" for name in DASH_NAMES)
MOVEMENT_OBSERVATION_SIZE = 28


def movement_action(horizontal=0, vertical=0, jump=False, dash=False):
    """Encode held buttons. Opposing directions should be cancelled by the caller."""
    if horizontal not in (-1, 0, 1) or vertical not in (-1, 0, 1):
        raise ValueError("directions must be -1, 0 or 1")
    if dash:
        return 6 + DASH_DIRECTIONS.index((horizontal, vertical)) + (9 if jump else 0)
    return (1 if horizontal < 0 else 2 if horizontal > 0 else 0) + (3 if jump else 0)


def decode_action(action, facing):
    if not isinstance(action, int) or not 0 <= action < len(MOVEMENT_ACTIONS):
        raise ValueError("invalid movement action")
    if action < 6:
        return (-1 if action in (1, 4) else 1 if action in (2, 5) else 0), action >= 3, False, (0., 0.)
    dx, dy = DASH_DIRECTIONS[(action - 6) % 9]
    horizontal = dx
    if dx == dy == 0:
        dx = facing
    length = math.hypot(dx, dy)
    return horizontal, action >= 15, True, (dx / length, dy / length)


class MovementEnv(PlatformerEnv):
    action_count = len(MOVEMENT_ACTIONS)
    movement_version = MOVEMENT_VERSION

    def __init__(self, level=1, time_penalty=0.04, movement_version=MOVEMENT_VERSION):
        if movement_version not in MOVEMENT_VERSIONS:
            raise ValueError("unsupported movement physics version")
        self.movement_version = movement_version
        if not math.isfinite(time_penalty) or time_penalty <= 0:
            raise ValueError("time_penalty must be positive and finite")
        self.time_penalty = time_penalty
        super().__init__(level)
        self.reward_mode = "speedrun"

    def reset(self, start_x=None, platform_index=0):
        self.facing = 1
        self.dash_energy = 1.0
        self.air_dash_used = False
        self.jump_was_held = False
        self.dash_was_held = False
        self.jump_active = False
        self.dash_remaining = 0
        self.dash_direction = (0., 0.)
        self.dash_count = 0
        self.jump_release_count = 0
        return super().reset(start_x, platform_index)

    def observe(self):
        return super().observe() + (
            float(self.facing), self.dash_energy, float(self.air_dash_used),
            float(self.jump_was_held), float(self.dash_was_held),
            self.dash_remaining / DASH_FRAMES, *self.dash_direction,
            float(self.jump_active), self.steps / self.level.max_steps,
        )

    def step(self, action):
        if self.done:
            raise RuntimeError("reset the environment after an episode ends")
        direction, jump, dash, dash_direction = decode_action(action, self.facing)
        old_x, old_bottom = self.x, self.y + PLAYER_H
        was_grounded = self.grounded
        if direction:
            self.facing = direction
        if jump and not self.jump_was_held and self.grounded:
            self.vy = -11.8
            self.grounded = False
            self.jump_active = True
            self.jump_count += 1
        dash_started = dash and not self.dash_was_held and self.dash_energy >= 1 and not self.air_dash_used and self.dash_remaining == 0
        if dash_started:
            # A new dash replaces both components of the incoming momentum.
            # Do this once, after any simultaneous jump impulse, then accelerate.
            if self.movement_version == MOVEMENT_VERSION:
                self.vx = self.vy = 0.0
            self.dash_energy = 0.0
            self.air_dash_used = True
            self.dash_remaining = DASH_FRAMES
            self.dash_direction = dash_direction
            self.dash_count += 1
            self.jump_active = False
        if self.jump_active and not jump and self.vy < -JUMP_RELEASE_SPEED:
            self.vy = -JUMP_RELEASE_SPEED
            self.jump_active = False
            self.jump_release_count += 1
        if self.dash_remaining:
            dx, dy = self.dash_direction
            self.vx += dx * DASH_ACCELERATION
            self.vy += dy * DASH_ACCELERATION
            speed = math.hypot(self.vx, self.vy)
            if speed > DASH_MAX_SPEED:
                self.vx *= DASH_MAX_SPEED / speed
                self.vy *= DASH_MAX_SPEED / speed
            self.dash_remaining -= 1
        elif direction:
            if abs(self.vx) > 5.6:
                self.vx *= 0.94
                if direction * self.vx < 0:
                    self.vx += direction * 0.48
            else:
                self.vx = max(-5.6, min(5.6, self.vx + direction * (0.72 if self.grounded else 0.48)))
        else:
            self.vx *= 0.76 if self.grounded else 0.94
        self.x = max(0.0, min(self.level.world_width - PLAYER_W, self.x + self.vx))
        self.vy = min(DASH_MAX_SPEED, self.vy + 0.48)
        self.y += self.vy
        if self.y < 0:
            self.y, self.vy = 0., max(0., self.vy)
        self.grounded = False
        landed = None
        if self.vy >= 0:
            # Intersect the falling foot segment with platform tops. A fast
            # diagonal dash must not tunnel through a narrow landing surface.
            new_bottom = self.y + PLAYER_H
            hits = []
            for index, platform in enumerate(self.level.platforms):
                if old_bottom <= platform.y + 0.001 and new_bottom >= platform.y:
                    fraction = (platform.y - old_bottom) / max(new_bottom - old_bottom, 1e-12)
                    crossing_x = old_x + fraction * (self.x - old_x)
                    if crossing_x + PLAYER_W > platform.x and crossing_x < platform.x + platform.width:
                        hits.append((fraction, index, crossing_x))
            if hits:
                _, landed, crossing_x = min(hits)
                platform = self.level.platforms[landed]
                # Grounded movement keeps its full horizontal displacement;
                # airborne collisions stop at the actual contact point.
                if not was_grounded:
                    self.x = crossing_x
                if self.x + PLAYER_W > platform.x and self.x < platform.x + platform.width:
                    self.y, self.vy = platform.y - PLAYER_H, 0.
                    self.grounded = True
                    self.current_platform = landed
                    if not was_grounded:
                        self.landing_speeds.append(abs(self.vx))
                        self.dash_remaining = 0
                else:
                    landed = None
        if self.vy >= 0 or self.grounded:
            self.jump_active = False
        if self.grounded and self.dash_remaining == 0:
            self.dash_energy = 1.0
            self.air_dash_used = False
        if self.dash_remaining == 0:
            self.dash_direction = (0., 0.)
        self.jump_was_held, self.dash_was_held = jump, dash
        self.steps += 1
        self.best_x = max(self.best_x, self.x)
        # Reward speed only alongside actual progress/success. A failed run
        # never receives the finish-time bonus, even if it ends immediately.
        components = {"progress": 0.015 * (self.x - old_x), "time": -self.time_penalty,
                      "height": 0., "goal": 0., "finish_time": 0., "death": 0.}
        if landed is not None:
            self.visited.add(landed)
            surface = self.level.platforms[landed].y
            components["height"] = 0.01 * max(0., self.highest_landing_y - surface)
            self.highest_landing_y = min(self.highest_landing_y, surface)
        if self.x + PLAYER_W >= self.level.goal_x and self.grounded and self.current_platform == len(self.level.platforms) - 1:
            self.done = self.success = True
            components["goal"] = 50.
            components["finish_time"] = 20. * max(0., 1. - self.steps / self.level.max_steps)
        elif self.y > self.level.world_height + 40:
            self.done = True
            self.deaths = 1
            components["death"] = -30.
        elif self.steps >= self.level.max_steps:
            self.done = True
        return self.observe(), sum(components.values()), self.done, {
            "success": self.success, "death": bool(self.deaths),
            "terminated": self.success or bool(self.deaths),
            "truncated": self.done and not self.success and not self.deaths,
            "progress": min(1., (self.best_x + PLAYER_W) / self.level.goal_x),
            "reward_components": components, "landed_platform": landed if not was_grounded else None,
            "jumps": self.jump_count, "dashes": self.dash_count, "dash_started": dash_started,
            "dash_energy": self.dash_energy, "seconds": self.steps / FPS,
        }
