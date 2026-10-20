"""Shared gameplay pixels for human play and the screen-only agent."""

from pathlib import Path
import sys
from rl_platformer.paths import ROOT

try:
    import pygame
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT / ".deps"))
    import pygame

from rl_platformer.envs.platformer import HEIGHT, WIDTH, PLAYER_H, PLAYER_W


def camera_offset(env):
    return max(0.0, min(env.x - WIDTH * 0.35, env.level.world_width - WIDTH))


def camera_vertical_offset(env):
    if env.level.world_height <= HEIGHT:
        return 0.0
    return max(-180.0, min(env.y - HEIGHT * 0.55, env.level.world_height - HEIGHT))


def draw_world(screen, env, trace=()):
    """Only the visible viewport; no minimap, coordinates or level metadata."""
    screen.fill((20, 25, 42))
    camera, camera_y = camera_offset(env), camera_vertical_offset(env)
    for i, platform in enumerate(env.level.platforms):
        top = platform.y - camera_y
        thickness = 22 if env.level.world_height > HEIGHT else HEIGHT - top
        pygame.draw.rect(screen, (78, 181, 161) if i else (91, 136, 190),
                         (platform.x - camera, top, platform.width, thickness))
        pygame.draw.rect(screen, (148, 231, 204), (platform.x - camera, top, platform.width, 6))
    gx, gy = env.level.goal_x - camera, env.level.platforms[-1].y - camera_y
    pygame.draw.rect(screen, (247, 205, 99), (gx, gy - 68, 5, 68))
    pygame.draw.polygon(screen, (255, 238, 152), ((gx + 5, gy - 68), (gx + 45, gy - 54), (gx + 5, gy - 40)))
    if len(trace) > 1:
        pygame.draw.lines(screen, (124, 152, 202), False, [(x - camera, y - camera_y) for x, y in trace], 2)
    agile = hasattr(env, "dash_remaining")
    color = (112, 224, 255) if agile and env.dash_remaining else (246, 111, 126)
    pygame.draw.rect(screen, color, (round(env.x - camera), round(env.y - camera_y), PLAYER_W, PLAYER_H), border_radius=5)
    if agile:
        pygame.draw.rect(screen, (255, 255, 255),
                         (env.x - camera + (17 if env.facing > 0 else 4), env.y - camera_y + 7, 4, 5))


def draw_energy(screen, env):
    color = (94, 222, 183) if env.dash_energy else (98, 108, 128)
    pygame.draw.rect(screen, (51, 62, 82), (710, 46, 160, 12), border_radius=4)
    pygame.draw.rect(screen, color, (710, 46, round(160 * env.dash_energy), 12), border_radius=4)


def draw_agent_view(screen, env):
    draw_world(screen, env)
    draw_energy(screen, env)
