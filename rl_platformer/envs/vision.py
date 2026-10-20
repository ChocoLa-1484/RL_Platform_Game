"""Pixel observation boundary and an optional live view of actual rollouts."""

import time
from rl_platformer.agents.vision import FrameHistory, np

from rl_platformer.envs.movement import MovementEnv
from rl_platformer.envs.platformer import WIDTH, HEIGHT, FPS
from rl_platformer.ui.rendering import pygame, draw_agent_view


class PixelEnv:
    def __init__(self, level=1, time_penalty=.10):
        self.simulator = MovementEnv(level, time_penalty=time_penalty)
        self.surface = pygame.Surface((WIDTH, HEIGHT))
        self.history = FrameHistory()
        self.reset()

    def reset(self, start_x=None, level=None):
        if level is not None:
            self.simulator.level = level
        self.simulator.reset(start_x)
        draw_agent_view(self.surface, self.simulator)
        return self.history.reset(self.surface)

    def step(self, action):
        # Numeric simulator observations are deliberately discarded here.
        _, reward, done, info = self.simulator.step(int(action))
        draw_agent_view(self.surface, self.simulator)
        return self.history.push(self.surface), reward, done, info


class StopTraining(Exception):
    pass


class LivePreview:
    """Display only: captions and training statistics never enter observations."""
    def __init__(self, enabled=True, realtime=False):
        self.enabled, self.realtime = enabled, realtime
        self.last_draw = 0.
        self.paused = False
        if enabled:
            pygame.init()
            self.screen = pygame.display.set_mode((1200, 640))
            pygame.display.set_caption("Screen-only CNN - LIVE TRAINING")
            self.font = pygame.font.SysFont("consolas", 18)
            self.clock = pygame.time.Clock()

    def show(self, env, label, force=False):
        if not self.enabled:
            return
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    raise StopTraining()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.paused = not self.paused
            now = time.monotonic()
            if force or self.paused or now - self.last_draw >= 1 / 30:
                self.last_draw = now
                self.screen.fill((16, 20, 34))
                self.screen.blit(env.surface, (0, 0))
                self.screen.blit(self.font.render("CNN input: oldest -> newest", True, (230, 235, 245)), (906, 8))
                for i, gray in enumerate(env.history.frames):
                    rgb = np.repeat(gray.T[:, :, None], 3, axis=2)
                    thumb = pygame.surfarray.make_surface(rgb)
                    self.screen.blit(pygame.transform.scale(thumb, (200, 120)), (944, 36 + i * 128))
                sim = env.simulator
                lines = [label, f"{sim.level.name} | time {sim.steps / FPS:.2f}s | dashes {sim.dash_count} | success {sim.success}",
                         ("PAUSED | " if self.paused else "") + "Space: pause/resume | Esc: save and stop | preview may skip frames at training speed"]
                for i, line in enumerate(lines):
                    self.screen.blit(self.font.render(line, True, (190, 214, 231)), (16, 550 + 28 * i))
                pygame.display.flip()
            if not self.paused:
                break
            self.clock.tick(30)
        if self.realtime:
            self.clock.tick(FPS)

    def snapshot(self, path):
        if self.enabled:
            pygame.image.save(self.screen, str(path))

    def close(self):
        if self.enabled:
            pygame.quit()
