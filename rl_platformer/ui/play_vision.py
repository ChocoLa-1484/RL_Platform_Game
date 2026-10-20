"""Watch an independently saved CNN play using only rendered screenshots."""

import argparse
import os
from pathlib import Path
import time

from rl_platformer.agents.vision import VisionAgent
from rl_platformer.envs.vision import PixelEnv, LivePreview, StopTraining
from rl_platformer.envs.platformer import LEVELS, FPS
from rl_platformer.ui.rendering import pygame
from rl_platformer.training.state import generated_level, layout_seed

from rl_platformer.paths import ROOT


def latest_policy():
    root = ROOT / "experiments" / "vision"
    paths = [p for p in root.glob("*/policy.pt") if (p.parent / "status.json").is_file()]
    if not paths:
        raise FileNotFoundError("No saved vision run. Run python -B train_vision.py, or pass --policy.")
    return max(paths, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--level", type=int, choices=range(1, len(LEVELS) + 1), default=1)
    parser.add_argument("--family", choices=("full", "height"))
    parser.add_argument("--stage", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--layout-seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--headless", action="store_true", help="one episode, exit 0 on success or 1 on failure")
    parser.add_argument("--screenshot", type=Path, help="save final agent-view pixels to a new file")
    args = parser.parse_args()
    if args.family == "height" and args.stage == 4:
        parser.error("height has only three stages")
    if args.screenshot and args.screenshot.exists():
        parser.error("screenshot already exists; choose a new filename")
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
    try:
        path = args.policy or latest_policy()
    except FileNotFoundError as exc:
        parser.error(str(exc))
    model = VisionAgent.load(path, args.device)
    level = generated_level(args.family, args.stage, args.layout_seed if args.layout_seed is not None else layout_seed(args.family, args.stage, "test", 0)) if args.family else args.level
    env = PixelEnv(level)
    observation = env.reset()
    preview = LivePreview(not args.headless, realtime=True)
    if not args.headless:
        pygame.display.set_caption("Screen-only CNN - PLAYBACK")
    print(f"Vision policy: {path}", flush=True)
    try:
        while True:
            observation, _, done, _ = env.step(model.action(observation))
            preview.show(env, f"CNN PLAYBACK | {path.parent.name} | screenshots only")
            if done:
                print(f"success={env.simulator.success}, seconds={env.simulator.steps / FPS:.3f}, dashes={env.simulator.dash_count}", flush=True)
                if args.headless:
                    break
                until = time.monotonic() + 1.5
                while time.monotonic() < until:
                    preview.show(env, "EPISODE FINISHED | restarting shortly")
                observation = env.reset()
    except StopTraining:
        pass
    finally:
        if args.screenshot:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(env.surface, str(args.screenshot))
        preview.close()
    return (0 if env.simulator.success else 1) if args.headless else 0


if __name__ == "__main__":
    raise SystemExit(main())
