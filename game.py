"""Play any difficulty, or watch its trained PPO policy."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

try:
    import pygame
except ModuleNotFoundError:
    local_deps = Path(__file__).with_name(".deps")
    if not local_deps.is_dir():
        raise
    sys.path.insert(0, str(local_deps))
    import pygame

from platformer import HEIGHT, LEVELS, MAX_STEPS, PLAYER_H, PLAYER_W, PlatformerEnv, WIDTH
from ppo import load_policy
from procedural import generate_level, layout_seed

ROOT = Path(__file__).resolve().parent


def policy_path_for_level(level, model_dir, custom_path=None):
    if custom_path is not None:
        return custom_path
    path = model_dir / f"level{level}_policy.json"
    if level == 1 and not path.is_file() and model_dir == ROOT / "models":
        return ROOT / "level1_policy.json"
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=range(1, len(LEVELS) + 1), default=1)
    parser.add_argument("--agent", action="store_true", help="watch this level's saved PPO policy")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--policy", type=Path, help="use a specific policy, including the original linear model")
    parser.add_argument("--headless", action="store_true", help="render one agent episode offscreen and exit (0=success, 1=failure)")
    parser.add_argument("--screenshot", type=Path, help="save the final rendered frame")
    parser.add_argument("--procedural", action="store_true", help="play a generated layout from the three-stage curriculum")
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--layout-seed", type=int, help="replay an exact generated layout; defaults to the first held-out test layout")
    parser.add_argument("--auto-jump", action="store_true", help="automatically jump when grounded (three directional actions)")
    parser.add_argument("--trace", action="store_true", help="show the movement trajectory; T toggles it")
    parser.add_argument("--trajectory", type=Path, help="export the last episode's states and actions as JSON")
    args = parser.parse_args()
    if args.headless and not args.agent:
        parser.error("--headless requires --agent")
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
    model = None
    stage = args.stage
    current_seed = args.layout_seed if args.layout_seed is not None else layout_seed(stage, "test", 0)

    def selected_policy(level):
        if args.procedural:
            if args.policy:
                return args.policy
            directory = ROOT / "experiments" / "easy_pdf" / "ppo_shaped" if args.model_dir == ROOT / "models" else args.model_dir
            return directory / "policy.json"
        return policy_path_for_level(level, args.model_dir, args.policy)

    def create_env(level):
        auto_jump = getattr(model, "action_count", 6) == 3 if model is not None else args.auto_jump
        reward_mode = getattr(model, "metadata", {}).get("reward_mode", "shaped")
        layout = generate_level(stage, current_seed) if args.procedural else level
        return PlatformerEnv(layout, reward_mode=reward_mode, auto_jump=auto_jump)

    if args.agent:
        path = selected_policy(args.level)
        if not path.is_file():
            command = "python run_experiments.py" if args.procedural else f"python train_ppo.py --level {args.level}"
            parser.error(f"policy missing: {path}. Run: {command}")
        model = load_policy(path)
        if args.auto_jump and getattr(model, "action_count", 6) != 3:
            parser.error("--auto-jump requires a policy trained with three auto-jump actions")
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Precision Platformer - PPO Project")
    font = pygame.font.SysFont("arial", 20)
    large = pygame.font.SysFont("arial", 38, bold=True)
    clock = pygame.time.Clock()
    env = create_env(args.level)
    trace, trajectory = [], []
    notice = ""
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if pygame.K_1 <= event.key < pygame.K_1 + (3 if args.procedural else len(LEVELS)):
                    level = event.key - pygame.K_1 + 1
                    path = selected_policy(level)
                    if args.agent and not path.is_file():
                        notice = f"No policy for level {level}. Run: python train_ppo.py --level {level}"
                    else:
                        if args.agent:
                            model = load_policy(path)
                        if args.procedural:
                            stage = level
                            current_seed = args.layout_seed if args.layout_seed is not None else layout_seed(stage, "test", 0)
                        env = create_env(level)
                        trace, trajectory = [], []
                        notice = ""
                elif event.key == pygame.K_r:
                    env.reset()
                    trace, trajectory = [], []
                elif event.key == pygame.K_n and args.procedural:
                    current_seed += 1
                    env = create_env(args.level)
                    trace, trajectory = [], []
                elif event.key == pygame.K_t:
                    args.trace = not args.trace
                elif event.key == pygame.K_ESCAPE:
                    running = False
        keys = pygame.key.get_pressed()
        if not env.done:
            if model:
                action = model.action(env.observe(), greedy=True)[0]
            else:
                left = keys[pygame.K_LEFT] or keys[pygame.K_a]
                right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
                jump = keys[pygame.K_SPACE] or keys[pygame.K_UP] or keys[pygame.K_w]
                direction = -1 if left and not right else 1 if right and not left else 0
                action = (4 if direction < 0 else 5 if direction > 0 else 3) if jump else (1 if direction < 0 else 2 if direction > 0 else 0)
                if env.auto_jump:
                    action %= 3
            before = {"x": env.x, "y": env.y, "vx": env.vx, "vy": env.vy, "grounded": env.grounded}
            _, reward, _, info = env.step(action)
            trace.append((round(env.x + PLAYER_W / 2), round(env.y + PLAYER_H / 2)))
            trajectory.append({"step": env.steps, "before": before, "action": action, "reward": reward,
                               "x": env.x, "y": env.y, "vx": env.vx, "vy": env.vy,
                               "grounded": env.grounded, "success": env.success,
                               "landed_platform": info["landed_platform"]})
        screen.fill((20, 25, 42))
        for i, platform in enumerate(env.level.platforms):
            pygame.draw.rect(screen, (78, 181, 161) if i else (91, 136, 190), (platform.x, platform.y, platform.width, HEIGHT - platform.y))
            pygame.draw.rect(screen, (148, 231, 204), (platform.x, platform.y, platform.width, 6))
        pygame.draw.rect(screen, (247, 205, 99), (env.level.goal_x, env.level.platforms[-1].y - 68, 5, 68))
        pygame.draw.polygon(screen, (255, 238, 152), ((env.level.goal_x + 5, env.level.platforms[-1].y - 68), (env.level.goal_x + 45, env.level.platforms[-1].y - 54), (env.level.goal_x + 5, env.level.platforms[-1].y - 40)))
        if args.trace and len(trace) > 1:
            pygame.draw.lines(screen, (124, 152, 202), False, trace, 2)
        pygame.draw.rect(screen, (246, 111, 126), (round(env.x), round(env.y), PLAYER_W, PLAYER_H), border_radius=5)
        screen.blit(large.render(env.level.name, True, (244, 248, 255)), (25, 20))
        help_text = "Arrows/A,D + Space: move/jump  |  1-5: level  |  R: restart" if not args.agent else "PPO demo  |  1-5: level  |  R: restart  |  Esc: quit"
        if args.procedural:
            mode = "PPO auto jump" if model and env.auto_jump else "PPO demo" if model else "Arrows + Space"
            help_text = f"{mode} | 1-3: stage | N: new layout | R: retry | T: trace | Esc: quit"
        screen.blit(font.render(help_text, True, (188, 202, 220)), (26, 70))
        progress = min(1.0, (env.best_x + PLAYER_W) / env.level.goal_x)
        screen.blit(font.render(f"{env.level.difficulty}    Step {env.steps}/{MAX_STEPS}    Progress {progress:.0%}    Jumps {env.jump_count}", True, (188, 202, 220)), (26, 102))
        screen.blit(font.render(notice or env.level.challenge, True, (188, 202, 220)), (26, 132))
        if env.done:
            message = "LEVEL COMPLETE!" if env.success else "FAILED - PRESS R TO RETRY"
            screen.blit(large.render(message, True, (255, 234, 155)), (WIDTH // 2 - 225, 195))
        pygame.display.flip()
        if args.headless and env.done:
            running = False
        if not args.headless:
            clock.tick(60)
    if args.screenshot:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(screen, str(args.screenshot))
    if args.trajectory:
        args.trajectory.parent.mkdir(parents=True, exist_ok=True)
        args.trajectory.write_text(json.dumps({"level": asdict(env.level), "layout_seed": current_seed if args.procedural else None,
                                              "auto_jump": env.auto_jump, "success": env.success, "frames": trajectory}, indent=2), encoding="utf-8")
    pygame.quit()
    if args.headless:
        name = f"stage {stage}, layout {current_seed}" if args.procedural else f"level {env.level_number}"
        print(f"{name}: success={env.success}, steps={env.steps}")
        return 0 if env.success else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
