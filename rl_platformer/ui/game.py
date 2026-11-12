"""Play any difficulty, or watch its trained PPO policy."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

from rl_platformer.ui.rendering import pygame, camera_offset, camera_vertical_offset, draw_world, draw_energy

from rl_platformer.envs.platformer import FPS, HEIGHT, LEVELS, PLAYER_H, PLAYER_W, WIDTH
from rl_platformer.envs.movement import MOVEMENT_VERSION, MOVEMENT_VERSIONS, MovementEnv, movement_action
from rl_platformer.agents.ppo import load_policy
from rl_platformer.levels.procedural import generate_level, layout_seed
from rl_platformer.levels import full_curriculum
from rl_platformer.levels import height_curriculum

from rl_platformer.paths import DEFAULT_STATE_POLICY as DEFAULT_MOVEMENT_POLICY


def level_from_key(key, count, current=1):
    if key == pygame.K_LEFTBRACKET:
        return (current - 2) % count + 1
    if key == pygame.K_RIGHTBRACKET:
        return current % count + 1
    if key == pygame.K_0:
        number = 10
    elif pygame.K_1 <= key <= pygame.K_9:
        number = key - pygame.K_1 + 1
    else:
        return None
    return number if 1 <= number <= min(count, 10) else None


ADVANCE_DELAY_FRAMES = FPS  # show "LEVEL COMPLETE!" for one second before moving on


def next_level_after_success(level_number, count):
    """The fixed level that follows a cleared one, or None after the last level."""
    return level_number + 1 if level_number < count else None


def record_clear(results, level_number, seconds, dashes):
    """Keep one clear per level; a retry after clearing replaces the earlier time."""
    kept = [row for row in results if row[0] != level_number]
    return sorted(kept + [(level_number, seconds, dashes)])


def run_summary(results, count):
    """Title and lines for the end-of-run screen."""
    total = sum(seconds for _, seconds, _ in results)
    dashes = sum(dash for _, _, dash in results)
    cleared = [level for level, _, _ in results]
    title = f"ALL {count} LEVELS CLEARED" if cleared == list(range(1, count + 1)) else f"{len(cleared)} OF {count} LEVELS CLEARED"
    lines = [f"Level {level:>2}   {seconds:5.2f}s   {dash:>2} dashes" for level, seconds, dash in results]
    return title, lines, f"Total {total:.2f}s ({round(total * FPS)} frames), {dashes} dashes"


def draw_summary(screen, font, large, results, count):
    title, lines, total = run_summary(results, count)
    panel = pygame.Surface((620, 400), pygame.SRCALPHA)
    panel.fill((16, 22, 34, 248))
    left, top = (WIDTH - 620) // 2, (HEIGHT - 400) // 2
    screen.blit(panel, (left, top))
    heading = large.render(title, True, (255, 234, 155))
    screen.blit(heading, (WIDTH // 2 - heading.get_width() // 2, top + 18))
    rows = (len(lines) + 1) // 2
    for i, line in enumerate(lines):
        column, row = divmod(i, rows)
        screen.blit(font.render(line, True, (220, 228, 240)), (left + 40 + column * 290, top + 80 + row * 30))
    screen.blit(font.render(total, True, (255, 234, 155)), (left + 40, top + 300))
    screen.blit(font.render("[ , ] or 1-9, 0: replay a level   Esc: quit", True, (188, 202, 220)), (left + 40, top + 345))


def policy_path_for_level(level, model_dir=None, custom_path=None):
    if custom_path is not None:
        return custom_path
    return Path(model_dir) / "policy.json" if model_dir else DEFAULT_MOVEMENT_POLICY


def keyboard_movement_action(keys):
    horizontal = int(bool(keys[pygame.K_RIGHT] or keys[pygame.K_d])) - int(bool(keys[pygame.K_LEFT] or keys[pygame.K_a]))
    vertical = int(bool(keys[pygame.K_DOWN] or keys[pygame.K_s])) - int(bool(keys[pygame.K_UP] or keys[pygame.K_w]))
    return movement_action(horizontal, vertical, bool(keys[pygame.K_SPACE]),
                           bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] or keys[pygame.K_x]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=range(1, len(LEVELS) + 1), default=1)
    parser.add_argument("--agent", action="store_true", help="watch this level's saved PPO policy")
    parser.add_argument("--model-dir", type=Path, help="folder containing a state agent policy.json")
    parser.add_argument("--policy", type=Path, help="use a specific jump/dash state policy")
    parser.add_argument("--headless", action="store_true", help="render one agent episode offscreen and exit (0=success, 1=failure)")
    parser.add_argument("--screenshot", type=Path, help="save the final rendered frame")
    parser.add_argument("--procedural", action="store_true", help="play a generated layout from the three-stage curriculum")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--full-curriculum", action="store_true", help="play the full four-stage generator")
    modes.add_argument("--height-curriculum", action="store_true", help="play the three-stage climbing generator")
    parser.add_argument("--stage", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--layout-seed", type=int, help="replay an exact generated layout; defaults to the first held-out test layout")
    parser.add_argument("--trace", action="store_true", help="show the movement trajectory; T toggles it")
    parser.add_argument("--trajectory", type=Path, help="export the last episode's states and actions as JSON")
    parser.add_argument("--no-advance", action="store_true", help="stay on a fixed level after clearing it instead of moving to the next")
    args = parser.parse_args()
    if args.full_curriculum or args.height_curriculum:
        args.procedural = True
    if args.stage == 4 and not args.full_curriculum:
        parser.error("--stage 4 requires --full-curriculum")
    generate_layout = height_curriculum.generate_level if args.height_curriculum else full_curriculum.generate_level if args.full_curriculum else generate_level
    make_seed = height_curriculum.layout_seed if args.height_curriculum else full_curriculum.layout_seed if args.full_curriculum else layout_seed
    stage_count = 4 if args.full_curriculum else 3
    if args.headless and not args.agent:
        parser.error("--headless requires --agent")
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
    model = None
    stage = args.stage
    current_seed = args.layout_seed if args.layout_seed is not None else make_seed(stage, "test", 0)

    def selected_policy(level):
        return policy_path_for_level(level, args.model_dir, args.policy)

    def create_env(level):
        layout = generate_layout(stage, current_seed) if args.procedural else level
        metadata = getattr(model, "metadata", {})
        version = MOVEMENT_VERSION if model is None else metadata.get("movement_version")
        if version not in MOVEMENT_VERSIONS:
            parser.error(f"unsupported movement physics version: {version}; use a jump/dash state policy")
        return MovementEnv(layout, time_penalty=metadata.get("time_penalty", 0.04), movement_version=version)

    if args.agent:
        path = selected_policy(args.level)
        if not path.is_file():
            parser.error(f"policy missing: {path}. Run: python -B train_movement.py --output-dir experiments/state/new_run")
        model = load_policy(path)
        if getattr(model, "action_count", 0) != 24:
            parser.error("this entry point requires a 24-action jump/dash state policy")
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Precision Platformer - PPO Project")
    font = pygame.font.SysFont("arial", 20)
    large = pygame.font.SysFont("arial", 38, bold=True)
    clock = pygame.time.Clock()
    env = create_env(args.level)
    trace, trajectory = [], []
    notice = ""
    advance = not (args.procedural or args.headless or args.no_advance)
    results, recorded, advance_timer, show_summary = [], False, None, False
    next_requested = False
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                level = level_from_key(event.key, stage_count if args.procedural else len(LEVELS), stage if args.procedural else env.level_number)
                if level is not None:
                    path = selected_policy(level)
                    if args.agent and not path.is_file():
                        notice = f"No policy at {path}"
                    else:
                        if args.agent:
                            model = load_policy(path)
                        if args.procedural:
                            stage = level
                            current_seed = args.layout_seed if args.layout_seed is not None else make_seed(stage, "test", 0)
                        env = create_env(level)
                        trace, trajectory = [], []
                        notice = ""
                        results, recorded, advance_timer, show_summary = [], False, None, False
                elif event.key == pygame.K_r:
                    env.reset()
                    trace, trajectory = [], []
                    recorded, advance_timer, show_summary = False, None, False
                elif event.key == pygame.K_n and advance:
                    next_requested = recorded and not show_summary
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
                action = keyboard_movement_action(keys)
            before = {"x": env.x, "y": env.y, "vx": env.vx, "vy": env.vy, "grounded": env.grounded}
            _, reward, _, info = env.step(action)
            trace.append((round(env.x + PLAYER_W / 2), round(env.y + PLAYER_H / 2)))
            trajectory.append({"step": env.steps, "before": before, "action": action, "reward": reward,
                               "x": env.x, "y": env.y, "vx": env.vx, "vy": env.vy,
                               "grounded": env.grounded, "success": env.success,
                               "landed_platform": info["landed_platform"],
                               "dash_energy": info.get("dash_energy"), "dash_started": info.get("dash_started", False)})
        if advance and env.done and env.success and not recorded:
            recorded = True
            results = record_clear(results, env.level_number, env.steps / FPS, getattr(env, "dash_count", 0))
            # The agent moves on by itself; a human player chooses N (next) or R (retry).
            advance_timer = ADVANCE_DELAY_FRAMES if model else None
        if advance_timer is not None:
            advance_timer -= 1
            if advance_timer <= 0:
                next_requested = True
        if next_requested:
            next_requested, advance_timer = False, None
            following = next_level_after_success(env.level_number, len(LEVELS))
            if following is None:
                show_summary = True
            else:
                env = create_env(following)
                trace, trajectory = [], []
                recorded = False
        draw_world(screen, env, trace if args.trace else ())
        agile = isinstance(env, MovementEnv)
        screen.blit(large.render(env.level.name, True, (244, 248, 255)), (25, 20))
        help_text = "Arrows + Space: move/jump | [,]: level | R: retry | T: trace" if not args.agent else "PPO demo | [,]: level | 1-9,0: shortcut | R: retry | T: trace"
        if args.procedural:
            mode = "PPO auto jump" if model and env.auto_jump else "PPO demo" if model else "Arrows + Space"
            help_text = f"{mode} | 1-{stage_count}: stage | N: new layout | R: retry | T: trace | Esc: quit"
        if agile and not model:
            help_text = "A/D: move | Space: hold jump | Shift/X + arrows/WASD: dash | R: retry"
        screen.blit(font.render(help_text, True, (188, 202, 220)), (26, 70))
        progress = min(1.0, (env.best_x + PLAYER_W) / env.level.goal_x)
        climbed = max(0, env.level.platforms[0].y - env.highest_landing_y)
        altitude = f"    Climb {climbed:.0f}px" if env.level.world_height > HEIGHT else ""
        timer = f"Time {env.steps / FPS:.2f}s" if agile else f"Step {env.steps}/{env.level.max_steps}"
        dash_count = f"    Dashes {env.dash_count}" if agile else ""
        screen.blit(font.render(f"{env.level.difficulty}    {timer}    Progress {progress:.0%}    Jumps {env.jump_count}{dash_count}{altitude}", True, (188, 202, 220)), (26, 102))
        if agile:
            energy_color = (94, 222, 183) if env.dash_energy else (98, 108, 128)
            screen.blit(font.render("Dash READY" if env.dash_energy else "Dash EMPTY", True, energy_color), (710, 18))
            draw_energy(screen, env)
        screen.blit(font.render(notice or env.level.challenge, True, (188, 202, 220)), (26, 132))
        if env.level.world_width > WIDTH or env.level.world_height > HEIGHT:
            scale = (WIDTH - 52) / env.level.world_width
            for platform in env.level.platforms:
                mini_y = 165 + platform.y / env.level.world_height * 35 if env.level.world_height > HEIGHT else 172
                pygame.draw.rect(screen, (78, 181, 161), (26 + platform.x * scale, mini_y, max(2, platform.width * scale), 4))
            mini_y = 165 + (env.y + PLAYER_H) / env.level.world_height * 35 if env.level.world_height > HEIGHT else 172
            pygame.draw.circle(screen, (246, 111, 126), (round(26 + (env.x + PLAYER_W / 2) * scale), round(mini_y)), 4)
        if advance:
            run_seconds = sum(seconds for _, seconds, _ in results) + (0 if recorded else env.steps / FPS)
            screen.blit(font.render(f"Level {env.level_number}/{len(LEVELS)}   Run {run_seconds:.2f}s", True, (188, 202, 220)), (710, 70))
        if show_summary:
            draw_summary(screen, font, large, results, len(LEVELS))
        elif env.done:
            message = "LEVEL COMPLETE!" if env.success else "FAILED - PRESS R TO RETRY"
            message_text = large.render(message, True, (255, 234, 155))
            screen.blit(message_text, (WIDTH // 2 - message_text.get_width() // 2, 195))
            if advance and env.success:
                hint = "N: next level   R: retry" if env.level_number < len(LEVELS) else "N: run summary   R: retry"
                hint_text = font.render(hint, True, (220, 228, 240))
                screen.blit(hint_text, (WIDTH // 2 - hint_text.get_width() // 2, 245))
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
                                              "auto_jump": env.auto_jump, "movement_version": getattr(env, "movement_version", "classic"),
                                              "success": env.success, "seconds": env.steps / FPS,
                                              "dashes": getattr(env, "dash_count", 0), "frames": trajectory}, indent=2), encoding="utf-8")
    pygame.quit()
    if args.headless:
        name = f"stage {stage}, layout {current_seed}" if args.procedural else f"level {env.level_number}"
        print(f"{name}: success={env.success}, steps={env.steps}")
        return 0 if env.success else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
