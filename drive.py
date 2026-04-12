"""
Model-driven car: imitation learning policy navigates through obstacles.

The policy network directly predicts which action the A* expert would take,
given the 6x6 FOV and goal direction. FOV obstacle check as safety net.

Usage:
    python drive.py --checkpoint model/checkpoints/policy_best.pt
    python drive.py --checkpoint model/checkpoints/policy_best.pt --expert-compare
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import pygame

sys.path.insert(0, str(Path(__file__).parent / "game"))
sys.path.insert(0, str(Path(__file__).parent / "model"))

from car_game import (
    CarGame,
    DFSNavigator,
    LOCAL_FOV,
    VISIBILITY,
    CAR_SIZE,
    DISPLAY_W,
    DISPLAY_H,
)
from renderer import GameRenderer
from policy import DrivingPolicy, ACTION_TABLE

GOAL_DIST_MIN = 10
GOAL_DIST_MAX = 25
GOAL_REACH_DIST = 3


def _random_goal(game, rng):
    angle = rng.uniform(0, 2 * math.pi)
    dist = rng.integers(GOAL_DIST_MIN, GOAL_DIST_MAX + 1)
    gy = game.car_y + int(round(dist * math.sin(angle)))
    gx = game.car_x + int(round(dist * math.cos(angle)))
    return gy, gx


def _goal_dir(game, gy, gx):
    dy = gy - game.car_y
    dx = gx - game.car_x
    d = math.sqrt(dy * dy + dx * dx)
    if d > 0:
        return [dy / d, dx / d]
    return [0.0, 0.0]


def _fov_danger(fov, ax, ay):
    """Check if moving (ax, ay) places the car on a visible obstacle."""
    new_r = VISIBILITY + int(ay)
    new_c = VISIBILITY + int(ax)
    danger = 0
    for dr in range(CAR_SIZE):
        for dc in range(CAR_SIZE):
            r, c = new_r + dr, new_c + dc
            if 0 <= r < LOCAL_FOV and 0 <= c < LOCAL_FOV:
                if fov[r, c] < -0.5:
                    danger += 1
    return danger


def load_policy(ckpt_path, device):
    model = DrivingPolicy()
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def policy_action(model, fov_np, goal_dir, device, visit_counts, car_y, car_x):
    """Get action from policy, with FOV safety check and anti-loop."""
    with torch.no_grad():
        fov_t = torch.from_numpy(fov_np).float().unsqueeze(0).to(device)
        gd_t = torch.tensor([goal_dir], dtype=torch.float32).to(device)
        logits = model(fov_t, gd_t)[0].cpu().numpy()  # (8,)

    # score each action: policy logit + safety + anti-loop
    gd = goal_dir
    best_score = -1e9
    best_action = ACTION_TABLE[0]

    for i, (ax, ay) in enumerate(ACTION_TABLE):
        obs = _fov_danger(fov_np, ax, ay)
        visits = visit_counts.get((car_y + int(ay), car_x + int(ax)), 0)

        # policy confidence as base score, penalise obstacles and revisits
        score = logits[i] - 20.0 * obs - 1.5 * visits

        if score > best_score:
            best_score = score
            best_action = (ax, ay)

    return best_action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--expert-compare", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    print(f"Loading: {args.checkpoint}")
    model = load_policy(args.checkpoint, device)

    rng = np.random.default_rng(args.seed)

    pygame.init()
    W = DISPLAY_W * 2 + 20 if args.expert_compare else DISPLAY_W
    screen = pygame.display.set_mode((W, DISPLAY_H))
    title = "Self-Driving — POLICY vs DFS" if args.expert_compare else "Self-Driving — POLICY"
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 18, bold=True)
    font_big = pygame.font.SysFont("monospace", 32, bold=True)

    renderer = GameRenderer()

    # --- policy game ---
    game_m = CarGame(seed=args.seed)
    game_m.reset()
    gy, gx = _random_goal(game_m, rng)
    model_trail = []
    visit_counts = {}
    goals_reached_m = 0

    # --- expert game ---
    game_e = None
    nav = None
    goals_reached_e = 0
    if args.expert_compare:
        game_e = CarGame(seed=args.seed)
        game_e.reset()
        nav = DFSNavigator(goal_y=gy, goal_x=gx)
        nav.reset(start_y=game_e.car_y, start_x=game_e.car_x)

    running = True
    game_num = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                if event.key == pygame.K_r:
                    game_m.reset()
                    gy, gx = _random_goal(game_m, rng)
                    model_trail.clear()
                    visit_counts.clear()
                    goals_reached_m = 0
                    if game_e:
                        game_e.reset()
                        nav = DFSNavigator(goal_y=gy, goal_x=gx)
                        nav.reset(start_y=game_e.car_y, start_x=game_e.car_x)
                        goals_reached_e = 0

        # --- policy drives ---
        fov_m = game_m.get_local_fov()
        gdir = _goal_dir(game_m, gy, gx)
        ax_m, ay_m = policy_action(
            model, fov_m, gdir, device,
            visit_counts, game_m.car_y, game_m.car_x,
        )
        _, done_m, info_m = game_m.step(ax_m, ay_m)
        pos = (game_m.car_y, game_m.car_x)
        model_trail.append(pos)
        visit_counts[pos] = visit_counts.get(pos, 0) + 1

        # check if reached goal
        dist_m = abs(game_m.car_y - gy) + abs(game_m.car_x - gx)
        if dist_m <= GOAL_REACH_DIST:
            goals_reached_m += 1
            gy, gx = _random_goal(game_m, rng)
            if nav:
                nav.reset(
                    start_y=game_e.car_y, start_x=game_e.car_x,
                    goal_y=gy, goal_x=gx,
                )

        # --- expert drives ---
        done_e = False
        info_e = {}
        if game_e and nav:
            if nav.reached:
                goals_reached_e += 1
                nav.reset(
                    start_y=game_e.car_y, start_x=game_e.car_x,
                    goal_y=gy, goal_x=gx,
                )
            ax_e, ay_e = nav.next_action(game_e)
            _, done_e, info_e = game_e.step(ax_e, ay_e)

        # ---- draw policy view ----
        renderer.draw(screen, game_m)
        renderer.draw_goal(screen, game_m, gy, gx, reached=(dist_m <= GOAL_REACH_DIST))

        if len(model_trail) > 1:
            renderer.draw_path(screen, game_m, model_trail[-80:], backtracking=False)

        # HUD
        hud_str = (
            f" POLICY  Score:{info_m['score']}  Steps:{game_m.steps}"
            f"  Goals:{goals_reached_m}  Dist:{dist_m}"
        )
        bar = pygame.Surface((DISPLAY_W, 28), pygame.SRCALPHA)
        bar.fill((0, 0, 0, 120))
        screen.blit(bar, (0, 0))
        hud = font.render(hud_str, True, (255, 255, 100))
        screen.blit(hud, (6, 5))

        # FOV preview
        renderer.draw_grid_preview(
            screen, fov_m,
            x=DISPLAY_W - LOCAL_FOV * 6 - 10, y=32, scale=6,
        )

        if done_m:
            game_num += 1
            overlay = pygame.Surface((DISPLAY_W, DISPLAY_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 100))
            screen.blit(overlay, (0, 0))
            crash = font_big.render(
                f"CRASH  Score:{info_m['score']}  Goals:{goals_reached_m}",
                True, (255, 70, 70),
            )
            screen.blit(crash, crash.get_rect(center=(DISPLAY_W // 2, DISPLAY_H // 2 - 20)))
            sub = font.render("Press R to restart", True, (220, 220, 220))
            screen.blit(sub, sub.get_rect(center=(DISPLAY_W // 2, DISPLAY_H // 2 + 20)))

            print(f"Game {game_num} | POLICY score:{info_m['score']} goals:{goals_reached_m}", end="")
            if game_e:
                print(f"  | DFS score:{info_e.get('score', '?')} goals:{goals_reached_e}", end="")
            print()

        # ---- draw expert view ----
        if game_e:
            expert_surf = pygame.Surface((DISPLAY_W, DISPLAY_H))
            renderer.draw(expert_surf, game_e)
            renderer.draw_visited(expert_surf, game_e, nav.visited)
            renderer.draw_path(expert_surf, game_e, nav.path, backtracking=False)
            renderer.draw_goal(expert_surf, game_e, gy, gx, reached=nav.reached)
            screen.blit(expert_surf, (DISPLAY_W + 20, 0))

            bar_e = pygame.Surface((DISPLAY_W, 28), pygame.SRCALPHA)
            bar_e.fill((0, 0, 0, 120))
            screen.blit(bar_e, (DISPLAY_W + 20, 0))
            hud_e = font.render(
                f" DFS  Score:{info_e.get('score', 0)}  Goals:{goals_reached_e}",
                True, (100, 255, 100),
            )
            screen.blit(hud_e, (DISPLAY_W + 26, 5))

            if done_e and not done_m:
                game_e.reset(seed=args.seed)
                nav = DFSNavigator(goal_y=gy, goal_x=gx)
                nav.reset(start_y=game_e.car_y, start_x=game_e.car_x)

        pygame.display.flip()
        clock.tick(10)

        if done_m:
            _wait_for_restart(clock, game_m, game_e, nav, rng,
                              model_trail, visit_counts)

    pygame.quit()


def _wait_for_restart(clock, game_m, game_e, nav, rng, trail, visits):
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game_m.reset()
                    trail.clear()
                    visits.clear()
                    if game_e:
                        game_e.reset()
                    return
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit()
                    raise SystemExit
        clock.tick(15)


if __name__ == "__main__":
    main()
