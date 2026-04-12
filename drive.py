"""
Model-driven car: the trained JEPA world model navigates through obstacles.

The model plans by predicting future embeddings for candidate action sequences
(each conditioned on action + goal_dir) and picks the safest trajectory.

Usage:
    python drive.py --checkpoint model/checkpoints/driving_jepa_epoch_30.pt
    python drive.py --checkpoint model/checkpoints/driving_jepa_epoch_30.pt --expert-compare
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
    FOV_H,
    FOV_W,
)
from renderer import GameRenderer

GOAL_DIST_MIN = 10
GOAL_DIST_MAX = 25
GOAL_REACH_DIST = 3  # Manhattan distance to count as "reached"


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


def load_model(ckpt_path, device):
    model = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.eval()
    model.requires_grad_(False)
    return model


def _fov_danger(fov, ax, ay):
    """Check if moving (ax, ay) places the car on an obstacle in the FOV.

    Returns number of obstacle cells the car's 2x2 body would overlap.
    """
    # car's top-left in FOV is (VISIBILITY, VISIBILITY)
    new_r = VISIBILITY + int(ay)
    new_c = VISIBILITY + int(ax)
    danger = 0
    for dr in range(CAR_SIZE):
        for dc in range(CAR_SIZE):
            r, c = new_r + dr, new_c + dc
            if 0 <= r < LOCAL_FOV and 0 <= c < LOCAL_FOV:
                if fov[r, c] < -0.5:
                    danger += 1
            else:
                # outside FOV = unknown — small penalty
                danger += 0.25
    return danger


def plan_action(model, fov_np, goal_dir, device, car_y, car_x,
                visit_counts, horizon=3):
    """Pick best action: avoid obstacles, maximise goal alignment, don't loop.

    1. FOV obstacle check (hard safety)
    2. Goal alignment (direction)
    3. Full-history anti-loop: penalise cells proportional to visit count
    4. Model cost as small tiebreaker
    """
    with torch.no_grad():
        fov = torch.from_numpy(fov_np).float().unsqueeze(0).unsqueeze(0).to(device)
        info = model.encode({"grid": fov})
        init_emb = info["emb"]

        # 8 candidate actions (no STAY)
        candidates = []
        for dx in [-1.0, 0.0, 1.0]:
            for dy in [-1.0, 0.0, 1.0]:
                if dx == 0.0 and dy == 0.0:
                    continue
                candidates.append([dx, dy])

        S = len(candidates)
        gd = goal_dir

        # model cost
        cond = torch.tensor(
            [[ax, ay, gd[0], gd[1]] for ax, ay in candidates],
            dtype=torch.float32,
        )
        act_seq = cond.unsqueeze(0).unsqueeze(2).expand(1, S, horizon, 4).to(device)
        cost = model.plan_cost(init_emb, act_seq, history_size=1)
        costs_np = cost[0].cpu().numpy()
        c_min, c_max = costs_np.min(), costs_np.max()
        costs_norm = (costs_np - c_min) / (c_max - c_min + 1e-6)

        # score each candidate
        scores = []
        for i, (ax, ay) in enumerate(candidates):
            obs_penalty = _fov_danger(fov_np, ax, ay)

            mag = math.sqrt(ax * ax + ay * ay)
            alignment = (ay * gd[0] + ax * gd[1]) / mag if mag > 0 else 0.0

            # full-history visit penalty — grows with each revisit
            next_y = car_y + int(ay)
            next_x = car_x + int(ax)
            visits = visit_counts.get((next_y, next_x), 0)

            score = (alignment
                     - 10.0 * obs_penalty
                     - 2.0 * visits
                     - 0.1 * costs_norm[i])
            scores.append(score)

        best_idx = int(np.argmax(scores))
        return candidates[best_idx], costs_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=3)
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
    model = load_model(args.checkpoint, device)

    rng = np.random.default_rng(args.seed)

    pygame.init()
    W = DISPLAY_W * 2 + 20 if args.expert_compare else DISPLAY_W
    screen = pygame.display.set_mode((W, DISPLAY_H))
    title = "Self-Driving — MODEL vs DFS" if args.expert_compare else "Self-Driving — MODEL"
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 18, bold=True)
    font_big = pygame.font.SysFont("monospace", 32, bold=True)

    renderer = GameRenderer()

    # --- model game ---
    game_m = CarGame(seed=args.seed)
    game_m.reset()
    gy, gx = _random_goal(game_m, rng)
    model_trail = []
    visit_counts = {}  # (y, x) -> count — full history for anti-loop
    goals_reached_m = 0

    # --- expert game (same seed = same world) ---
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

        # --- model drives ---
        fov_m = game_m.get_local_fov()
        gdir = _goal_dir(game_m, gy, gx)
        (ax_m, ay_m), costs = plan_action(
            model, fov_m, gdir, device,
            car_y=game_m.car_y, car_x=game_m.car_x,
            visit_counts=visit_counts, horizon=args.horizon,
        )
        _, done_m, info_m = game_m.step(ax_m, ay_m)
        pos = (game_m.car_y, game_m.car_x)
        model_trail.append(pos)
        visit_counts[pos] = visit_counts.get(pos, 0) + 1

        # check if model reached goal
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

        # ---- draw model view ----
        renderer.draw(screen, game_m)
        renderer.draw_goal(screen, game_m, gy, gx, reached=(dist_m <= GOAL_REACH_DIST))

        # draw model trail (last 50 positions)
        if len(model_trail) > 1:
            renderer.draw_path(screen, game_m, model_trail[-50:], backtracking=False)

        # HUD
        hud_str = (
            f" MODEL  Score:{info_m['score']}  Steps:{game_m.steps}"
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
            x=DISPLAY_W - LOCAL_FOV * 6 - 10,
            y=32,
            scale=6,
        )

        # crash overlay
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

            print(f"Game {game_num} | MODEL score:{info_m['score']} goals:{goals_reached_m}", end="")
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
            expert_surf_hud = pygame.Surface((DISPLAY_W, 28), pygame.SRCALPHA)
            expert_surf_hud.fill((0, 0, 0, 120))
            screen.blit(expert_surf_hud, (DISPLAY_W + 20, 0))
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
        clock.tick(8)

        if done_m:
            _wait_for_restart(clock, game_m, game_e, nav, rng)
            if game_m:
                model_trail.clear()
                visit_counts.clear()
                goals_reached_m = 0
                gy, gx = _random_goal(game_m, rng)
                if game_e and nav:
                    goals_reached_e = 0
                    nav = DFSNavigator(goal_y=gy, goal_x=gx)
                    nav.reset(start_y=game_e.car_y, start_x=game_e.car_x)

    pygame.quit()


def _wait_for_restart(clock, game_m, game_e, nav, rng):
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game_m.reset()
                    if game_e:
                        game_e.reset()
                    return
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit()
                    raise SystemExit
        clock.tick(15)


if __name__ == "__main__":
    main()
