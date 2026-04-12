"""
Model-driven car: the trained JEPA world model navigates through obstacles.

The model plans by predicting future embeddings for candidate action sequences
and picking the one whose predicted future is most stable.

Usage:
    python drive.py --checkpoint model/checkpoints/driving_jepa_epoch_20.pt
    python drive.py --checkpoint model/checkpoints/driving_jepa_epoch_20.pt --expert-compare
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import pygame

sys.path.insert(0, str(Path(__file__).parent / "game"))
sys.path.insert(0, str(Path(__file__).parent / "model"))

from car_game import (
    CarGame,
    expert_action,
    DISPLAY_W,
    DISPLAY_H,
    GRID_H,
    GRID_W,
    FOV_H,
    FOV_W,
)
from renderer import GameRenderer


def load_model(ckpt_path, device):
    model = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.eval()
    model.requires_grad_(False)
    return model


def plan_action(model, grid_np, device, horizon=3, history_size=1):
    """Try all 9 direction combos, pick the safest multi-step plan."""
    with torch.no_grad():
        grid = torch.from_numpy(grid_np).float().unsqueeze(0).unsqueeze(0).to(device)
        info = model.encode({"grid": grid})
        init_emb = info["emb"]  # (1, 1, D)

        # 9 candidate constant actions: all (dx, dy) combos
        candidates = []
        for dx in [-1.0, 0.0, 1.0]:
            for dy in [-1.0, 0.0, 1.0]:
                candidates.append([dx, dy])

        S = len(candidates)
        act_seq = torch.tensor(candidates).view(1, S, 1, 2).expand(1, S, horizon, 2).to(device)

        cost = model.plan_cost(init_emb, act_seq, history_size=history_size)  # (1, S)
        best_idx = cost[0].argmin().item()
        return candidates[best_idx], cost[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--expert-compare", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    print(f"Loading: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    pygame.init()
    W = DISPLAY_W * 2 + 20 if args.expert_compare else DISPLAY_W
    screen = pygame.display.set_mode((W, DISPLAY_H))
    title = "Self-Driving — MODEL vs EXPERT" if args.expert_compare else "Self-Driving — MODEL"
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 20)
    font_big = pygame.font.SysFont("monospace", 32, bold=True)

    renderer = GameRenderer()

    game_model = CarGame(seed=42)
    game_model.reset()

    game_expert = CarGame(seed=42) if args.expert_compare else None
    if game_expert:
        game_expert.reset()

    model_scores, expert_scores = [], []
    running = True
    game_num = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                running = False

        # --- model drives ---
        grid_m = game_model.get_grid()
        (ax_m, ay_m), costs = plan_action(model, grid_m, device, horizon=args.horizon)
        grid_m, done_m, info_m = game_model.step(ax_m, ay_m)

        # --- expert drives ---
        if game_expert:
            ax_e, ay_e = expert_action(game_expert)
            _, done_e, info_e = game_expert.step(ax_e, ay_e)

        # --- draw model ---
        renderer.draw(screen, game_model)
        hud = font.render(f"MODEL  Score: {info_m['score']}  Steps: {game_model.steps}", True, (255, 255, 0))
        screen.blit(hud, (10, 8))

        # grid preview
        renderer.draw_grid_preview(screen, grid_m, x=DISPLAY_W - FOV_W * 4 - 8, y=30, scale=4)

        if done_m:
            game_num += 1
            model_scores.append(info_m["score"])
            avg_m = np.mean(model_scores[-20:])
            print(f"Game {game_num} | MODEL score: {info_m['score']}  avg: {avg_m:.0f}", end="")
            if game_expert and done_e:
                expert_scores.append(info_e["score"])
                avg_e = np.mean(expert_scores[-20:])
                print(f"  | EXPERT: {info_e['score']}  avg: {avg_e:.0f}", end="")
            print()
            go = font_big.render(f"CRASH  Score: {info_m['score']}", True, (255, 60, 60))
            screen.blit(go, go.get_rect(center=(DISPLAY_W // 2, DISPLAY_H // 2)))
            game_model.reset()
            if game_expert:
                game_expert.reset()

        # --- draw expert ---
        if game_expert:
            # draw expert view offset to the right
            expert_surf = pygame.Surface((DISPLAY_W, DISPLAY_H))
            renderer.draw(expert_surf, game_expert)
            screen.blit(expert_surf, (DISPLAY_W + 20, 0))
            hud_e = font.render(f"EXPERT  Score: {info_e['score']}", True, (100, 255, 100))
            screen.blit(hud_e, (DISPLAY_W + 30, 8))
            if done_e and not done_m:
                game_expert.reset()

        pygame.display.flip()
        clock.tick(8)

    pygame.quit()
    if model_scores:
        print(f"\nMODEL  — games: {len(model_scores)}  avg: {np.mean(model_scores):.0f}")
        if expert_scores:
            print(f"EXPERT — games: {len(expert_scores)}  avg: {np.mean(expert_scores):.0f}")


def _draw_grid(screen, grid, x, y, scale=4):
    for r in range(FOV_H):
        for c in range(FOV_W):
            val = grid[r, c]
            if val > 0.5:
                color = (0, 255, 100)
            elif val < -0.5:
                color = (255, 60, 60)
            else:
                color = (30, 30, 30)
            pygame.draw.rect(screen, color, (x + c * scale, y + r * scale, scale, scale))


if __name__ == "__main__":
    main()
