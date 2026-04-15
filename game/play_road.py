"""
Interactive Road Driving — curved road with moving NPC cars.

Controls:
    ARROW KEYS    — move in any direction (or stay still)
    SPACE         — toggle expert auto-pilot (road follower)
    R             — restart
    Q / ESC       — quit

Usage:
    python play_road.py                 # human plays
    python play_road.py --auto          # watch the expert drive
"""

import argparse
import pygame
from road_driving import (
    RoadDrivingGame,
    road_expert_action,
    DISPLAY_W,
    DISPLAY_H,
    FOV_W,
)
from renderer import GameRenderer

GOAL_DISTANCE = 40   # cells ahead


def _new_goal(game, from_y):
    gy = from_y - GOAL_DISTANCE
    gx = game._road_center_x(gy)
    return gy, gx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true", help="Start with expert auto-pilot")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((DISPLAY_W, DISPLAY_H))
    pygame.display.set_caption("Road Driving")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 18, bold=True)
    font_big = pygame.font.SysFont("monospace", 36, bold=True)

    renderer = GameRenderer()
    game = RoadDrivingGame(seed=args.seed)
    game.reset()

    gy, gx = _new_goal(game, game.car_y)
    trail = []
    goals_reached = 0
    auto_pilot = args.auto
    running = True

    while running:
        ax, ay = 0.0, 0.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                if event.key == pygame.K_SPACE:
                    auto_pilot = not auto_pilot
                if event.key == pygame.K_r:
                    game.reset()
                    gy, gx = _new_goal(game, game.car_y)
                    trail.clear()
                    goals_reached = 0
                    continue

        if auto_pilot:
            ax, ay = road_expert_action(game)
        else:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:   ax = -1.0
            elif keys[pygame.K_RIGHT]: ax = 1.0
            if keys[pygame.K_UP]:     ay = -1.0
            elif keys[pygame.K_DOWN]: ay = 1.0

        _, done, info = game.step(ax, ay)
        trail.append((game.car_y, game.car_x))

        dist = abs(game.car_y - gy) + abs(game.car_x - gx)
        if dist <= 3:
            goals_reached += 1
            gy, gx = _new_goal(game, game.car_y)

        renderer.draw(screen, game)
        renderer.draw_path(screen, game, trail[-80:], backtracking=False)
        renderer.draw_goal(screen, game, gy, gx, reached=(dist <= 3))

        mode = "AUTO" if auto_pilot else "MANUAL"
        hud = font.render(
            f" Score:{info['score']}  Steps:{game.steps}  Goals:{goals_reached}  "
            f"NPCs:{len(game.npc_cars())}  [{mode}]  SPACE=auto  R=reset",
            True, (255, 255, 200),
        )
        bar = pygame.Surface((DISPLAY_W, 28), pygame.SRCALPHA)
        bar.fill((0, 0, 0, 120))
        screen.blit(bar, (0, 0))
        screen.blit(hud, (6, 5))

        renderer.draw_grid_preview(
            screen, game.get_grid(),
            x=DISPLAY_W - FOV_W * 4 - 10, y=32, scale=4,
        )

        if done:
            overlay = pygame.Surface((DISPLAY_W, DISPLAY_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 100))
            screen.blit(overlay, (0, 0))
            crash = font_big.render("CRASH!", True, (255, 70, 70))
            screen.blit(crash, crash.get_rect(center=(DISPLAY_W // 2, DISPLAY_H // 2 - 20)))
            sub = font.render(
                f"Score: {info['score']}  Goals: {goals_reached}  —  R to restart",
                True, (220, 220, 220),
            )
            screen.blit(sub, sub.get_rect(center=(DISPLAY_W // 2, DISPLAY_H // 2 + 25)))

        pygame.display.flip()
        clock.tick(10)

        if done:
            _wait_for_key(clock, game, trail)
            gy, gx = _new_goal(game, game.car_y)
            goals_reached = 0

    pygame.quit()


def _wait_for_key(clock, game, trail):
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game.reset()
                    trail.clear()
                    return
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit()
                    raise SystemExit
        clock.tick(15)


if __name__ == "__main__":
    main()
