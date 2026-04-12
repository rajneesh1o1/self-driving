"""
Interactive Car Game — drive with arrow keys or watch DFS navigation.

Controls:
    ARROW KEYS    — move in any direction (or stay still)
    SPACE         — toggle DFS auto-pilot
    R             — restart
    Q / ESC       — quit

The DFS navigator steers toward a goal flag using depth-first search.
When blocked it backtracks and tries alternative routes.  Among open
neighbours it picks the one whose move vector points most directly
toward the destination (minimum Euclidean distance).

Usage:
    python play.py              # human plays
    python play.py --auto       # watch DFS navigate
"""

import argparse
import pygame
from car_game import (
    CarGame,
    DFSNavigator,
    DISPLAY_W,
    DISPLAY_H,
    FOV_H,
    FOV_W,
)
from renderer import GameRenderer

GOAL_DISTANCE = 50   # cells north per goal


def _new_goal(game, base_y):
    """Place the next goal along the corridor."""
    gy = base_y - GOAL_DISTANCE
    gx = game._corridor_center_x(gy)
    return gy, gx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true", help="Start with DFS auto-pilot")
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((DISPLAY_W, DISPLAY_H))
    pygame.display.set_caption("Self-Driving Car — DFS Navigation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 18, bold=True)
    font_big = pygame.font.SysFont("monospace", 36, bold=True)

    renderer = GameRenderer()
    game = CarGame()
    game.reset()

    # ── DFS navigator ──
    gy, gx = _new_goal(game, game.car_y)
    navigator = DFSNavigator(goal_y=gy, goal_x=gx)
    navigator.reset(start_y=game.car_y, start_x=game.car_x)

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
                    navigator = DFSNavigator(goal_y=gy, goal_x=gx)
                    navigator.reset(start_y=game.car_y, start_x=game.car_x)
                    continue

        # ── action ──
        if auto_pilot:
            # advance goal when reached
            if navigator.reached:
                gy, gx = _new_goal(game, navigator.goal_y)
                navigator.reset(
                    start_y=game.car_y, start_x=game.car_x,
                    goal_y=gy, goal_x=gx,
                )
            ax, ay = navigator.next_action(game)
        else:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                ax = -1.0
            elif keys[pygame.K_RIGHT]:
                ax = 1.0
            if keys[pygame.K_UP]:
                ay = -1.0
            elif keys[pygame.K_DOWN]:
                ay = 1.0

        grid, done, info = game.step(ax, ay)

        # ── draw scene ──
        renderer.draw(screen, game)

        # ── DFS overlays ──
        renderer.draw_visited(screen, game, navigator.visited)
        renderer.draw_path(screen, game, navigator.path,
                           backtracking=navigator.is_backtracking)
        renderer.draw_goal(screen, game,
                           navigator.goal_y, navigator.goal_x,
                           reached=navigator.reached)

        # ── HUD ──
        if auto_pilot:
            dist = abs(game.car_y - navigator.goal_y) + abs(game.car_x - navigator.goal_x)
            status = "BACKTRACK" if navigator.is_backtracking else "EXPLORING"
            mode_str = f"DFS  dist:{dist}  {status}"
        else:
            mode_str = "MANUAL"
        _draw_hud(screen, font, info["score"], game.steps, mode_str)

        # ── grid preview (model's view) ──
        renderer.draw_grid_preview(
            screen, grid,
            x=DISPLAY_W - FOV_W * 4 - 10,
            y=32,
            scale=4,
        )

        if done:
            overlay = pygame.Surface((DISPLAY_W, DISPLAY_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 100))
            screen.blit(overlay, (0, 0))

            crash = font_big.render("CRASH!", True, (255, 70, 70))
            screen.blit(crash, crash.get_rect(center=(DISPLAY_W // 2, DISPLAY_H // 2 - 20)))
            sub = font.render(
                f"Score: {info['score']}   Press R to restart",
                True, (220, 220, 220),
            )
            screen.blit(sub, sub.get_rect(center=(DISPLAY_W // 2, DISPLAY_H // 2 + 25)))

        pygame.display.flip()
        clock.tick(10)

        if done:
            _wait_for_key(clock, game, navigator)

    pygame.quit()


def _draw_hud(screen, font, score, steps, mode):
    bar = pygame.Surface((DISPLAY_W, 28), pygame.SRCALPHA)
    bar.fill((0, 0, 0, 120))
    screen.blit(bar, (0, 0))
    txt = font.render(
        f" Score: {score}    Steps: {steps}    [{mode}]  SPACE=auto  R=restart",
        True, (255, 255, 200),
    )
    screen.blit(txt, (6, 5))


def _wait_for_key(clock, game, navigator):
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game.reset()
                    gy, gx = _new_goal(game, game.car_y)
                    navigator.reset(
                        start_y=game.car_y, start_x=game.car_x,
                        goal_y=gy, goal_x=gx,
                    )
                    return
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit()
                    raise SystemExit
        clock.tick(15)


if __name__ == "__main__":
    main()
