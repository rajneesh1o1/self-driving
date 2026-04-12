"""
Beautiful pygame renderer for the self-driving game.

Draws cute car, trees, containers, and textured ground.
The model still sees the plain pixel grid — this is purely cosmetic.
"""

import pygame
import math
from car_game import FOV_H, FOV_W, CAR_SIZE, CELL_SIZE, DISPLAY_W, DISPLAY_H

# ── colour palette ──────────────────────────────────────────────

GROUND_BASE = (88, 130, 68)

TREE_GREENS = [
    (40, 135, 55),
    (50, 150, 60),
    (35, 120, 48),
    (55, 142, 52),
    (45, 128, 58),
]
TRUNK_COLOR = (105, 75, 45)

CONTAINER_COLORS = [
    (195, 65, 55),   # red
    (55, 90, 175),   # blue
    (200, 155, 50),  # yellow
    (140, 95, 60),   # wood
    (110, 115, 125), # metal
]

CAR_BODY      = (55, 170, 240)
CAR_BODY_DARK = (40, 140, 200)
CAR_ROOF      = (75, 190, 255)
CAR_WINDOW    = (190, 225, 255)
CAR_WHEEL     = (35, 35, 35)
CAR_WHEEL_RIM = (80, 80, 80)
CAR_HEADLIGHT = (255, 245, 120)
CAR_TAILLIGHT = (255, 45, 45)

SHADOW = (45, 70, 35)


class GameRenderer:
    """Draws the game beautifully using pygame primitives."""

    def __init__(self):
        # pre-render a tiling ground tile (CELL_SIZE x CELL_SIZE)
        self._ground_cache = {}

    # ════════════════════════════════════════════════════════════
    #  Main entry point
    # ════════════════════════════════════════════════════════════

    def draw(self, screen, game):
        """Draw the full scene for the current FOV."""
        fov_top, fov_left = self._fov_origin(game)

        self._draw_ground(screen, game, fov_top, fov_left)
        self._draw_obstacles(screen, game, fov_top, fov_left)
        self._draw_car(screen)

    # ════════════════════════════════════════════════════════════
    #  Ground
    # ════════════════════════════════════════════════════════════

    def _draw_ground(self, screen, game, fov_top, fov_left):
        cs = CELL_SIZE
        for r in range(FOV_H):
            for c in range(FOV_W):
                wy = fov_top + r
                wx = fov_left + c
                # deterministic per-cell shade
                h = ((wy * 17 + wx * 31 + 7) % 21) - 10
                col = (
                    max(0, min(255, GROUND_BASE[0] + h)),
                    max(0, min(255, GROUND_BASE[1] + h)),
                    max(0, min(255, GROUND_BASE[2] + h)),
                )
                pygame.draw.rect(screen, col, (c * cs, r * cs, cs, cs))

                # tiny grass tufts (2 per cell, deterministic)
                seed = (wy * 997 + wx * 313) & 0xFFFFFF
                for _ in range(2):
                    gx = c * cs + (seed % (cs - 6)) + 3
                    seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
                    gy = r * cs + (seed % (cs - 6)) + 3
                    seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
                    gl = 3 + seed % 5
                    seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
                    gc = (
                        max(0, col[0] - 15 + seed % 10),
                        min(255, col[1] + 5 + seed % 15),
                        max(0, col[2] - 10 + seed % 8),
                    )
                    seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
                    pygame.draw.line(screen, gc, (gx, gy), (gx, gy - gl), 1)

    # ════════════════════════════════════════════════════════════
    #  Obstacles
    # ════════════════════════════════════════════════════════════

    def _draw_obstacles(self, screen, game, fov_top, fov_left):
        cs = CELL_SIZE
        for r in range(FOV_H):
            for c in range(FOV_W):
                wy = fov_top + r
                wx = fov_left + c

                val = game.world_val(wy, wx)
                if val < -0.5:
                    otype = game.obs_type_at(wy, wx)
                    ocolor = game.obs_color_at(wy, wx)
                    if otype == 1:
                        self._draw_tree(screen, c * cs, r * cs, ocolor)
                    else:
                        self._draw_container(screen, c * cs, r * cs, ocolor)

    def _draw_tree(self, screen, x, y, color_idx):
        cs = CELL_SIZE
        green = TREE_GREENS[color_idx % len(TREE_GREENS)]
        lighter = tuple(min(255, c + 35) for c in green)
        darker = tuple(max(0, c - 25) for c in green)

        cx, cy = x + cs // 2, y + cs // 2
        radius = cs // 2 - 3

        # shadow
        pygame.draw.circle(screen, SHADOW, (cx + 3, cy + 3), radius)
        # trunk (visible behind canopy)
        pygame.draw.rect(screen, TRUNK_COLOR, (cx - 3, cy, 6, cs // 2 - 2))
        # main canopy
        pygame.draw.circle(screen, green, (cx, cy - 2), radius)
        # inner highlight
        pygame.draw.circle(screen, lighter, (cx - 3, cy - 5), radius // 2)
        # outline
        pygame.draw.circle(screen, darker, (cx, cy - 2), radius, 1)

    def _draw_container(self, screen, x, y, color_idx):
        cs = CELL_SIZE
        color = CONTAINER_COLORS[color_idx % len(CONTAINER_COLORS)]
        darker = tuple(max(0, c - 50) for c in color)
        lighter = tuple(min(255, c + 40) for c in color)

        pad = 2
        bx, by, bw, bh = x + pad, y + pad, cs - 2 * pad, cs - 2 * pad

        # shadow
        pygame.draw.rect(screen, SHADOW, (bx + 3, by + 3, bw, bh), border_radius=3)
        # main body
        pygame.draw.rect(screen, color, (bx, by, bw, bh), border_radius=3)
        # top bevel highlight
        pygame.draw.rect(screen, lighter, (bx + 2, by + 2, bw - 4, bh // 3), border_radius=2)
        # border
        pygame.draw.rect(screen, darker, (bx, by, bw, bh), 2, border_radius=3)
        # cross straps
        pygame.draw.line(screen, darker, (bx + 4, by + 4), (bx + bw - 4, by + bh - 4), 2)
        pygame.draw.line(screen, darker, (bx + bw - 4, by + 4), (bx + 4, by + bh - 4), 2)

    def _draw_rock(self, screen, x, y, color_idx):
        """Boundary obstacle — simple grey rock."""
        cs = CELL_SIZE
        cx, cy = x + cs // 2, y + cs // 2
        pygame.draw.circle(screen, (70, 70, 65), (cx + 2, cy + 2), cs // 2 - 2)
        pygame.draw.circle(screen, (110, 108, 100), (cx, cy), cs // 2 - 2)
        pygame.draw.circle(screen, (130, 128, 120), (cx - 2, cy - 3), cs // 3)

    # ════════════════════════════════════════════════════════════
    #  Car  (always drawn at centre of FOV)
    # ════════════════════════════════════════════════════════════

    def _draw_car(self, screen):
        # car occupies 2x2 cells at centre of FOV
        car_r = FOV_H // 2 - CAR_SIZE // 2
        car_c = FOV_W // 2 - CAR_SIZE // 2
        x = car_c * CELL_SIZE
        y = car_r * CELL_SIZE
        w = CAR_SIZE * CELL_SIZE   # 80
        h = CAR_SIZE * CELL_SIZE   # 80

        # ── shadow ──
        shadow_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, (0, 0, 0, 50), (6, 6, w - 6, h - 6))
        screen.blit(shadow_surf, (x + 3, y + 3))

        # ── wheels (behind body) ──
        ww, wh = 10, 20
        for wx, wy in [
            (x + 2,      y + 10),
            (x + w - 12, y + 10),
            (x + 2,      y + h - 30),
            (x + w - 12, y + h - 30),
        ]:
            pygame.draw.rect(screen, CAR_WHEEL, (wx, wy, ww, wh), border_radius=3)
            pygame.draw.rect(screen, CAR_WHEEL_RIM, (wx + 1, wy + wh // 2 - 2, ww - 2, 4), border_radius=1)

        # ── body ──
        bx, by, bw, bh = x + 10, y + 4, w - 20, h - 8
        pygame.draw.rect(screen, CAR_BODY, (bx, by, bw, bh), border_radius=14)
        # lower body accent
        pygame.draw.rect(screen, CAR_BODY_DARK, (bx + 2, by + bh // 2, bw - 4, bh // 2 - 2), border_radius=10)

        # ── roof ──
        rx, ry, rw, rh = bx + 6, by + 14, bw - 12, bh - 28
        pygame.draw.rect(screen, CAR_ROOF, (rx, ry, rw, rh), border_radius=8)

        # ── windshield (front = top) ──
        pygame.draw.rect(screen, CAR_WINDOW, (bx + 8, by + 6, bw - 16, 18), border_radius=6)

        # ── rear window ──
        pygame.draw.rect(screen, CAR_WINDOW, (bx + 12, by + bh - 22, bw - 24, 14), border_radius=5)

        # ── headlights ──
        pygame.draw.circle(screen, CAR_HEADLIGHT, (bx + 10, by + 3), 5)
        pygame.draw.circle(screen, CAR_HEADLIGHT, (bx + bw - 10, by + 3), 5)
        # glow
        glow = pygame.Surface((14, 14), pygame.SRCALPHA)
        pygame.draw.circle(glow, (255, 245, 120, 60), (7, 7), 7)
        screen.blit(glow, (bx + 10 - 7, by + 3 - 7))
        screen.blit(glow, (bx + bw - 10 - 7, by + 3 - 7))

        # ── taillights ──
        pygame.draw.circle(screen, CAR_TAILLIGHT, (bx + 10, by + bh - 3), 4)
        pygame.draw.circle(screen, CAR_TAILLIGHT, (bx + bw - 10, by + bh - 3), 4)

        # ── side mirrors ──
        pygame.draw.ellipse(screen, CAR_BODY_DARK, (x + 6, y + h // 2 - 5, 8, 10))
        pygame.draw.ellipse(screen, CAR_BODY_DARK, (x + w - 14, y + h // 2 - 5, 8, 10))

    # ════════════════════════════════════════════════════════════
    #  Grid preview (model's view)
    # ════════════════════════════════════════════════════════════

    def draw_grid_preview(self, screen, grid, x, y, scale=4):
        """Tiny overlay showing what the model sees."""
        # background
        bg = pygame.Rect(x - 2, y - 2, FOV_W * scale + 4, FOV_H * scale + 4)
        pygame.draw.rect(screen, (20, 20, 20), bg, border_radius=3)

        for r in range(FOV_H):
            for c in range(FOV_W):
                val = grid[r, c]
                if val > 0.5:
                    color = (0, 255, 100)
                elif val < -0.5:
                    color = (255, 60, 60)
                else:
                    color = (35, 35, 35)
                pygame.draw.rect(screen, color, (x + c * scale, y + r * scale, scale, scale))

    # ════════════════════════════════════════════════════════════
    #  Navigation overlays (DFS path, goal, visited)
    # ════════════════════════════════════════════════════════════

    def draw_visited(self, screen, game, visited):
        """Subtle tint on cells the DFS has explored."""
        fov_top, fov_left = self._fov_origin(game)
        cs = CELL_SIZE
        cell_overlay = pygame.Surface((cs, cs), pygame.SRCALPHA)
        cell_overlay.fill((100, 200, 255, 25))
        for r in range(FOV_H):
            for c in range(FOV_W):
                if (fov_top + r, fov_left + c) in visited:
                    screen.blit(cell_overlay, (c * cs, r * cs))

    def draw_path(self, screen, game, path, backtracking=False):
        """Draw the DFS path as connected dots."""
        if not path:
            return
        fov_top, fov_left = self._fov_origin(game)
        cs = CELL_SIZE
        color = (255, 140, 60) if backtracking else (100, 200, 255)
        dim = tuple(max(0, c - 60) for c in color)

        pts = []
        for py, px in path:
            sx = int((px - fov_left) * cs + cs // 2)
            sy = int((py - fov_top) * cs + cs // 2)
            if 0 <= sx < DISPLAY_W and 0 <= sy < DISPLAY_H:
                pts.append((sx, sy))

        if len(pts) > 1:
            pygame.draw.lines(screen, dim, False, pts, 2)
        for pt in pts:
            pygame.draw.circle(screen, color, pt, 3)

    def draw_goal(self, screen, game, goal_y, goal_x, reached=False):
        """Draw a flag at the goal, or an arrow at screen edge if off-screen."""
        fov_top, fov_left = self._fov_origin(game)
        cs = CELL_SIZE
        sx = (goal_x - fov_left) * cs + cs // 2
        sy = (goal_y - fov_top) * cs + cs // 2

        if 0 <= sx < DISPLAY_W and 0 <= sy < DISPLAY_H:
            flag_col = (0, 255, 0) if reached else (255, 215, 0)
            # pole
            pygame.draw.line(screen, (200, 200, 200),
                             (int(sx), int(sy) + 15), (int(sx), int(sy) - 20), 3)
            # flag
            pygame.draw.polygon(screen, flag_col, [
                (int(sx) + 2, int(sy) - 20),
                (int(sx) + 18, int(sy) - 13),
                (int(sx) + 2, int(sy) - 6),
            ])
            # base
            pygame.draw.circle(screen, (200, 200, 200), (int(sx), int(sy) + 15), 5)
        else:
            # off-screen arrow pointing toward goal
            cx, cy = DISPLAY_W // 2, DISPLAY_H // 2
            dx, dy = sx - cx, sy - cy
            length = math.sqrt(dx * dx + dy * dy)
            if length < 1:
                return
            dx, dy = dx / length, dy / length
            margin = 30
            ax = max(margin, min(DISPLAY_W - margin,
                                 int(cx + dx * (DISPLAY_W // 2 - margin))))
            ay = max(margin, min(DISPLAY_H - margin,
                                 int(cy + dy * (DISPLAY_H // 2 - margin))))
            angle = math.atan2(dy, dx)
            sz = 12
            pts = [
                (ax + sz * math.cos(angle),
                 ay + sz * math.sin(angle)),
                (ax + sz * math.cos(angle + 2.4),
                 ay + sz * math.sin(angle + 2.4)),
                (ax + sz * math.cos(angle - 2.4),
                 ay + sz * math.sin(angle - 2.4)),
            ]
            pygame.draw.polygon(screen, (255, 215, 0), pts)

    # ════════════════════════════════════════════════════════════
    #  Helpers
    # ════════════════════════════════════════════════════════════

    @staticmethod
    def _fov_origin(game):
        fov_cy = game.car_y + CAR_SIZE // 2
        fov_cx = game.car_x + CAR_SIZE // 2
        fov_top = fov_cy - FOV_H // 2
        fov_left = fov_cx - FOV_W // 2
        return fov_top, fov_left
