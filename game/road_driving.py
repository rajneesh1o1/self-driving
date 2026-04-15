"""
Road Driving — curved road with moving NPC cars.

Map:
  - A curved corridor ("road") winding along the -y direction.
  - Both sides of the road are impassable walls.
  - Other cars (NPCs) drive forward at 0.5 cells/step (1 cell every 2 steps).
  - Our car moves 1 cell/step via the same 8-directional action as CarGame,
    so it can overtake NPCs or stay in a lane.
  - Goal is placed ahead along the road.

Exposes the same API as CarGame so the trained policy + renderer can be
used without changes:
    reset(seed) -> grid
    step(ax, ay) -> (grid, done, info)
    get_local_fov() -> (LOCAL_FOV, LOCAL_FOV)
    get_grid()      -> (FOV_H, FOV_W)
    world_val(y,x), obs_type_at(y,x), obs_color_at(y,x)

Extra (used by the renderer for prettier road visuals):
    is_road(y, x) -> bool       — cell is inside the road corridor
    npc_cars()    -> list[(y,x)]
"""

import math
import numpy as np

from car_game import (
    CHUNK_SIZE, FOV_H, FOV_W, CAR_SIZE, VISIBILITY, LOCAL_FOV,
    CELL_SIZE, DISPLAY_W, DISPLAY_H,
)

# ─── Road geometry ─────────────────────────────────────────────
ROAD_HALF_WIDTH = 5          # cells from centerline to wall (road total ~10 wide)
FORWARD_DY      = -1         # "forward" = -y direction

# ─── NPC cars ──────────────────────────────────────────────────
NPC_STEP_EVERY     = 2       # NPC advances 1 cell every N player steps (0.5 pace)
NPC_SPAWN_EVERY    = 6       # attempt to spawn every N steps
NPC_SPAWN_AHEAD    = 25      # spawn this many cells ahead of our car
NPC_SPAWN_JITTER   = 20      # extra random forward offset on spawn
NPC_DESPAWN_BEHIND = 15      # remove NPCs that fall this far behind
MAX_NPC            = 6
MAX_NPC_PER_BAND   = 2       # max NPCs in any Y-band (prevents impassable walls)
NPC_BAND_HEIGHT    = 4       # Y-band size for density check (2× car height)

# ─── Obstacle type codes (read by renderer) ────────────────────
TYPE_WALL    = 5             # side-of-road wall
TYPE_NPC_CAR = 6             # moving other-car


class RoadDrivingGame:
    """Curved-road driving world with moving NPC cars."""

    # ================================================================
    #  Public API
    # ================================================================

    def __init__(self, seed=None):
        self._base_seed = seed if seed is not None else 42
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self._base_seed = seed
        self._rng = np.random.default_rng(self._base_seed)

        # Start on road center, heading forward (-y)
        self.car_y = 0
        self.car_x = self._road_center_x(0) - CAR_SIZE // 2

        self.done = False
        self.steps = 0
        self.score = 0
        self._npcs = []                # list[(y, x)] — each NPC is 2x2 anchored at top-left

        # Seed the road ahead with a few NPCs so gameplay starts lively
        for _ in range(4):
            self._try_spawn_npc()

        return self.get_grid()

    def step(self, action_x, action_y):
        if self.done:
            return self.get_grid(), True, {"score": self.score, "collision": True}

        action_x = float(np.clip(action_x, -1.0, 1.0))
        action_y = float(np.clip(action_y, -1.0, 1.0))

        new_x, new_y = self.car_x, self.car_y
        if action_x < -0.3:   new_x -= 1
        elif action_x > 0.3:  new_x += 1
        if action_y < -0.3:   new_y -= 1
        elif action_y > 0.3:  new_y += 1

        moved = (new_x != self.car_x or new_y != self.car_y)

        # Player move check
        if moved and self._collides(new_y, new_x):
            self.done = True
            return self.get_grid(), True, {"score": self.score, "collision": True}

        if moved:
            self.car_y = new_y
            self.car_x = new_x

        self.steps += 1
        self.score = max(self.score, -self.car_y)

        # Advance NPCs (they move at 0.5 cells/step = 1 cell every 2 steps)
        if self.steps % NPC_STEP_EVERY == 0:
            self._advance_npcs()
            self._break_blockades()

        # Spawn / despawn housekeeping
        if self.steps % NPC_SPAWN_EVERY == 0:
            self._try_spawn_npc()
        self._despawn_far_npcs()

        # NPC may have just crashed into us
        if self._cell_occupied_by_npc(self.car_y, self.car_x):
            self.done = True
            return self.get_grid(), True, {"score": self.score, "collision": True}

        return self.get_grid(), False, {"score": self.score, "collision": False}

    # ================================================================
    #  Representations (identical signatures to CarGame)
    # ================================================================

    def get_local_fov(self):
        size = LOCAL_FOV
        grid = np.zeros((size, size), dtype=np.float32)
        top = self.car_y - VISIBILITY
        left = self.car_x - VISIBILITY
        for r in range(size):
            for c in range(size):
                if self.world_val(top + r, left + c) < -0.5:
                    grid[r, c] = -1.0
        for dr in range(CAR_SIZE):
            for dc in range(CAR_SIZE):
                grid[VISIBILITY + dr, VISIBILITY + dc] = 1.0
        return grid

    def get_grid(self):
        fov_top = self.car_y + CAR_SIZE // 2 - FOV_H // 2
        fov_left = self.car_x + CAR_SIZE // 2 - FOV_W // 2
        grid = np.zeros((FOV_H, FOV_W), dtype=np.float32)
        for r in range(FOV_H):
            for c in range(FOV_W):
                grid[r, c] = self.world_val(fov_top + r, fov_left + c)
        car_r = FOV_H // 2 - CAR_SIZE // 2
        car_c = FOV_W // 2 - CAR_SIZE // 2
        for dr in range(CAR_SIZE):
            for dc in range(CAR_SIZE):
                grid[car_r + dr, car_c + dc] = 1.0
        return grid

    # ================================================================
    #  World accessors
    # ================================================================

    def world_val(self, y, x):
        if self._is_wall(y, x):
            return -1.0
        if self._cell_occupied_by_npc(y, x):
            return -1.0
        return 0.0

    def obs_type_at(self, y, x):
        if self._cell_occupied_by_npc(y, x):
            return TYPE_NPC_CAR
        if self._is_wall(y, x):
            return TYPE_WALL
        return 0

    def obs_color_at(self, y, x):
        if self._cell_occupied_by_npc(y, x):
            for i, (ny, nx, _) in enumerate(self._npcs):
                if ny <= y < ny + CAR_SIZE and nx <= x < nx + CAR_SIZE:
                    return i % 5
        return 0

    # Extras used by the renderer
    def is_road(self, y, x):
        """Cell is inside the driveable corridor (between the walls)."""
        cx = self._road_center_x(y)
        return abs(x - cx) < ROAD_HALF_WIDTH

    def npc_cars(self):
        """Public view: list of (y, x) for rendering. Internally we also
        track each NPC's lane offset from the road center so it turns with
        the road (see `_npcs` entries: (y, x, lane))."""
        return [(y, x) for (y, x, _) in self._npcs]

    # ================================================================
    #  Road geometry
    # ================================================================

    def _road_center_x(self, y):
        """Smooth deterministic road centerline winding around x=0."""
        s = self._base_seed * 0.1
        x = 0.0
        x += 12.0 * math.sin(y * 0.035 + s)
        x += 6.0  * math.sin(y * 0.091 + s * 3.7)
        x += 3.0  * math.sin(y * 0.21  + s * 7.3)
        return int(round(x))

    def _is_wall(self, y, x):
        cx = self._road_center_x(y)
        return abs(x - cx) >= ROAD_HALF_WIDTH

    # ================================================================
    #  NPC management
    # ================================================================

    def _cell_occupied_by_npc(self, y, x):
        for (ny, nx, _) in self._npcs:
            if ny <= y < ny + CAR_SIZE and nx <= x < nx + CAR_SIZE:
                return True
        return False

    def _cells_free_for_npc(self, y, x, skip_idx=-1):
        """Check a candidate NPC 2x2 placement against walls, our car, other NPCs."""
        for dr in range(CAR_SIZE):
            for dc in range(CAR_SIZE):
                if self._is_wall(y + dr, x + dc):
                    return False
        # our car (2x2) overlap check
        if abs(self.car_y - y) < CAR_SIZE and abs(self.car_x - x) < CAR_SIZE:
            return False
        # other NPCs
        for i, (oy, ox, _) in enumerate(self._npcs):
            if i == skip_idx:
                continue
            if abs(oy - y) < CAR_SIZE and abs(ox - x) < CAR_SIZE:
                return False
        return True

    def _x_for_lane(self, y, lane):
        """Top-left x of a 2x2 NPC positioned in the given lane at row y."""
        anchor = y + CAR_SIZE // 2
        return self._road_center_x(anchor) + lane - CAR_SIZE // 2

    def _try_spawn_npc(self):
        if len(self._npcs) >= MAX_NPC:
            return
        ny = self.car_y + FORWARD_DY * (NPC_SPAWN_AHEAD + int(self._rng.integers(0, NPC_SPAWN_JITTER)))
        # Density check: don't spawn into a Y-band that already has enough NPCs
        band_count = sum(1 for (y, _, _) in self._npcs if abs(y - ny) < NPC_BAND_HEIGHT)
        if band_count >= MAX_NPC_PER_BAND:
            return
        # choose a lane offset (in cells, measured from road center to car center)
        lo = -(ROAD_HALF_WIDTH - CAR_SIZE)
        hi = (ROAD_HALF_WIDTH - CAR_SIZE)
        lane = int(self._rng.integers(lo, hi + 1))
        nx = self._x_for_lane(ny, lane)
        if self._cells_free_for_npc(ny, nx):
            self._npcs.append((ny, nx, lane))

    def _advance_npcs(self):
        """Move each NPC 1 cell forward, hugging its lane as the road curves.

        Each NPC remembers its lane offset from the road center. When it
        advances forward by 1 cell, its x is recomputed from the (possibly
        shifted) center at the new y — so NPCs naturally turn with the road.
        If curvature demands a lateral step larger than 1 cell, we take it
        in one go; if that would crash into a wall/car we stay in place.
        """
        updated = []
        for i, (ny, nx, lane) in enumerate(self._npcs):
            new_y = ny + FORWARD_DY
            target_x = self._x_for_lane(new_y, lane)

            # Try: move forward into lane-correct x.
            # Fall back to: move forward but keep current x (if lane shift blocked).
            # Fall back to: stay (if forward is entirely blocked).
            candidates = [(new_y, target_x, lane)]
            if target_x != nx:
                candidates.append((new_y, nx, lane))
            candidates.append((ny, nx, lane))

            for cand in candidates:
                cy, cx, cl = cand
                if self._cells_free_for_npc(cy, cx, skip_idx=i):
                    updated.append(cand)
                    break
            else:
                updated.append((ny, nx, lane))
        self._npcs = updated

    def _break_blockades(self):
        """Remove excess NPCs when too many cluster in the same Y-band."""
        if len(self._npcs) < 3:
            return
        to_remove = set()
        for i in range(len(self._npcs)):
            if i in to_remove:
                continue
            y_i = self._npcs[i][0]
            band = [j for j in range(len(self._npcs))
                    if j != i and j not in to_remove
                    and abs(self._npcs[j][0] - y_i) < NPC_BAND_HEIGHT]
            total = 1 + len(band)
            while total > MAX_NPC_PER_BAND:
                # Remove the member closest to player (largest y, since forward=-y)
                worst = max(band, key=lambda j: self._npcs[j][0])
                to_remove.add(worst)
                band.remove(worst)
                total -= 1
        if to_remove:
            self._npcs = [npc for i, npc in enumerate(self._npcs)
                          if i not in to_remove]

    def _despawn_far_npcs(self):
        """Drop NPCs that are far behind the player or way too far ahead."""
        keep = []
        ahead_limit = NPC_SPAWN_AHEAD + NPC_SPAWN_JITTER + 10
        for (y, x, lane) in self._npcs:
            # Forward = -y, so "ahead" means y << car_y and "behind" means y > car_y.
            behind = y - self.car_y                       # >0 if behind
            ahead  = self.car_y - y                       # >0 if ahead
            if behind > NPC_DESPAWN_BEHIND:
                continue
            if ahead > ahead_limit:
                continue
            keep.append((y, x, lane))
        self._npcs = keep

    # ================================================================
    #  Internals
    # ================================================================

    def _collides(self, y, x):
        for dr in range(CAR_SIZE):
            for dc in range(CAR_SIZE):
                if self._is_wall(y + dr, x + dc):
                    return True
                if self._cell_occupied_by_npc(y + dr, x + dc):
                    return True
        return False


# ================================================================
#  Simple road-follower expert (for demos / data generation)
# ================================================================

def road_expert_action(game):
    """
    Road-follower expert.

    Strategy:
      1. Aim at the road centerline a few cells ahead (lookahead waypoint).
      2. Pick the action whose resulting position minimises distance to
         that waypoint, while rejecting any action that hits a wall or NPC
         car directly, and lightly penalising proximity to obstacles.

    Uses only the local FOV grid, so it still respects limited visibility.
    """
    LOOKAHEAD = 5              # cells ahead of car to aim at
    grid = game.get_grid()
    car_r = FOV_H // 2 - CAR_SIZE // 2
    car_c = FOV_W // 2 - CAR_SIZE // 2

    # Waypoint: the road centerline LOOKAHEAD cells ahead of us.
    aim_y = game.car_y + CAR_SIZE // 2 + FORWARD_DY * LOOKAHEAD
    aim_x = game._road_center_x(aim_y)

    best_score = float("inf")
    best = (0.0, -1.0)

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            tr, tc = car_r + dy, car_c + dx
            if tr < 0 or tr + CAR_SIZE > FOV_H or tc < 0 or tc + CAR_SIZE > FOV_W:
                continue
            # reject actions that step directly onto an obstacle
            if any(grid[tr + r, tc + c] < -0.5
                   for r in range(CAR_SIZE) for c in range(CAR_SIZE)):
                continue

            # distance from new car-center to the waypoint (primary signal)
            new_cy = game.car_y + dy + CAR_SIZE // 2
            new_cx = game.car_x + dx + CAR_SIZE // 2
            dist = math.hypot(new_cy - aim_y, new_cx - aim_x)

            # mild penalty for very close obstacles (1-cell ring only)
            close = 0
            for r in range(max(0, tr - 1), min(FOV_H, tr + CAR_SIZE + 1)):
                for c in range(max(0, tc - 1), min(FOV_W, tc + CAR_SIZE + 1)):
                    if grid[r, c] < -0.5:
                        close += 1

            # tie-break: prefer moving (avoid staying idle in one axis)
            score = dist + 0.4 * close + 0.02 * abs(dx)

            if score < best_score:
                best_score = score
                best = (float(dx), float(dy))

    return best


# ================================================================
#  Quick visual test
# ================================================================

if __name__ == "__main__":
    import pygame
    from renderer import GameRenderer

    pygame.init()
    screen = pygame.display.set_mode((DISPLAY_W, DISPLAY_H))
    pygame.display.set_caption("Road Driving — expert demo")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 18, bold=True)

    renderer = GameRenderer()
    game = RoadDrivingGame(seed=42)
    game.reset()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                running = False

        ax, ay = road_expert_action(game)
        _, done, info = game.step(ax, ay)

        if done:
            print(f"Crash! Score: {info['score']}")
            game.reset()

        renderer.draw(screen, game)
        txt = font.render(
            f"Score: {info['score']}  Steps: {game.steps}  NPCs: {len(game.npc_cars())}",
            True, (255, 255, 80),
        )
        bar = pygame.Surface((txt.get_width() + 12, 26), pygame.SRCALPHA)
        bar.fill((0, 0, 0, 130))
        screen.blit(bar, (4, 4))
        screen.blit(txt, (10, 6))
        pygame.display.flip()
        clock.tick(10)

    pygame.quit()
