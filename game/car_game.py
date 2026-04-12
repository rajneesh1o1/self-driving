"""
Simple 2D Self-Driving Game — infinite procedural world.

Obstacles of various sizes are scattered across an endless plane.
The car can move in any direction or stay still.

Model sees: a small grid (FOV_H x FOV_W) — the car's field of view.
  - 0.0  = empty
  - 1.0  = car (2x2 block)
  - -1.0 = obstacle

The world is generated in chunks on demand — no boundaries.
"""

import heapq
import math
import numpy as np

# ---------- Chunk system ----------
CHUNK_SIZE = 32

# ---------- Field of View (what the model sees) ----------
FOV_H = 16
FOV_W = 16

# ---------- Car ----------
CAR_SIZE = 2

# ---------- Visibility (what the car can sense) ----------
VISIBILITY = 2                              # cells from car outline
LOCAL_FOV = 2 * VISIBILITY + CAR_SIZE       # 6x6

# ---------- Visual rendering ----------
CELL_SIZE = 40
DISPLAY_W = FOV_W * CELL_SIZE   # 640
DISPLAY_H = FOV_H * CELL_SIZE   # 640

# Backward-compatible aliases
GRID_H = FOV_H
GRID_W = FOV_W

# Obstacle sizes used during generation
_OBS_SIZES = [
    (1, 1), (1, 2), (2, 1), (2, 2),
    (1, 3), (3, 1), (2, 3), (3, 2), (3, 3),
]


class CarGame:
    """Car freely navigates an infinite procedural obstacle field."""

    def __init__(self, seed=None):
        self._base_seed = seed if seed is not None else 42
        self.reset()

    # ================================================================
    #  Public API
    # ================================================================

    def reset(self, seed=None):
        if seed is not None:
            self._base_seed = seed

        self._chunks = {}          # (cy, cx) -> {world, obs_type, obs_color}
        self.car_y = 0
        self.car_x = 0
        self.done = False
        self.steps = 0
        self.score = 0

        # clear a small area around spawn so the car never starts on an obstacle
        self._clear_area(self.car_y - 3, self.car_x - 3, 8, 8)

        return self.get_grid()

    def step(self, action_x, action_y):
        """
        Move the car.

        Args:
            action_x: float [-1,1].  < -0.3 left, > 0.3 right
            action_y: float [-1,1].  < -0.3 up,   > 0.3 down

        Returns:
            grid  (FOV_H, FOV_W) float32
            done  bool
            info  dict
        """
        if self.done:
            return self.get_grid(), True, {"score": self.score, "collision": True}

        action_x = float(np.clip(action_x, -1.0, 1.0))
        action_y = float(np.clip(action_y, -1.0, 1.0))

        new_x, new_y = self.car_x, self.car_y

        if action_x < -0.3:
            new_x -= 1
        elif action_x > 0.3:
            new_x += 1

        if action_y < -0.3:
            new_y -= 1
        elif action_y > 0.3:
            new_y += 1

        moved = (new_x != self.car_x or new_y != self.car_y)

        if moved and self._collides(new_y, new_x):
            self.done = True
            return self.get_grid(), True, {"score": self.score, "collision": True}

        if moved:
            self.car_x = new_x
            self.car_y = new_y
            self.steps += 1
            furthest = -self.car_y          # lower y = further north
            self.score = max(self.score, furthest)

        return self.get_grid(), False, {"score": self.score, "collision": False}

    # ================================================================
    #  Representations
    # ================================================================

    def get_local_fov(self):
        """What the car can actually see: (LOCAL_FOV, LOCAL_FOV) grid.

        Car outline + VISIBILITY cells in every direction.
        Values:  1.0 = car,  -1.0 = obstacle,  0.0 = empty.
        Car is always at the centre (rows/cols VISIBILITY..VISIBILITY+CAR_SIZE-1).
        """
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
        """Full render FOV grid centred on the car.  Shape (FOV_H, FOV_W)."""
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
    #  Cell accessors (infinite coordinates)
    # ================================================================

    def world_val(self, y, x):
        chunk = self._ensure_chunk(y // CHUNK_SIZE, x // CHUNK_SIZE)
        return float(chunk["world"][y % CHUNK_SIZE, x % CHUNK_SIZE])

    def obs_type_at(self, y, x):
        chunk = self._ensure_chunk(y // CHUNK_SIZE, x // CHUNK_SIZE)
        return int(chunk["obs_type"][y % CHUNK_SIZE, x % CHUNK_SIZE])

    def obs_color_at(self, y, x):
        chunk = self._ensure_chunk(y // CHUNK_SIZE, x // CHUNK_SIZE)
        return int(chunk["obs_color"][y % CHUNK_SIZE, x % CHUNK_SIZE])

    # ================================================================
    #  Chunk generation
    # ================================================================

    def _ensure_chunk(self, cy, cx):
        key = (cy, cx)
        if key not in self._chunks:
            self._generate_chunk(cy, cx)
        return self._chunks[key]

    def _corridor_center_x(self, world_y):
        """Smooth deterministic corridor winding around x = 0."""
        s = self._base_seed * 0.1
        x = 0.0
        x += 15.0 * math.sin(world_y * 0.05 + s)
        x += 8.0 * math.sin(world_y * 0.13 + s * 7)
        x += 4.0 * math.sin(world_y * 0.31 + s * 13)
        return int(round(x))

    def _generate_chunk(self, cy, cx):
        chunk_seed = abs(self._base_seed * 100003 + cy * 10007 + cx * 1003) % (2**31)
        rng = np.random.default_rng(chunk_seed)

        world = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32)
        obs_type = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.int8)
        obs_color = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.int8)

        base_y = cy * CHUNK_SIZE
        base_x = cx * CHUNK_SIZE

        # ── corridor cells (keep clear) ──
        corridor = set()
        corr_w = CAR_SIZE + 2          # tighter corridor (was +3)
        for ly in range(CHUNK_SIZE):
            center = self._corridor_center_x(base_y + ly)
            for dx in range(corr_w):
                lx = (center - corr_w // 2 + dx) - base_x
                if 0 <= lx < CHUNK_SIZE:
                    corridor.add((ly, lx))

        # ── scatter obstacles ──
        n_obs = int(CHUNK_SIZE * CHUNK_SIZE * 0.04)   # denser (was 0.04)
        for _ in range(n_obs):
            oh, ow = _OBS_SIZES[int(rng.integers(0, len(_OBS_SIZES)))]
            oy = int(rng.integers(0, CHUNK_SIZE))
            ox = int(rng.integers(0, max(1, CHUNK_SIZE - ow + 1)))

            if any((oy + dy, ox + dx) in corridor
                   for dy in range(oh) for dx in range(ow)):
                continue

            area = oh * ow
            if area <= 2:
                otype = 1           # tree
            elif area <= 4:
                otype = 1 if rng.random() < 0.5 else 2
            else:
                otype = 2           # container
            ocolor = int(rng.integers(0, 5))

            for dy in range(oh):
                for dx in range(ow):
                    yy, xx = oy + dy, ox + dx
                    if 0 <= yy < CHUNK_SIZE and 0 <= xx < CHUNK_SIZE:
                        world[yy, xx] = -1.0
                        obs_type[yy, xx] = otype
                        obs_color[yy, xx] = ocolor

        self._chunks[(cy, cx)] = {
            "world": world,
            "obs_type": obs_type,
            "obs_color": obs_color,
        }

    # ================================================================
    #  Internals
    # ================================================================

    def _collides(self, y, x):
        for dr in range(CAR_SIZE):
            for dc in range(CAR_SIZE):
                if self.world_val(y + dr, x + dc) < -0.5:
                    return True
        return False

    def _clear_area(self, y, x, h, w):
        """Force a rectangle to be empty (used for spawn)."""
        for dy in range(h):
            for dx in range(w):
                wy, wx = y + dy, x + dx
                cy, cx = wy // CHUNK_SIZE, wx // CHUNK_SIZE
                chunk = self._ensure_chunk(cy, cx)
                ly, lx = wy % CHUNK_SIZE, wx % CHUNK_SIZE
                chunk["world"][ly, lx] = 0.0
                chunk["obs_type"][ly, lx] = 0
                chunk["obs_color"][ly, lx] = 0


# ================================================================
#  Policies
# ================================================================

def expert_action(game):
    """Steer toward the clearest direction by checking danger in the FOV."""
    grid = game.get_grid()
    car_r = FOV_H // 2 - CAR_SIZE // 2
    car_c = FOV_W // 2 - CAR_SIZE // 2

    best_ax, best_ay = 0.0, -1.0        # default: go north
    best_score = float("inf")

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            tr, tc = car_r + dy, car_c + dx
            if tr < 0 or tr + CAR_SIZE > FOV_H or tc < 0 or tc + CAR_SIZE > FOV_W:
                continue

            collision = any(
                grid[tr + dr, tc + dc] < -0.5
                for dr in range(CAR_SIZE) for dc in range(CAR_SIZE)
            )
            if collision:
                continue

            danger = 0.0
            for r in range(max(0, tr - 4), min(FOV_H, tr + CAR_SIZE + 4)):
                for c in range(max(0, tc - 4), min(FOV_W, tc + CAR_SIZE + 4)):
                    if grid[r, c] < -0.5:
                        dist = max(1, abs(r - tr) + abs(c - tc))
                        danger += 3.0 / dist

            danger += dy * 2.0
            danger += abs(dx) * 0.1

            if danger < best_score:
                best_score = danger
                best_ax = float(dx)
                best_ay = float(dy)

    ax = -1.0 if best_ax < 0 else (1.0 if best_ax > 0 else 0.0)
    ay = -1.0 if best_ay < 0 else (1.0 if best_ay > 0 else 0.0)
    return ax, ay


def random_action(game):
    ax = float(np.random.choice([-1.0, 0.0, 1.0]))
    ay = float(np.random.choice([-1.0, 0.0, 1.0]))
    return ax, ay


def noisy_expert_action(game):
    if np.random.random() < 0.3:
        return random_action(game)
    return expert_action(game)


# ================================================================
#  Navigator — limited-visibility A* (fog of war)
# ================================================================

def _dir_to_action(dx, dy):
    ax = -1.0 if dx < 0 else (1.0 if dx > 0 else 0.0)
    ay = -1.0 if dy < 0 else (1.0 if dy > 0 else 0.0)
    return ax, ay


class DFSNavigator:
    """
    Limited-visibility navigator.

    The car only knows:
      • its own coordinates
      • the goal coordinates
      • cells within VISIBILITY of its outline (revealed as it moves)

    Each step the car reveals its local area, then plans a path with
    A* over what it has seen so far.  Unknown cells are assumed clear.
    When new obstacles are discovered on the planned path the car
    replans automatically.  It never has global map knowledge.
    """

    def __init__(self, goal_y=-50, goal_x=0):
        self.goal_y = goal_y
        self.goal_x = goal_x
        self._known = {}          # (y,x) → bool  (True = obstacle)
        self._plan = []           # solution path (excl. start)
        self._step = 0
        self._explored = set()    # last A* closed set
        self._trail = []          # positions the car has been
        self.reached = False

    # ── lifecycle ──────────────────────────────────────────────

    def reset(self, start_y=0, start_x=0, goal_y=None, goal_x=None):
        if goal_y is not None:
            self.goal_y = goal_y
        if goal_x is not None:
            self.goal_x = goal_x
        self._known = {}
        self._plan = []
        self._step = 0
        self._explored = set()
        self._trail = []
        self.reached = False

    # ── main step ─────────────────────────────────────────────

    def next_action(self, game):
        cur = (game.car_y, game.car_x)

        # record where the car has been
        if not self._trail or self._trail[-1] != cur:
            self._trail.append(cur)

        if abs(cur[0] - self.goal_y) <= 1 and abs(cur[1] - self.goal_x) <= 1:
            self.reached = True
            return 0.0, 0.0

        # 1. reveal local area (the only world access)
        self._reveal(game)

        # 2. replan when needed
        if self._needs_replan(cur):
            self._plan = self._astar(cur)
            self._step = 0

        if not self._plan or self._step >= len(self._plan):
            return 0.0, 0.0

        # 3. skip past current position
        while (self._step < len(self._plan)
               and self._plan[self._step] == cur):
            self._step += 1
        if self._step >= len(self._plan):
            return 0.0, 0.0

        # 4. move one step along the plan
        nxt = self._plan[self._step]
        return _dir_to_action(nxt[1] - cur[1], nxt[0] - cur[0])

    # ── perception ────────────────────────────────────────────

    def _reveal(self, game):
        """Reveal cells within VISIBILITY of the car outline."""
        cy, cx = game.car_y, game.car_x
        for dy in range(-VISIBILITY, CAR_SIZE + VISIBILITY):
            for dx in range(-VISIBILITY, CAR_SIZE + VISIBILITY):
                cell = (cy + dy, cx + dx)
                if cell not in self._known:
                    self._known[cell] = (
                        game.world_val(cell[0], cell[1]) < -0.5
                    )

    # ── collision using known map only ────────────────────────

    def _is_blocked(self, y, x):
        """Would the 2×2 car collide with a *known* obstacle here?
        Unknown cells are assumed clear (optimistic)."""
        for dr in range(CAR_SIZE):
            for dc in range(CAR_SIZE):
                if self._known.get((y + dr, x + dc), False):
                    return True
        return False

    # ── A* on the known map ───────────────────────────────────

    def _astar(self, start):
        goal = (self.goal_y, self.goal_x)
        open_heap = []
        counter = 0
        heapq.heappush(open_heap, (self._h(start, goal), counter, start))
        came_from = {}
        g = {start: 0.0}
        closed = set()

        while open_heap:
            _, _, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            closed.add(cur)

            cy, cx = cur
            if abs(cy - goal[0]) <= 1 and abs(cx - goal[1]) <= 1:
                self._explored = closed
                return self._reconstruct(came_from, cur)[1:]

            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    nb = (cy + dy, cx + dx)
                    if nb in closed:
                        continue
                    if self._is_blocked(nb[0], nb[1]):
                        closed.add(nb)
                        continue
                    cost = 1.414 if (dy != 0 and dx != 0) else 1.0
                    ng = g[cur] + cost
                    if ng < g.get(nb, float('inf')):
                        g[nb] = ng
                        came_from[nb] = cur
                        counter += 1
                        heapq.heappush(open_heap, (
                            ng + self._h(nb, goal), counter, nb))

            if len(closed) > 50000:
                break

        self._explored = closed
        return []

    # ── helpers ────────────────────────────────────────────────

    def _needs_replan(self, cur):
        if not self._plan or self._step >= len(self._plan):
            return True
        nxt = self._plan[self._step]
        if abs(nxt[0] - cur[0]) > 1 or abs(nxt[1] - cur[1]) > 1:
            return True                       # off-track
        # any newly-discovered obstacle on the remaining path?
        for i in range(self._step, len(self._plan)):
            if self._is_blocked(*self._plan[i]):
                return True
        return False

    @staticmethod
    def _reconstruct(came_from, end):
        path = [end]
        while end in came_from:
            end = came_from[end]
            path.append(end)
        path.reverse()
        return path

    @staticmethod
    def _h(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    # ── read-only state ───────────────────────────────────────

    @property
    def is_backtracking(self):
        return False

    @property
    def path(self):
        """Cells the car has actually travelled through."""
        return list(self._trail)

    @property
    def visited(self):
        """Cells the car has actually seen (revealed clear cells)."""
        return {c for c, obs in self._known.items() if not obs}


# ================================================================
#  Quick visual test
# ================================================================

if __name__ == "__main__":
    import pygame
    from renderer import GameRenderer

    pygame.init()
    screen = pygame.display.set_mode((DISPLAY_W, DISPLAY_H))
    pygame.display.set_caption("Self-Driving — expert (infinite world)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 20, bold=True)

    renderer = GameRenderer()
    game = CarGame(seed=42)
    game.reset()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                running = False

        ax, ay = expert_action(game)
        grid, done, info = game.step(ax, ay)

        if done:
            print(f"Crash! Score: {info['score']}  Chunks loaded: {len(game._chunks)}")
            game.reset()

        renderer.draw(screen, game)
        txt = font.render(
            f"Score: {info['score']}  Steps: {game.steps}  Chunks: {len(game._chunks)}",
            True, (255, 255, 0),
        )
        bar = pygame.Surface((txt.get_width() + 12, 26), pygame.SRCALPHA)
        bar.fill((0, 0, 0, 130))
        screen.blit(bar, (4, 4))
        screen.blit(txt, (10, 6))
        pygame.display.flip()
        clock.tick(10)

    pygame.quit()
