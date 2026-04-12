"""
Generate self-driving training data as HDF5.

Runs the car in auto-pilot (A* navigator) and captures the 6×6
local field-of-view at every step.

Goals are placed in ALL directions (N/S/E/W/diagonals) at varying
distances so the action distribution is balanced.

Usage:
    python generate_data.py                     # 500 episodes
    python generate_data.py --episodes 2000
"""

import argparse
import math
import numpy as np
import h5py
from pathlib import Path

from car_game import (
    CarGame,
    DFSNavigator,
    LOCAL_FOV,
    VISIBILITY,
    CAR_SIZE,
)

# Goal placement: random direction, random distance
GOAL_DIST_MIN = 15
GOAL_DIST_MAX = 40


def _random_goal(game, rng):
    """Pick a goal in a random direction at a random distance."""
    angle = rng.uniform(0, 2 * math.pi)
    dist = rng.integers(GOAL_DIST_MIN, GOAL_DIST_MAX + 1)
    gy = game.car_y + int(round(dist * math.sin(angle)))
    gx = game.car_x + int(round(dist * math.cos(angle)))
    return gy, gx


def _goal_dir(game, gy, gx):
    """Normalised (dy, dx) from car to goal."""
    dy = gy - game.car_y
    dx = gx - game.car_x
    d = math.sqrt(dy * dy + dx * dx)
    if d > 0:
        return [dy / d, dx / d]
    return [0.0, 0.0]


def generate_episode(game, rng, max_steps=300):
    """One episode: navigate toward several random-direction goals."""
    game.reset(seed=int(rng.integers(0, 2**31)))

    gy, gx = _random_goal(game, rng)
    nav = DFSNavigator(goal_y=gy, goal_x=gx)
    nav.reset(start_y=game.car_y, start_x=game.car_x)

    fovs = []
    actions = []
    goal_dirs = []

    prev_fov = game.get_local_fov()

    for _ in range(max_steps):
        ax, ay = nav.next_action(game)

        # skip frames where the car doesn't move
        if ax == 0.0 and ay == 0.0:
            continue

        _, done, info = game.step(ax, ay)
        cur_fov = game.get_local_fov()

        # skip duplicate frames (identical view)
        if len(fovs) > 0 and np.array_equal(cur_fov, prev_fov):
            if done:
                break
            continue

        fovs.append(prev_fov)          # frame before the move
        actions.append([ax, ay])        # the move taken
        goal_dirs.append(_goal_dir(game, gy, gx))
        prev_fov = cur_fov

        if done:
            break

        if nav.reached:
            gy, gx = _random_goal(game, rng)
            nav.reset(
                start_y=game.car_y, start_x=game.car_x,
                goal_y=gy, goal_x=gx,
            )

    # append final frame so we have (frame_t, action, frame_t+1) pairs
    if fovs:
        fovs.append(prev_fov)
        actions.append(actions[-1])     # repeat last real action
        goal_dirs.append(_goal_dir(game, gy, gx))

    return {
        "fov": np.array(fovs, dtype=np.float32) if fovs else np.zeros((0, LOCAL_FOV, LOCAL_FOV), dtype=np.float32),
        "action": np.array(actions, dtype=np.float32) if actions else np.zeros((0, 2), dtype=np.float32),
        "goal_dir": np.array(goal_dirs, dtype=np.float32) if goal_dirs else np.zeros((0, 2), dtype=np.float32),
        "score": info.get("score", 0),
        "length": len(fovs),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate driving training data")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=str, default="driving_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--append", action="store_true",
                        help="Append to existing HDF5 instead of overwriting")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    game = CarGame(seed=args.seed)

    print(f"Local FOV: {LOCAL_FOV}x{LOCAL_FOV}  (VISIBILITY={VISIBILITY})")
    print(f"Goals: random direction, dist {GOAL_DIST_MIN}-{GOAL_DIST_MAX}")
    print(f"Generating {args.episodes} episodes …")

    episodes = []
    scores = []

    for i in range(args.episodes):
        ep = generate_episode(game, rng)
        episodes.append(ep)
        scores.append(ep["score"])

        if (i + 1) % 10 == 0:
            avg = np.mean(scores[-10:])
            print(f"  [{i+1}/{args.episodes}] last-10 avg score: {avg:.1f}")

    print(f"\nNew episodes: {len(episodes)}")
    print(f"Average score: {np.mean(scores):.1f}")

    # ---- load existing data if appending ----
    out_path = Path(args.output + ".h5")
    old_episodes = []

    if args.append and out_path.exists():
        with h5py.File(out_path, "r") as f:
            old_fov = f["fov"][:]
            old_action = f["action"][:]
            old_goal_dir = f["goal_dir"][:]
            old_ep_len = f["ep_len"][:]
            old_ep_offset = f["ep_offset"][:]

        # reconstruct old episodes
        for i, (off, ln) in enumerate(zip(old_ep_offset, old_ep_len)):
            s, e = int(off), int(off + ln)
            old_episodes.append({
                "fov": old_fov[s:e],
                "action": old_action[s:e],
                "goal_dir": old_goal_dir[s:e],
                "length": int(ln),
            })
        print(f"Loaded {len(old_episodes)} existing episodes "
              f"({sum(e['length'] for e in old_episodes)} frames)")

    all_episodes = old_episodes + episodes
    total_frames = sum(ep["length"] for ep in all_episodes)

    print(f"Writing {total_frames} total frames "
          f"({len(all_episodes)} episodes) to {out_path}")

    with h5py.File(out_path, "w") as f:
        f.create_dataset(
            "fov",
            shape=(total_frames, LOCAL_FOV, LOCAL_FOV),
            dtype=np.float32,
            chunks=(min(512, total_frames), LOCAL_FOV, LOCAL_FOV),
            compression="gzip",
        )
        f.create_dataset(
            "action",
            shape=(total_frames, 2),
            dtype=np.float32,
            chunks=(min(2048, total_frames), 2),
            compression="gzip",
        )
        f.create_dataset(
            "goal_dir",
            shape=(total_frames, 2),
            dtype=np.float32,
            chunks=(min(2048, total_frames), 2),
            compression="gzip",
        )
        f.create_dataset("episode_idx", shape=(total_frames,), dtype=np.int64)
        f.create_dataset("step_idx", shape=(total_frames,), dtype=np.int64)

        ep_lens = []
        ep_offsets = []
        offset = 0

        for ep_i, ep in enumerate(all_episodes):
            n = ep["length"]
            f["fov"][offset: offset + n] = ep["fov"]
            f["action"][offset: offset + n] = ep["action"]
            f["goal_dir"][offset: offset + n] = ep["goal_dir"]
            f["episode_idx"][offset: offset + n] = ep_i
            f["step_idx"][offset: offset + n] = np.arange(n)

            ep_lens.append(n)
            ep_offsets.append(offset)
            offset += n

        f.create_dataset("ep_len", data=np.array(ep_lens, dtype=np.int64))
        f.create_dataset("ep_offset", data=np.array(ep_offsets, dtype=np.int64))

        f.attrs["local_fov"] = LOCAL_FOV
        f.attrs["visibility"] = VISIBILITY
        f.attrs["car_size"] = CAR_SIZE

    print("Done!")


if __name__ == "__main__":
    main()
