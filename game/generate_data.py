"""
Generate self-driving training data as HDF5.

Records expert, noisy-expert and random driving episodes.
Now with 2D actions: (steer_x, steer_y).

Usage:
    python generate_data.py [--episodes 500] [--output driving_data]
"""

import argparse
import numpy as np
import h5py
from pathlib import Path

from car_game import (
    CarGame,
    expert_action,
    noisy_expert_action,
    random_action,
    GRID_H,
    GRID_W,
)


def generate_episode(game, policy_fn, max_steps=300):
    """Run one episode, return grids and actions."""
    grids = []
    actions = []

    grid = game.reset()
    grids.append(grid)
    actions.append([0.0, 0.0])  # no action on first frame

    step = 0
    while not game.done and step < max_steps:
        ax, ay = policy_fn(game)
        grid, done, info = game.step(ax, ay)
        grids.append(grid)
        actions.append([ax, ay])
        step += 1

    return {
        "grid": np.array(grids, dtype=np.float32),          # (L, H, W)
        "action": np.array(actions, dtype=np.float32),       # (L, 2)
        "score": info["score"],
        "length": len(grids),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate driving training data")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--output", type=str, default="driving_data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    game = CarGame(seed=args.seed)

    policies = {
        "expert": expert_action,
        "noisy":  noisy_expert_action,
        "random": random_action,
    }
    policy_schedule = (
        ["expert"] * int(args.episodes * 0.4)
        + ["noisy"] * int(args.episodes * 0.4)
        + ["random"] * (args.episodes - int(args.episodes * 0.4) * 2)
    )
    np.random.shuffle(policy_schedule)

    episodes = []
    scores = []

    for i, policy_name in enumerate(policy_schedule):
        ep = generate_episode(game, policies[policy_name])
        episodes.append(ep)
        scores.append(ep["score"])

        if (i + 1) % 100 == 0:
            avg = np.mean(scores[-100:])
            print(f"  [{i+1}/{args.episodes}] last-100 avg score: {avg:.1f}")

    print(f"\nTotal episodes: {len(episodes)}")
    print(f"Average score : {np.mean(scores):.1f}")
    print(f"Max score     : {np.max(scores)}")

    # ---- write HDF5 ----
    out_path = Path(args.output + ".h5")
    total_frames = sum(ep["length"] for ep in episodes)
    action_dim = 2

    print(f"Writing {total_frames} frames across {len(episodes)} episodes to {out_path}")

    with h5py.File(out_path, "w") as f:
        f.create_dataset(
            "grid",
            shape=(total_frames, GRID_H, GRID_W),
            dtype=np.float32,
            chunks=(min(512, total_frames), GRID_H, GRID_W),
            compression="gzip",
        )
        f.create_dataset(
            "action",
            shape=(total_frames, action_dim),
            dtype=np.float32,
            chunks=(min(2048, total_frames), action_dim),
            compression="gzip",
        )
        f.create_dataset("episode_idx", shape=(total_frames,), dtype=np.int64)
        f.create_dataset("step_idx", shape=(total_frames,), dtype=np.int64)

        ep_lens = []
        ep_offsets = []
        offset = 0

        for ep_i, ep in enumerate(episodes):
            n = ep["length"]
            f["grid"][offset : offset + n] = ep["grid"]
            f["action"][offset : offset + n] = ep["action"]
            f["episode_idx"][offset : offset + n] = ep_i
            f["step_idx"][offset : offset + n] = np.arange(n)

            ep_lens.append(n)
            ep_offsets.append(offset)
            offset += n

        f.create_dataset("ep_len", data=np.array(ep_lens, dtype=np.int64))
        f.create_dataset("ep_offset", data=np.array(ep_offsets, dtype=np.int64))

    print("Done!")


if __name__ == "__main__":
    main()
