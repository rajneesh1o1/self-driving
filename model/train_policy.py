"""
Train the imitation learning driving policy (classification).

Behavioural cloning: given FOV + goal_dir, predict which of 8 directions
the A* navigator would choose. Cross-entropy loss.

Usage:
    python train_policy.py --data ../game/driving_data.h5 --epochs 50
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from policy import DrivingPolicy, ACTION_TO_IDX


# ---------- dataset ----------

def _action_key(ax, ay):
    return (float(round(ax)), float(round(ay)))


def _augment(fovs, goal_dirs, actions):
    """4x augmentation: original + H-flip + V-flip + 180° rotation."""
    all_fov, all_gd, all_act = [fovs], [goal_dirs], [actions]

    # horizontal flip: mirror columns, negate ax and goal_dx
    f = fovs[:, :, ::-1].copy()
    g = goal_dirs.copy(); g[:, 1] *= -1      # negate dx
    a = actions.copy();   a[:, 0] *= -1      # negate ax
    all_fov.append(f); all_gd.append(g); all_act.append(a)

    # vertical flip: mirror rows, negate ay and goal_dy
    f = fovs[:, ::-1, :].copy()
    g = goal_dirs.copy(); g[:, 0] *= -1      # negate dy
    a = actions.copy();   a[:, 1] *= -1      # negate ay
    all_fov.append(f); all_gd.append(g); all_act.append(a)

    # 180° rotation: both flips
    f = fovs[:, ::-1, ::-1].copy()
    g = goal_dirs * -1
    a = actions * -1
    all_fov.append(f); all_gd.append(g); all_act.append(a)

    return (np.concatenate(all_fov), np.concatenate(all_gd),
            np.concatenate(all_act))


class PolicyDataset(Dataset):
    """Single-step (FOV, goal_dir) -> action_class with 4x augmentation."""

    def __init__(self, h5_path, augment=True):
        with h5py.File(h5_path, "r") as f:
            fovs = f["fov"][:]            # (N, 6, 6)
            actions = f["action"][:]      # (N, 2)
            goal_dirs = f["goal_dir"][:]  # (N, 2)

        print(f"Raw samples: {len(fovs)}")

        if augment:
            fovs, goal_dirs, actions = _augment(fovs, goal_dirs, actions)
            print(f"After 4x augmentation: {len(fovs)}")

        # convert (ax, ay) to class indices
        self.labels = np.zeros(len(actions), dtype=np.int64)
        valid = np.ones(len(actions), dtype=bool)
        for i, (ax, ay) in enumerate(actions):
            key = _action_key(ax, ay)
            if key in ACTION_TO_IDX:
                self.labels[i] = ACTION_TO_IDX[key]
            else:
                valid[i] = False

        if not valid.all():
            n_dropped = (~valid).sum()
            print(f"Dropped {n_dropped} invalid samples")
            fovs = fovs[valid]
            goal_dirs = goal_dirs[valid]
            self.labels = self.labels[valid]

        self.fovs = fovs
        self.goal_dirs = goal_dirs
        print(f"Final dataset: {len(self.fovs)} samples, 8 action classes")

        unique, counts = np.unique(self.labels, return_counts=True)
        for cls, cnt in zip(unique, counts):
            pct = cnt / len(self.labels) * 100
            print(f"  class {cls}: {cnt} ({pct:.1f}%)")

    def __len__(self):
        return len(self.fovs)

    def __getitem__(self, idx):
        return {
            "fov": torch.from_numpy(self.fovs[idx]).float(),
            "goal_dir": torch.from_numpy(self.goal_dirs[idx]).float(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------- training ----------

def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    dataset = PolicyDataset(args.data)

    n_train = int(0.9 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"Train: {n_train}  Val: {n_val}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
    )

    model = DrivingPolicy(hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Policy parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0

        for batch in train_loader:
            fov = batch["fov"].to(device)
            goal_dir = batch["goal_dir"].to(device)
            label = batch["label"].to(device)

            logits = model(fov, goal_dir)
            loss = F.cross_entropy(logits, label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == label).sum().item()
            total += label.size(0)
            n_batches += 1

        scheduler.step()
        train_loss = total_loss / max(n_batches, 1)
        train_acc = correct / max(total, 1) * 100

        # --- val ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_n = 0

        with torch.no_grad():
            for batch in val_loader:
                fov = batch["fov"].to(device)
                goal_dir = batch["goal_dir"].to(device)
                label = batch["label"].to(device)

                logits = model(fov, goal_dir)
                val_loss += F.cross_entropy(logits, label).item()
                val_correct += (logits.argmax(1) == label).sum().item()
                val_total += label.size(0)
                val_n += 1

        val_avg = val_loss / max(val_n, 1)
        val_acc = val_correct / max(val_total, 1) * 100
        lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss {train_loss:.4f} acc {train_acc:.1f}% | "
            f"val {val_avg:.4f} acc {val_acc:.1f}% | lr {lr:.2e}"
        )

        # --- save best by val accuracy ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_dir / "policy_best.pt")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), out_dir / f"policy_epoch_{epoch}.pt")
            print(f"  -> saved")

    print(f"\nDone! Best val accuracy: {best_val_acc:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train imitation learning policy")
    parser.add_argument("--data", type=str, default="../game/driving_data.h5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    args = parser.parse_args()
    train(args)
