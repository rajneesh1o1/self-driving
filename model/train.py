"""
Standalone training script for the self-driving JEPA world model.

No dependency on stable_worldmodel, stable_pretraining, hydra, or lightning.
Pure PyTorch.

Usage:
    python train.py --data ../game/driving_data.h5 --epochs 20
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from module import SIGReg, ARPredictor, Embedder, MLP
from jepa import GridEncoder, JEPA


# ---------- dataset ----------

class DrivingDataset(Dataset):
    """Loads sequences of (fov, action, goal_dir) from HDF5."""

    def __init__(self, h5_path, seq_len):
        with h5py.File(h5_path, "r") as f:
            self.fovs = f["fov"][:]          # (N, 6, 6) float32
            self.actions = f["action"][:]    # (N, 2)    float32
            self.goal_dirs = f["goal_dir"][:] # (N, 2)  float32
            ep_len = f["ep_len"][:]
            ep_offset = f["ep_offset"][:]

        # valid start indices for sequences that don't cross episode boundaries
        self.indices = []
        for offset, length in zip(ep_offset, ep_len):
            for j in range(int(length) - seq_len + 1):
                self.indices.append(int(offset) + j)

        self.seq_len = seq_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.indices[idx]
        e = s + self.seq_len
        # concat action (2D) + goal_dir (2D) → 4D conditioning vector
        cond = np.concatenate([self.actions[s:e], self.goal_dirs[s:e]], axis=-1)
        return {
            "grid": torch.from_numpy(self.fovs[s:e]).float(),
            "action": torch.from_numpy(cond).float(),
        }


# ---------- training ----------

def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    seq_len = args.history_size + args.num_preds
    dataset = DrivingDataset(args.data, seq_len)
    print(f"Dataset: {len(dataset)} sequences  (seq_len={seq_len})")

    n_train = int(0.9 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # --- build model ---
    encoder = GridEncoder(grid_h=args.grid_h, grid_w=args.grid_w, embed_dim=args.embed_dim)

    predictor = ARPredictor(
        num_frames=args.history_size,
        input_dim=args.embed_dim,
        hidden_dim=args.embed_dim,
        output_dim=args.embed_dim,
        depth=args.pred_depth,
        heads=args.pred_heads,
        mlp_dim=args.pred_mlp_dim,
        dim_head=32,
        dropout=0.1,
    )

    action_encoder = Embedder(input_dim=args.action_dim, emb_dim=args.embed_dim)

    projector = MLP(
        input_dim=args.embed_dim,
        hidden_dim=args.embed_dim * 2,
        output_dim=args.embed_dim,
        norm_fn=torch.nn.BatchNorm1d,
    )

    pred_proj = MLP(
        input_dim=args.embed_dim,
        hidden_dim=args.embed_dim * 2,
        output_dim=args.embed_dim,
        norm_fn=torch.nn.BatchNorm1d,
    )

    model = JEPA(encoder, predictor, action_encoder, projector, pred_proj).to(device)
    sigreg = SIGReg().to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- training loop ---
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_pred, total_sig = 0.0, 0.0, 0.0
        n_batches = 0

        for batch in train_loader:
            grid = batch["grid"].to(device)      # (B, T, H, W)
            action = batch["action"].to(device)  # (B, T, 1)

            # replace NaN at boundaries
            action = torch.nan_to_num(action, 0.0)

            info = model.encode({"grid": grid, "action": action})
            emb = info["emb"]           # (B, T, D)
            act_emb = info["act_emb"]   # (B, T, D)

            ctx_emb = emb[:, : args.history_size]
            ctx_act = act_emb[:, : args.history_size]
            tgt_emb = emb[:, args.num_preds :]

            pred_emb = model.predict(ctx_emb, ctx_act)

            pred_loss = (pred_emb - tgt_emb.detach()).pow(2).mean()
            sigreg_loss = sigreg(emb.transpose(0, 1))
            loss = pred_loss + args.sigreg_weight * sigreg_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_pred += pred_loss.item()
            total_sig += sigreg_loss.item()
            n_batches += 1

        scheduler.step()

        avg = total_loss / max(n_batches, 1)
        avg_p = total_pred / max(n_batches, 1)
        avg_s = total_sig / max(n_batches, 1)
        lr = scheduler.get_last_lr()[0]

        # --- validation ---
        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                grid = batch["grid"].to(device)
                action = torch.nan_to_num(batch["action"].to(device), 0.0)
                info = model.encode({"grid": grid, "action": action})
                emb = info["emb"]
                act_emb = info["act_emb"]
                ctx_emb = emb[:, : args.history_size]
                ctx_act = act_emb[:, : args.history_size]
                tgt_emb = emb[:, args.num_preds :]
                pred_emb = model.predict(ctx_emb, ctx_act)
                val_loss += (pred_emb - tgt_emb).pow(2).mean().item()
                val_n += 1

        val_avg = val_loss / max(val_n, 1)
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss {avg:.4f} (pred {avg_p:.4f} sig {avg_s:.4f}) | "
            f"val {val_avg:.4f} | lr {lr:.2e}"
        )

        # --- save ---
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = out_dir / f"driving_jepa_epoch_{epoch}.pt"
            torch.save(model, ckpt_path)
            print(f"  -> saved {ckpt_path}")

    print("Training complete!")


# ---------- CLI ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train self-driving JEPA world model")
    parser.add_argument("--data", type=str, default="../game/driving_data.h5")
    parser.add_argument("--seed", type=int, default=42)

    # grid dimensions (must match game: LOCAL_FOV = 6)
    parser.add_argument("--grid_h", type=int, default=6)
    parser.add_argument("--grid_w", type=int, default=6)
    parser.add_argument("--action_dim", type=int, default=4)  # action(2) + goal_dir(2)

    # model
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--pred_depth", type=int, default=2)
    parser.add_argument("--pred_heads", type=int, default=4)
    parser.add_argument("--pred_mlp_dim", type=int, default=256)

    # world model
    parser.add_argument("--history_size", type=int, default=1)
    parser.add_argument("--num_preds", type=int, default=3)

    # training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sigreg_weight", type=float, default=1.0)

    # output
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")

    args = parser.parse_args()
    train(args)
