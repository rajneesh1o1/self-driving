"""
JEPA (Joint Embedding Predictive Architecture) for grid-based self-driving.

Replaces the ViT encoder from le-wm with a lightweight CNN encoder
since our observations are tiny grids (6 x 6 local FOV, single channel).
Action conditioning includes action (2D) + goal_dir (2D) = 4D.
"""

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange


class GridEncoder(nn.Module):
    """Small CNN that encodes a 1-channel grid into a flat embedding."""

    def __init__(self, grid_h=16, grid_w=16, embed_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),  # -> (64, 3, 3) = 576
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 3 * 3, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        """x: (N, 1, H, W) -> (N, embed_dim)"""
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


def _detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


class JEPA(nn.Module):
    """Grid-based JEPA world model."""

    def __init__(self, encoder, predictor, action_encoder, projector=None, pred_proj=None):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        """Encode grid observations and actions into embeddings.

        info['grid']:   (B, T, H, W) float
        info['action']: (B, T, 1)     float   (optional)
        """
        grid = info["grid"].float()
        b, t = grid.shape[:2]
        grid_flat = grid.view(b * t, 1, grid.shape[2], grid.shape[3])
        raw_emb = self.encoder(grid_flat)          # (B*T, D)
        emb = self.projector(raw_emb)              # (B*T, D)
        info["emb"] = emb.view(b, t, -1)           # (B, T, D)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next-state embeddings.

        emb:     (B, T, D)
        act_emb: (B, T, D)
        Returns: (B, T, D)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    # ---------- inference helpers ----------

    def rollout(self, init_emb, action_sequence, history_size=1):
        """Autoregressive rollout for planning.

        init_emb:        (B, 1, D)  — encoded current state
        action_sequence: (B, S, T, 1)  — S candidate action sequences of length T
        history_size:    int

        Returns predicted embeddings (B, S, T+1, D).
        """
        B, S, T, _ = action_sequence.shape
        HS = history_size

        # expand init for all candidates
        emb = init_emb.unsqueeze(1).expand(B, S, -1, -1)        # (B,S,1,D)
        emb = rearrange(emb, "b s t d -> (b s) t d").clone()    # (BS,1,D)
        acts = rearrange(action_sequence, "b s t a -> (b s) t a")  # (BS,T,1)

        for t in range(T):
            act_so_far = acts[:, : t + 1]
            act_emb = self.action_encoder(act_so_far)            # (BS, t+1, D)
            emb_trunc = emb[:, -HS:]
            act_trunc = act_emb[:, -HS:]
            pred = self.predict(emb_trunc, act_trunc)[:, -1:]    # (BS,1,D)
            emb = torch.cat([emb, pred], dim=1)

        return rearrange(emb, "(b s) t d -> b s t d", b=B, s=S)

    def plan_cost(self, init_emb, action_sequence, history_size=1):
        """Cost = L2 distance between final predicted embedding and initial."""
        pred_emb = self.rollout(init_emb, action_sequence, history_size)
        final = pred_emb[:, :, -1, :]   # (B, S, D)
        init = init_emb[:, 0, :]         # (B, D)
        cost = (final - init.unsqueeze(1)).pow(2).sum(dim=-1)  # (B, S)
        return cost
