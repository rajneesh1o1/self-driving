"""
Imitation learning policy — learns to copy the A* navigator's decisions.

Input:  6x6 FOV grid + 2D goal direction
Output: one of 8 discrete actions (classification)

The 8 actions map to directions:
  0: (-1,-1)  1: (-1,0)  2: (-1,1)
  3: (0,-1)              4: (0,1)
  5: (1,-1)   6: (1,0)   7: (1,1)
"""

import torch
import torch.nn as nn

# 8 discrete actions (no STAY)
ACTION_TABLE = [
    (-1.0, -1.0), (-1.0, 0.0), (-1.0, 1.0),
    (0.0, -1.0),               (0.0, 1.0),
    (1.0, -1.0),  (1.0, 0.0),  (1.0, 1.0),
]

# reverse lookup: (ax, ay) -> class index
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_TABLE)}


class DrivingPolicy(nn.Module):
    """Behavioural cloning policy: (FOV, goal_dir) -> action class."""

    def __init__(self, fov_size=6, goal_dim=2, num_actions=8, hidden=128):
        super().__init__()

        # CNN branch for spatial FOV understanding
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),  # 6x6 -> 2x2
        )
        cnn_out = 64 * 2 * 2  # 256

        # Fuse CNN features + goal_dir -> action logits
        self.head = nn.Sequential(
            nn.Linear(cnn_out + goal_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_actions),  # raw logits
        )

    def forward(self, fov, goal_dir):
        """
        fov:      (B, 6, 6) float — local field of view
        goal_dir: (B, 2)    float — normalised direction to goal
        Returns:  (B, 8)    float — logits for each action class
        """
        x = fov.unsqueeze(1)           # (B, 1, 6, 6)
        x = self.conv(x)               # (B, 64, 2, 2)
        x = x.flatten(1)               # (B, 256)
        x = torch.cat([x, goal_dir], dim=1)  # (B, 258)
        return self.head(x)

    def predict_action(self, fov, goal_dir):
        """Return the (ax, ay) tuple for the highest-scoring action."""
        logits = self.forward(fov, goal_dir)
        idx = logits.argmax(dim=1).item()
        return ACTION_TABLE[idx]
