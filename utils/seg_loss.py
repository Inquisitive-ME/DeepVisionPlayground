"""Loss for semantic segmentation: pixel cross-entropy, optionally + Dice.

The target is a ``(B, H, W)`` long class map (background = ``num_classes``).
When the model runs at an output stride > 1 its logits are smaller than the
target; we downsample the target by nearest-neighbour so labels stay integral.
Cross-entropy alone solves this clean task; Dice is available (lambda_dice > 0)
for the imbalanced-foreground regime.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegLoss(nn.Module):
    def __init__(self, lambda_dice: float = 0.0) -> None:
        super().__init__()
        self.lambda_dice = lambda_dice

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """logits: (B, C, h, w) raw; target: (B, H, W) long. C includes background."""
        if target.shape[-2:] != logits.shape[-2:]:
            target = F.interpolate(
                target.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest",
            ).squeeze(1).long()
        loss = F.cross_entropy(logits, target)
        if self.lambda_dice > 0:
            num_classes = logits.shape[1]
            probs = logits.softmax(dim=1)
            onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
            dims = (0, 2, 3)
            inter = (probs * onehot).sum(dims)
            union = probs.sum(dims) + onehot.sum(dims)
            dice = 1.0 - ((2 * inter + 1.0) / (union + 1.0)).mean()
            loss = loss + self.lambda_dice * dice
        return loss
