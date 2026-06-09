"""Loss for semantic segmentation: pixel cross-entropy, optionally + Dice.

The target is a ``(B, H, W)`` long class map (background = ``num_classes``).
When the model runs at an output stride > 1 its logits are smaller than the
target; we downsample the target by nearest-neighbour so labels stay integral.
Cross-entropy alone solves this clean task; Dice is available (lambda_dice > 0)
for the imbalanced-foreground regime.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.heatmap_loss import focal_heatmap_loss, multi_gaussian_target


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


@dataclass
class InstanceSegLossTerms:
    total: torch.Tensor
    semantic: torch.Tensor
    heatmap: torch.Tensor


class InstanceSegLoss(nn.Module):
    """Semantic cross-entropy + a CenterNet focal loss on the center heatmap.

    Grouping (assigning pixels to centers) happens at decode time and needs no
    loss term; supervising "good semantics + well-localized centers" is enough
    to separate instances.
    """

    def __init__(self, sigma: float = 2.0, lambda_heatmap: float = 1.0) -> None:
        super().__init__()
        self.seg = SegLoss()
        self.sigma = sigma
        self.lambda_heatmap = lambda_heatmap

    def forward(self, out, class_map: torch.Tensor,
                centers_px_per_image: list[torch.Tensor]) -> InstanceSegLossTerms:
        """``out`` is an InstanceSegOutput; ``class_map`` is (B, H, W) long;
        ``centers_px_per_image`` is a per-image list of (N_b, 2) GT centers in
        image-pixel coords."""
        seg = self.seg(out.semantic_logits, class_map)
        _, _, hh, ww = out.heatmap.shape
        target_hm = multi_gaussian_target(centers_px_per_image, hh, ww, out.stride, self.sigma)
        hm = focal_heatmap_loss(out.heatmap, target_hm)
        return InstanceSegLossTerms(total=seg + self.lambda_heatmap * hm, semantic=seg, heatmap=hm)
