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
    offset: torch.Tensor


class InstanceSegLoss(nn.Module):
    """Semantic CE + a CenterNet focal heatmap + an offset-to-center L1.

    The offset term (supervised only at foreground pixels) teaches each pixel to
    point at its instance's center, which is what the decoder uses to group
    pixels into instances — fixing the boundary/overlap pixels a raw
    nearest-center rule mis-assigns.
    """

    def __init__(self, sigma: float = 2.0, lambda_heatmap: float = 1.0,
                 lambda_offset: float = 0.1) -> None:
        super().__init__()
        self.seg = SegLoss()
        self.sigma = sigma
        self.lambda_heatmap = lambda_heatmap
        self.lambda_offset = lambda_offset

    def forward(self, out, class_map: torch.Tensor, instance_map: torch.Tensor,
                centers_px_per_image: list[torch.Tensor]) -> InstanceSegLossTerms:
        """``out`` is an InstanceSegOutput; ``class_map`` / ``instance_map`` are
        (B, H, W) long; ``centers_px_per_image`` is a per-image list of (N_b, 2)
        GT centers in image-pixel coords."""
        seg = self.seg(out.semantic_logits, class_map)
        _, _, hh, ww = out.heatmap.shape
        target_hm = multi_gaussian_target(centers_px_per_image, hh, ww, out.stride, self.sigma)
        hm = focal_heatmap_loss(out.heatmap, target_hm)
        offset = self._offset_loss(out, instance_map, centers_px_per_image, hh, ww)
        total = seg + self.lambda_heatmap * hm + self.lambda_offset * offset
        return InstanceSegLossTerms(total=total, semantic=seg, heatmap=hm, offset=offset)

    def _offset_loss(self, out, instance_map: torch.Tensor,
                     centers_px_per_image: list[torch.Tensor], hh: int, ww: int) -> torch.Tensor:
        """L1 between the predicted offset and (instance_center - pixel), in
        output cells, at foreground pixels only."""
        device = out.offset.device
        stride = out.stride
        # instance ids at output resolution.
        inst = F.interpolate(
            instance_map.float().unsqueeze(1), size=(hh, ww), mode="nearest",
        ).squeeze(1).long()  # (B, h, w)
        yy, xx = torch.meshgrid(
            torch.arange(hh, device=device, dtype=torch.float32),
            torch.arange(ww, device=device, dtype=torch.float32),
            indexing="ij",
        )
        cell_grid = torch.stack([xx, yy], dim=-1)  # (h, w, 2) cell coords
        total = torch.zeros((), device=device)
        n_valid = 0
        for b, centers in enumerate(centers_px_per_image):
            valid = inst[b] > 0
            n = int(valid.sum())
            if n == 0 or centers.numel() == 0:
                continue
            # center per instance id (in cell coords); lut[0] is unused background.
            lut = torch.zeros((centers.shape[0] + 1, 2), device=device)
            lut[1:] = centers.to(device) / stride
            per_cell_center = lut[inst[b]]                 # (h, w, 2)
            target = (per_cell_center - cell_grid).permute(2, 0, 1)  # (2, h, w)
            pred = out.offset[b]
            total = total + F.l1_loss(pred[:, valid], target[:, valid], reduction="sum")
            n_valid += n
        return total / max(n_valid, 1)
