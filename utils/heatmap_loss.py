"""Loss for the CenterNet-style heatmap predictor.

Given GT centers in image-pixel space and class ids, build a (B, 1, H, W)
target heatmap (Gaussian peak at each GT cell), a (B, 2) target sub-pixel
offset, and a (B,) target class. Combine three terms:

- Heatmap: pixel-wise BCE-with-logits against the Gaussian target. Plain
  BCE is enough here because the foreground/background imbalance is mild
  (one image, one shape); we don't need the full CenterNet focal loss.
- Offset: L1 between predicted offset at the GT cell and the true sub-cell
  offset. Only the GT cell contributes, since the rest of the heatmap has
  no canonical offset.
- Class: cross-entropy between the per-pixel class logits at the GT cell
  and the true class.

This file does NOT implement the full CenterNet multi-object pipeline.
Single object only — multi-object would need NMS on the heatmap and a
matching step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from models.center_heatmap_net import HeatmapOutput


@dataclass
class HeatmapLossTerms:
    total: torch.Tensor
    heatmap: torch.Tensor
    offset: torch.Tensor
    class_: torch.Tensor


def gaussian_target(
    centers_px: torch.Tensor,
    out_h: int,
    out_w: int,
    stride: int,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Per-image Gaussian heatmap target.

    Peak is placed at the rounded heatmap cell so it's exactly 1.0 there
    (the focal loss splits on "y == 1" vs "y < 1", so we need an exact 1).
    """
    B = centers_px.shape[0]
    device = centers_px.device
    # Round GT to nearest cell so the peak hits exactly 1.0.
    gx = (centers_px[:, 0] / stride).round()
    gy = (centers_px[:, 1] / stride).round()

    yy = torch.arange(out_h, device=device, dtype=torch.float32).view(1, out_h, 1)
    xx = torch.arange(out_w, device=device, dtype=torch.float32).view(1, 1, out_w)
    gx_b = gx.view(B, 1, 1)
    gy_b = gy.view(B, 1, 1)
    sq = (xx - gx_b) ** 2 + (yy - gy_b) ** 2
    target = torch.exp(-sq / (2 * sigma ** 2))
    return target.unsqueeze(1)  # (B, 1, H, W)


def focal_heatmap_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """Penalty-reduced focal loss from CenterNet (Zhou 2019, eq 1).

    For pixels where the Gaussian target peaks (target == 1):
        L = -(1 - p)^alpha * log(p)
    For pixels where target < 1 (Gaussian shoulders + far background):
        L = -(1 - target)^beta * p^alpha * log(1 - p)

    Normalized by the number of peak pixels (one per image in our setup).
    """
    pred = torch.sigmoid(pred_logits).clamp(1e-6, 1 - 1e-6)
    is_peak = target.eq(1.0).float()
    pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * is_peak
    neg_loss = -((1 - target) ** beta) * (pred ** alpha) * torch.log(1 - pred) * (1 - is_peak)
    n_peaks = is_peak.sum().clamp_min(1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n_peaks


class HeatmapLoss(nn.Module):
    def __init__(
        self,
        lambda_heatmap: float = 1.0,
        lambda_offset: float = 1.0,
        lambda_class: float = 1.0,
        sigma: float = 1.5,
    ) -> None:
        super().__init__()
        self.lambda_heatmap = lambda_heatmap
        self.lambda_offset = lambda_offset
        self.lambda_class = lambda_class
        self.sigma = sigma

    def forward(
        self,
        out: HeatmapOutput,
        gt_centers_px: torch.Tensor,
        gt_classes: torch.Tensor,
    ) -> HeatmapLossTerms:
        """All three terms.

        Args:
            out: model output bundle.
            gt_centers_px: (B, 2) GT centers in image pixel coords.
            gt_classes: (B,) long.
        """
        B, _, H, W = out.heatmap.shape
        device = out.heatmap.device
        stride = out.stride

        target_hm = gaussian_target(gt_centers_px, H, W, stride, self.sigma)
        heatmap_loss = focal_heatmap_loss(out.heatmap, target_hm)

        # Use the rounded cell to match where the heatmap target peaks.
        # offset target is the residual in [-0.5, 0.5] of one cell;
        # combined with sigmoid-bounded predictions in [0, 1], we shift
        # by 0.5 so the network's natural output center (sigmoid==0.5)
        # corresponds to "no offset".
        gx = gt_centers_px[:, 0] / stride
        gy = gt_centers_px[:, 1] / stride
        cell_x = gx.round().long().clamp(0, W - 1)
        cell_y = gy.round().long().clamp(0, H - 1)
        target_offset = torch.stack(
            [gx - cell_x.float() + 0.5, gy - cell_y.float() + 0.5], dim=-1,
        )

        batch_idx = torch.arange(B, device=device)
        pred_offset = out.offset[batch_idx, :, cell_y, cell_x]
        offset_loss = F.l1_loss(pred_offset, target_offset)

        pred_class_logits = out.class_logits[batch_idx, :, cell_y, cell_x]
        class_loss = F.cross_entropy(pred_class_logits, gt_classes)

        total = (
            self.lambda_heatmap * heatmap_loss
            + self.lambda_offset * offset_loss
            + self.lambda_class * class_loss
        )
        return HeatmapLossTerms(
            total=total, heatmap=heatmap_loss, offset=offset_loss, class_=class_loss,
        )
