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


def multi_gaussian_target(
    centers_per_image: list[torch.Tensor],
    out_h: int,
    out_w: int,
    stride: int,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Like gaussian_target but with one Gaussian per object and per-pixel max.

    Args:
        centers_per_image: list of length B; each entry is (N_b, 2) GT centers
            in image pixel space. Variable N_b.

    Returns:
        (B, 1, H, W) target heatmap. For each pixel, the max over all
        per-object Gaussians at that pixel. Lets two nearby objects each
        be "the peak" of their own Gaussian without one cancelling the
        other (which a sum would do).
    """
    B = len(centers_per_image)
    device = centers_per_image[0].device if B > 0 else torch.device("cpu")
    yy = torch.arange(out_h, device=device, dtype=torch.float32).view(1, out_h, 1)
    xx = torch.arange(out_w, device=device, dtype=torch.float32).view(1, 1, out_w)
    target = torch.zeros(B, out_h, out_w, device=device, dtype=torch.float32)
    for b, c in enumerate(centers_per_image):
        if c.numel() == 0:
            continue
        gx = (c[:, 0] / stride).round()
        gy = (c[:, 1] / stride).round()
        # (N, 1, 1) so broadcasting against (1, H, 1) and (1, 1, W) gives (N, H, W).
        sq = (xx - gx.view(-1, 1, 1)) ** 2 + (yy - gy.view(-1, 1, 1)) ** 2
        per_obj = torch.exp(-sq / (2 * sigma ** 2))
        target[b] = per_obj.max(dim=0).values
    return target.unsqueeze(1)  # (B, 1, H, W)


class MultiHeatmapLoss(nn.Module):
    """Multi-object analogue of HeatmapLoss.

    Heatmap term is focal loss on the per-pixel max of all per-object
    Gaussians. Offset and class terms are summed across every GT cell
    (one entry per object) and normalized by total number of objects in
    the batch.
    """

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
        gt_centers_px_per_image: list[torch.Tensor],
        gt_classes_per_image: list[torch.Tensor],
    ) -> HeatmapLossTerms:
        B, _, H, W = out.heatmap.shape
        device = out.heatmap.device
        stride = out.stride

        target_hm = multi_gaussian_target(
            gt_centers_px_per_image, H, W, stride, self.sigma,
        )
        heatmap_loss = focal_heatmap_loss(out.heatmap, target_hm)

        # Pool all objects across the batch into flat tensors so we can
        # compute one offset L1 and one class CE in one pass.
        flat_b: list[int] = []
        flat_cell_y: list[int] = []
        flat_cell_x: list[int] = []
        flat_target_offset: list[torch.Tensor] = []
        flat_class: list[int] = []
        for b, (centers, classes) in enumerate(zip(
            gt_centers_px_per_image, gt_classes_per_image,
        )):
            if centers.numel() == 0:
                continue
            gx = centers[:, 0] / stride
            gy = centers[:, 1] / stride
            cx = gx.round().long().clamp(0, W - 1)
            cy = gy.round().long().clamp(0, H - 1)
            offsets = torch.stack(
                [gx - cx.float() + 0.5, gy - cy.float() + 0.5], dim=-1,
            )
            for i in range(centers.shape[0]):
                flat_b.append(b)
                flat_cell_y.append(int(cy[i].item()))
                flat_cell_x.append(int(cx[i].item()))
                flat_target_offset.append(offsets[i])
                flat_class.append(int(classes[i].item()))

        if not flat_b:
            offset_loss = torch.tensor(0.0, device=device)
            class_loss = torch.tensor(0.0, device=device)
        else:
            b_idx = torch.tensor(flat_b, dtype=torch.long, device=device)
            y_idx = torch.tensor(flat_cell_y, dtype=torch.long, device=device)
            x_idx = torch.tensor(flat_cell_x, dtype=torch.long, device=device)
            target_off = torch.stack(flat_target_offset)
            target_cls = torch.tensor(flat_class, dtype=torch.long, device=device)
            pred_off = out.offset[b_idx, :, y_idx, x_idx]
            pred_cls = out.class_logits[b_idx, :, y_idx, x_idx]
            offset_loss = F.l1_loss(pred_off, target_off)
            class_loss = F.cross_entropy(pred_cls, target_cls)

        total = (
            self.lambda_heatmap * heatmap_loss
            + self.lambda_offset * offset_loss
            + self.lambda_class * class_loss
        )
        return HeatmapLossTerms(
            total=total, heatmap=heatmap_loss, offset=offset_loss, class_=class_loss,
        )
