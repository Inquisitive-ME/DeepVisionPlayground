"""CenterNet-style heatmap predictor for single-object center + class.

Why this model exists: a regression-only head can't localize finer than
one feature cell (16 px on our 16x16 backbone, 256x256 input), and
empirically sits at ~16 px median error no matter how long we train.
The heatmap formulation breaks past that floor:

- Predict a 64x64 (stride-4) heatmap whose argmax is the center cell,
  trained to peak at the GT location with a Gaussian target.
- Predict a (2,) sub-pixel offset *at* the heatmap peak, trained on the
  fractional part of the GT center.
- Predict per-pixel class logits at the heatmap peak.

This is a faithful single-object simplification of CenterNet (Zhou et al.
2019). Multi-object would need NMS on the heatmap; we skip that here.

Inference:
    cell_idx = heatmap.argmax over (H, W)
    pred_pixel = cell_idx * stride + offset_at_cell * stride
    pred_class = class_logits_at_cell.argmax
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class HeatmapOutput:
    """Bundle of per-image predictions from CenterHeatmapNet.

    Stored on whatever device the model is on.
    """
    heatmap: torch.Tensor       # (B, 1, H, W) raw logits (use sigmoid for prob)
    offset: torch.Tensor        # (B, 2, H, W) (dx, dy) in [0, 1] of stride
    class_logits: torch.Tensor  # (B, num_classes, H, W)
    stride: int


class CenterHeatmapNet(nn.Module):
    """Encoder (stride 16) + 2-step transposed-conv decoder (stride 4 output).

    For a 256x256 input, the output heatmap is 64x64. That's a per-cell
    coverage of 4 pixels; the offset head bridges the remaining sub-pixel.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        # Encoder: 4 strided convs with GroupNorm (8 groups). BatchNorm
        # was an early choice, but on this fresh-data-every-epoch task
        # the BN running stats drifted enough that eval-mode predictions
        # diverged from the train-mode loss. GroupNorm is per-image so
        # train/eval behave identically.
        def gn(c: int) -> nn.GroupNorm:
            return nn.GroupNorm(8, c)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            gn(32), nn.ReLU(inplace=True),                                    # 128
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            gn(64), nn.ReLU(inplace=True),                                    # 64
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            gn(128), nn.ReLU(inplace=True),                                   # 32
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            gn(128), nn.ReLU(inplace=True),                                   # 16
        )
        # Decoder: upsample 16->32->64 with transposed convs.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            gn(64), nn.ReLU(inplace=True),                                    # 32
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),
            gn(32), nn.ReLU(inplace=True),                                    # 64
        )
        # Heads. 1x1 convs so each spatial location is treated independently.
        self.heatmap_head = nn.Conv2d(32, 1, 1)
        self.offset_head = nn.Conv2d(32, 2, 1)
        self.class_head = nn.Conv2d(32, num_classes, 1)
        # Bias the heatmap toward "no object" at init so the BCE loss
        # doesn't explode the first few steps.
        if self.heatmap_head.bias is not None:
            nn.init.constant_(self.heatmap_head.bias, -2.19)  # ≈ logit(0.1)
        self.num_classes = num_classes
        self.stride = 4

    def forward(self, x: torch.Tensor) -> HeatmapOutput:
        feat = self.encoder(x)
        feat = self.decoder(feat)
        # Sub-pixel offset is in [0, 1] of one cell — sigmoid keeps it bounded
        # so a wandering raw output can't decode the predicted center to off-canvas.
        return HeatmapOutput(
            heatmap=self.heatmap_head(feat),
            offset=torch.sigmoid(self.offset_head(feat)),
            class_logits=self.class_head(feat),
            stride=self.stride,
        )

    @torch.no_grad()
    def decode(self, out: HeatmapOutput) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode an output bundle into (centers_px, class_ids).

        Returns:
            centers_px: (B, 2) float tensor with sub-pixel center positions
                in image-pixel coordinates.
            class_ids: (B,) long tensor of predicted class.
        """
        B, _, H, W = out.heatmap.shape
        flat = out.heatmap.view(B, -1)
        peak_idx = flat.argmax(dim=1)  # (B,)
        py = peak_idx // W
        px = peak_idx % W
        # Gather offset and class at the peak.
        # offset has shape (B, 2, H, W) — index along last two dims.
        batch_idx = torch.arange(B, device=out.heatmap.device)
        peak_offset = out.offset[batch_idx, :, py, px]  # (B, 2)
        peak_class_logits = out.class_logits[batch_idx, :, py, px]  # (B, num_classes)
        class_ids = peak_class_logits.argmax(dim=-1)
        # The training target shifts the per-cell offset by +0.5 so the
        # sigmoid's natural midpoint corresponds to "no offset". Reverse
        # that shift here so the decoded position lines up with image pixels.
        cx_px = (px.float() + peak_offset[:, 0] - 0.5) * out.stride
        cy_px = (py.float() + peak_offset[:, 1] - 0.5) * out.stride
        centers_px = torch.stack([cx_px, cy_px], dim=-1)
        return centers_px, class_ids
