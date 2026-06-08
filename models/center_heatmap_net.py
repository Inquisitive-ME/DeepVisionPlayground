"""CenterNet-style heatmap predictor for single-object center + class.

Why this model exists: a regression-only head can't localize finer than
one feature cell (16 px on our 16x16 backbone, 256x256 input), and
empirically sits at ~16 px median error no matter how long we train.
The heatmap formulation breaks past that floor:

- Predict a (256/stride)x(256/stride) heatmap whose argmax is the
  center cell, trained to peak at the GT location with a Gaussian target.
- Predict a (2,) sub-pixel offset *at* the heatmap peak, trained on the
  fractional part of the GT center.
- Predict per-pixel class logits at the heatmap peak.

Stride is configurable: stride=4 is fast (64x64 heatmap on a 256-px
input) and converges to a few-pixel median error; stride=2 (128x128)
gives sub-pixel error at ~2x the decoder cost; stride=1 (256x256) is
exact but expensive.

This is a faithful single-object simplification of CenterNet (Zhou et al.
2019). Multi-object would need NMS on the heatmap; we skip that here.

Inference:
    cell_idx = heatmap.argmax over (H, W)
    pred_pixel = (cell_idx + offset_at_cell - 0.5) * stride
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


def _gn(c: int) -> nn.GroupNorm:
    """8-group GroupNorm — train/eval-safe replacement for BN on this task.

    BN running stats drifted on fresh-data-every-epoch training and made
    eval predictions diverge from train; GroupNorm is per-image so the
    two phases behave identically.
    """
    return nn.GroupNorm(8, c)


class CenterHeatmapNet(nn.Module):
    """Strided encoder + transposed-conv decoder + (heatmap, offset, class) heads.

    For a 256x256 input the output heatmap is (256/stride) on each side.
    stride=4 (default) → 64x64; stride=2 → 128x128; stride=1 → 256x256.
    """

    def __init__(self, num_classes: int, stride: int = 4) -> None:
        super().__init__()
        if stride not in (1, 2, 4, 8, 16):
            raise ValueError(f"stride must be a power of two in 1..16, got {stride}")
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            _gn(32), nn.ReLU(inplace=True),                                   # 128
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            _gn(64), nn.ReLU(inplace=True),                                   # 64
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            _gn(128), nn.ReLU(inplace=True),                                  # 32
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            _gn(128), nn.ReLU(inplace=True),                                  # 16
        )
        # The encoder produces a 16x16 feature map (stride 16). Each
        # transposed conv halves the stride. To reach the requested
        # output stride we add upsamples until the spatial size matches.
        decoder_layers: list[nn.Module] = []
        cur_channels = 128
        # Channel schedule for upsampling stages, in order.
        channel_schedule = [64, 32, 32, 32, 32]
        cur_stride = 16
        i = 0
        while cur_stride > stride:
            out_ch = channel_schedule[i]
            decoder_layers.extend([
                nn.ConvTranspose2d(cur_channels, out_ch, 4, stride=2, padding=1, bias=False),
                _gn(out_ch),
                nn.ReLU(inplace=True),
            ])
            cur_channels = out_ch
            cur_stride //= 2
            i += 1
        self.decoder = nn.Sequential(*decoder_layers) if decoder_layers else nn.Identity()
        head_in = cur_channels
        self.heatmap_head = nn.Conv2d(head_in, 1, 1)
        self.offset_head = nn.Conv2d(head_in, 2, 1)
        self.class_head = nn.Conv2d(head_in, num_classes, 1)
        # Bias the heatmap toward "no object" at init so the focal loss
        # doesn't explode the first few steps.
        if self.heatmap_head.bias is not None:
            nn.init.constant_(self.heatmap_head.bias, -2.19)  # ≈ logit(0.1)
        self.num_classes = num_classes
        self.stride = stride

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
        batch_idx = torch.arange(B, device=out.heatmap.device)
        peak_offset = out.offset[batch_idx, :, py, px]  # (B, 2)
        peak_class_logits = out.class_logits[batch_idx, :, py, px]
        class_ids = peak_class_logits.argmax(dim=-1)
        # Reverse the +0.5 shift the loss applies to the offset target.
        cx_px = (px.float() + peak_offset[:, 0] - 0.5) * out.stride
        cy_px = (py.float() + peak_offset[:, 1] - 0.5) * out.stride
        centers_px = torch.stack([cx_px, cy_px], dim=-1)
        return centers_px, class_ids
