"""Semantic segmentation network for the synthetic shapes.

Same strided GroupNorm encoder + transposed-conv decoder as
``CenterHeatmapNet``, but the head is a per-pixel ``(num_classes + 1)`` logit
map — the ``+1`` is the background class (index ``num_classes``), matching the
label convention the data path emits.

Output stride is configurable; ``stride=1`` is full resolution, the target for
"fully solve it" (mIoU -> 1.0). Larger strides are cheaper and upsampled by the
loss/metric via nearest-neighbour, trading boundary precision for speed.

    logits = net(images)              # (B, num_classes + 1, H/stride, W/stride)
    pred_labels = net.decode(logits)  # (B, H/stride, W/stride) argmax classes
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.center_heatmap_net import _gn


class ShapeSegNet(nn.Module):
    def __init__(self, num_classes: int, stride: int = 1) -> None:
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
        decoder_layers: list[nn.Module] = []
        cur_channels = 128
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
        # +1 channel for the background class (index == num_classes).
        self.seg_head = nn.Conv2d(cur_channels, num_classes + 1, 1)
        self.num_classes = num_classes
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        feat = self.decoder(feat)
        return self.seg_head(feat)  # (B, num_classes + 1, H/stride, W/stride)

    @torch.no_grad()
    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Per-pixel argmax -> (B, H, W) long class map."""
        return logits.argmax(dim=1)
