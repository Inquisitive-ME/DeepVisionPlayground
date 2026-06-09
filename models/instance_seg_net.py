"""Instance segmentation — semantic + center-heatmap, nearest-center grouping.

A shared GroupNorm encoder-decoder feeds two heads:
- a per-pixel **semantic** head ``(num_classes + 1, h, w)`` (background = num_classes), and
- a single-channel **center heatmap** ``(1, h, w)`` (CenterNet-style).

Decode = group each foreground pixel to the nearest detected center (a
Panoptic-DeepLab-style grouping, minus the offset head). This separates
individual shapes that share a semantic class. It reuses ``SegLoss``, the
CenterNet ``focal_heatmap_loss`` + ``multi_gaussian_target``, and the
max-pool NMS, so it is a small delta over the semantic + detection tasks.

A per-pixel offset-to-center head (full Panoptic-DeepLab) would improve heavily
overlapping cases; see ``docs/instance_segmentation_design.md``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.center_heatmap_net import _gn


@dataclass
class InstanceSegOutput:
    semantic_logits: torch.Tensor  # (B, num_classes + 1, h, w)
    heatmap: torch.Tensor          # (B, 1, h, w) center heatmap logits
    stride: int


class InstanceSegNet(nn.Module):
    def __init__(self, num_classes: int, stride: int = 2) -> None:
        super().__init__()
        if stride not in (1, 2, 4, 8, 16):
            raise ValueError(f"stride must be a power of two in 1..16, got {stride}")
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            _gn(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            _gn(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            _gn(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            _gn(128), nn.ReLU(inplace=True),
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
        self.semantic_head = nn.Conv2d(cur_channels, num_classes + 1, 1)
        self.heatmap_head = nn.Conv2d(cur_channels, 1, 1)
        if self.heatmap_head.bias is not None:
            nn.init.constant_(self.heatmap_head.bias, -2.19)  # ≈ logit(0.1), CenterNet init
        self.num_classes = num_classes
        self.stride = stride

    def forward(self, x: torch.Tensor) -> InstanceSegOutput:
        feat = self.decoder(self.encoder(x))
        return InstanceSegOutput(
            semantic_logits=self.semantic_head(feat),
            heatmap=self.heatmap_head(feat),
            stride=self.stride,
        )

    @torch.no_grad()
    def decode(
        self,
        out: InstanceSegOutput,
        max_objects: int = 20,
        nms_kernel: int = 3,
        score_threshold: float = 0.3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode to ``(semantic, instances)``, both (B, h, w) long at the output
        stride. ``semantic`` is the per-pixel class (background = num_classes);
        ``instances`` labels each foreground pixel with the id (1-based) of its
        nearest detected center (0 = background)."""
        B, _, hh, ww = out.semantic_logits.shape
        device = out.semantic_logits.device
        bg = self.num_classes
        semantic = out.semantic_logits.argmax(dim=1)  # (B, h, w)

        prob = torch.sigmoid(out.heatmap)             # (B, 1, h, w)
        pooled = F.max_pool2d(prob, nms_kernel, stride=1, padding=nms_kernel // 2)
        peak = (pooled == prob) & (prob > score_threshold)  # (B, 1, h, w)
        flat_scores = (prob * peak).view(B, -1)
        top_scores, top_idx = flat_scores.topk(min(max_objects, hh * ww), dim=-1)
        # cell coords of the kept peaks
        cy = (top_idx // ww).float()
        cx = (top_idx % ww).float()

        # Pixel grid in cell coordinates for nearest-center assignment.
        yy, xx = torch.meshgrid(
            torch.arange(hh, device=device, dtype=torch.float32),
            torch.arange(ww, device=device, dtype=torch.float32),
            indexing="ij",
        )
        grid = torch.stack([xx, yy], dim=-1).view(-1, 2)  # (h*w, 2)

        instances = torch.zeros((B, hh, ww), dtype=torch.long, device=device)
        for b in range(B):
            keep = top_scores[b] > score_threshold
            n_centers = int(keep.sum())
            if n_centers == 0:
                continue
            centers = torch.stack([cx[b][keep], cy[b][keep]], dim=-1)  # (Ki, 2)
            fg = (semantic[b] != bg).view(-1)                          # (h*w,)
            if not bool(fg.any()):
                continue
            d = torch.cdist(grid[fg], centers)        # (M, Ki)
            nearest = d.argmin(dim=1) + 1             # 1-based instance ids
            inst_flat = torch.zeros(hh * ww, dtype=torch.long, device=device)
            inst_flat[fg] = nearest
            instances[b] = inst_flat.view(hh, ww)
        return semantic, instances
