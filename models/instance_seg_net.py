"""Instance segmentation — semantic + center heatmap + offset (Panoptic-DeepLab).

A shared GroupNorm encoder-decoder feeds three heads:
- a per-pixel **semantic** head ``(num_classes + 1, h, w)`` (background = num_classes),
- a single-channel **center heatmap** ``(1, h, w)`` (CenterNet-style), and
- a 2-channel **offset** head: each pixel's vector to its instance's center
  (in output-cell units).

Decode: each foreground pixel votes a center (``pixel + offset``) and is grouped
to the nearest *detected* center (heatmap NMS top-K). The offset lets the model
learn the assignment, which fixes the boundary/overlap pixels that a raw
nearest-center rule mis-assigns. Reuses ``SegLoss``, the CenterNet
``focal_heatmap_loss`` + ``multi_gaussian_target``, and the max-pool NMS.
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
    offset: torch.Tensor           # (B, 2, h, w) (dx, dy) to the instance center, in cells
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
        self.offset_head = nn.Conv2d(cur_channels, 2, 1)
        if self.heatmap_head.bias is not None:
            nn.init.constant_(self.heatmap_head.bias, -2.19)  # ≈ logit(0.1), CenterNet init
        self.num_classes = num_classes
        self.stride = stride

    def forward(self, x: torch.Tensor) -> InstanceSegOutput:
        feat = self.decoder(self.encoder(x))
        return InstanceSegOutput(
            semantic_logits=self.semantic_head(feat),
            heatmap=self.heatmap_head(feat),
            offset=self.offset_head(feat),
            stride=self.stride,
        )

    @torch.no_grad()
    def decode(
        self,
        out: InstanceSegOutput,
        max_objects: int = 20,
        nms_kernel: int = 3,
        score_threshold: float = 0.3,
        use_offset: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Decode to ``(semantic, instances, scores)``.

        ``semantic`` and ``instances`` are (B, h, w) long at the output stride
        (instance ids are 1-based, 0 = background). ``scores`` is a per-image
        list of (K_b,) center confidences, aligned so instance id ``i`` has
        score ``scores[b][i-1]`` — used to rank instances for mask AP."""
        B, _, hh, ww = out.semantic_logits.shape
        device = out.semantic_logits.device
        bg = self.num_classes
        semantic = out.semantic_logits.argmax(dim=1)  # (B, h, w)

        prob = torch.sigmoid(out.heatmap)             # (B, 1, h, w)
        pooled = F.max_pool2d(prob, nms_kernel, stride=1, padding=nms_kernel // 2)
        peak = (pooled == prob) & (prob > score_threshold)
        flat_scores = (prob * peak).view(B, -1)
        top_scores, top_idx = flat_scores.topk(min(max_objects, hh * ww), dim=-1)
        cy = (top_idx // ww).float()
        cx = (top_idx % ww).float()

        yy, xx = torch.meshgrid(
            torch.arange(hh, device=device, dtype=torch.float32),
            torch.arange(ww, device=device, dtype=torch.float32),
            indexing="ij",
        )
        grid = torch.stack([xx, yy], dim=-1).view(-1, 2)  # (h*w, 2) in cell coords

        instances = torch.zeros((B, hh, ww), dtype=torch.long, device=device)
        scores: list[torch.Tensor] = []
        for b in range(B):
            keep = top_scores[b] > score_threshold
            if not bool(keep.any()):
                scores.append(torch.empty(0, device=device))
                continue
            centers = torch.stack([cx[b][keep], cy[b][keep]], dim=-1)  # (Ki, 2) cells
            scores.append(top_scores[b][keep])
            fg = (semantic[b] != bg).view(-1)
            if not bool(fg.any()):
                continue
            if use_offset:
                voted = grid + out.offset[b].permute(1, 2, 0).reshape(-1, 2)  # (h*w, 2)
                pts = voted[fg]
            else:
                pts = grid[fg]
            nearest = torch.cdist(pts, centers).argmin(dim=1) + 1  # 1-based ids
            inst_flat = torch.zeros(hh * ww, dtype=torch.long, device=device)
            inst_flat[fg] = nearest
            instances[b] = inst_flat.view(hh, ww)
        return semantic, instances, scores
