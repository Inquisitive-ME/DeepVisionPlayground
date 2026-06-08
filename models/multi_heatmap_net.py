"""Multi-object heatmap predictor: CenterNet-style with top-K NMS decode.

Single-object CenterHeatmapNet returns one (cx, cy, class) per image;
this module generalizes to detecting up to ``max_objects`` per image:

- Same encoder/decoder + heatmap/offset/class heads as CenterHeatmapNet.
- Heatmap target is the per-pixel max of one Gaussian per ground-truth
  object (rather than a single Gaussian for the whole image), so each
  shape's cell can fire independently.
- Decode runs a 3x3 max-pool NMS on the heatmap, takes the top-K peaks,
  and returns (B, K, 4) tensors of (cx_px, cy_px, score, class_id).

The result is meant to flow straight into ``utils.metrics.evaluate_multi_object``
once you turn it into the (B, K, 3 + num_classes) tensor the existing
metrics expect.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.center_heatmap_net import HeatmapOutput, _gn


@dataclass
class MultiHeatmapDecode:
    centers_px: torch.Tensor  # (B, K, 2) sub-pixel center positions
    scores: torch.Tensor      # (B, K)    sigmoid'd heatmap peak heights
    class_ids: torch.Tensor   # (B, K)    long


class MultiHeatmapNet(nn.Module):
    """Same architecture as CenterHeatmapNet, distinguished only by decode.

    Kept as its own class so the train loop / driver can dispatch on
    `isinstance` and so the per-class head dimensionality can diverge
    later if we want a separate "objectness" channel.
    """

    def __init__(self, num_classes: int, stride: int = 4) -> None:
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
        self.heatmap_head = nn.Conv2d(cur_channels, 1, 1)
        self.offset_head = nn.Conv2d(cur_channels, 2, 1)
        self.class_head = nn.Conv2d(cur_channels, num_classes, 1)
        if self.heatmap_head.bias is not None:
            nn.init.constant_(self.heatmap_head.bias, -2.19)  # ≈ logit(0.1)
        self.num_classes = num_classes
        self.stride = stride

    def forward(self, x: torch.Tensor) -> HeatmapOutput:
        feat = self.encoder(x)
        feat = self.decoder(feat)
        return HeatmapOutput(
            heatmap=self.heatmap_head(feat),
            offset=torch.sigmoid(self.offset_head(feat)),
            class_logits=self.class_head(feat),
            stride=self.stride,
        )

    @torch.no_grad()
    def decode(
        self,
        out: HeatmapOutput,
        max_objects: int,
        nms_kernel: int = 3,
        dedup_radius: float | None = None,
    ) -> MultiHeatmapDecode:
        """Top-K NMS decode of a heatmap.

        For each image, suppress non-maximal heatmap pixels via a 3x3
        max pool, take the top-K remaining cells, and return their
        sub-pixel positions, peak heights (as confidences), and class
        argmax. K is fixed (max_objects); the caller filters by score
        threshold downstream.

        nms_kernel is the size of the local-max kernel. 3 means a peak
        must be a strict local max in its 3x3 neighborhood. Keeping it
        small avoids suppressing nearby distinct shapes.

        The max-pool ``==`` test does NOT break ties on a flat plateau
        (several adjacent cells with exactly equal scores), so a single
        object can decode to multiple near-identical peaks. ``dedup_radius``
        applies a greedy center-distance suppression after decode — a peak
        within ``dedup_radius`` px of an already-kept higher-scoring peak has
        its score zeroed (so the downstream confidence threshold drops it).
        Defaults to two output cells (``2 * stride`` px), enough to collapse a
        small plateau into one detection while staying well below the spacing of
        genuinely distinct shapes; set 0 to disable.
        """
        B, _, H, W = out.heatmap.shape
        device = out.heatmap.device
        prob = torch.sigmoid(out.heatmap)
        # NMS via max-pool: a pixel is kept iff it equals the local max.
        pooled = F.max_pool2d(prob, kernel_size=nms_kernel, stride=1, padding=nms_kernel // 2)
        keep_mask = (pooled == prob).float()
        suppressed = prob * keep_mask
        # Top-K over the remaining peaks.
        flat = suppressed.view(B, -1)
        top_scores, top_idx = flat.topk(max_objects, dim=-1)
        py = top_idx // W
        px = top_idx % W
        # Gather offsets and class logits per peak.
        # offset/class heads are (B, C, H, W); we need (B, K, C).
        batch_idx = torch.arange(B, device=device)[:, None].expand(-1, max_objects)
        peak_offset = out.offset[batch_idx, :, py, px]            # (B, K, 2)
        peak_class_logits = out.class_logits[batch_idx, :, py, px]  # (B, K, num_classes)
        class_ids = peak_class_logits.argmax(dim=-1)               # (B, K)
        cx_px = (px.float() + peak_offset[..., 0] - 0.5) * out.stride
        cy_px = (py.float() + peak_offset[..., 1] - 0.5) * out.stride
        centers_px = torch.stack([cx_px, cy_px], dim=-1)           # (B, K, 2)

        radius = (2.0 * out.stride) if dedup_radius is None else dedup_radius
        if radius > 0 and max_objects > 1:
            # top_scores is already sorted descending per image, so iterating
            # k in order is highest-score-first — standard greedy NMS.
            for b in range(B):
                kept: list[int] = []
                for k in range(max_objects):
                    if top_scores[b, k] <= 0:
                        continue
                    c = centers_px[b, k]
                    if any(
                        torch.linalg.norm(c - centers_px[b, j]) <= radius for j in kept
                    ):
                        top_scores[b, k] = 0.0  # duplicate -> filtered downstream
                    else:
                        kept.append(k)

        return MultiHeatmapDecode(
            centers_px=centers_px,
            scores=top_scores,
            class_ids=class_ids,
        )
