"""Tests for the CenterNet-style heatmap model + loss.

These pin the invariant the headline pixel metric depends on: that the model's
``decode`` exactly inverts the target the loss is trained against. A transpose
or off-by-stride here would silently inflate localization error with no test
catching it, so it is worth a known-answer test.
"""
from __future__ import annotations

import torch

from models.center_heatmap_net import CenterHeatmapNet, HeatmapOutput
from models.multi_heatmap_net import MultiHeatmapNet
from utils.heatmap_loss import HeatmapLoss, gaussian_target


def test_gaussian_target_peak_location_and_value():
    """Peak is exactly 1.0 at the rounded GT cell, indexed [row=y, col=x]
    (a transpose would put it at [x, y])."""
    stride, out_hw = 4, 16  # 64-px image
    centers_px = torch.tensor([[40.0, 24.0]])  # cx=40 -> col 10, cy=24 -> row 6
    target = gaussian_target(centers_px, out_hw, out_hw, stride)
    assert target.shape == (1, 1, out_hw, out_hw)
    assert torch.isclose(target.max(), torch.tensor(1.0))
    flat_peak = int(target.view(-1).argmax())
    assert (flat_peak // out_hw, flat_peak % out_hw) == (6, 10)  # (row=y, col=x)


def test_decode_inverts_loss_target():
    """Build the output a perfectly-trained model would produce for a known
    sub-pixel center (peak at the GT cell + the loss's offset target) and
    confirm decode recovers the center to sub-pixel precision."""
    stride, hw = 4, 16
    cx, cy = 41.0, 23.0
    cell_x, cell_y = round(cx / stride), round(cy / stride)  # 10, 6
    off_x = cx / stride - cell_x + 0.5  # the +0.5 shift the loss applies
    off_y = cy / stride - cell_y + 0.5

    heatmap = torch.full((1, 1, hw, hw), -10.0)
    heatmap[0, 0, cell_y, cell_x] = 10.0
    offset = torch.full((1, 2, hw, hw), 0.5)
    offset[0, 0, cell_y, cell_x] = off_x
    offset[0, 1, cell_y, cell_x] = off_y
    class_logits = torch.zeros((1, 3, hw, hw))
    class_logits[0, 2, cell_y, cell_x] = 5.0  # class 2

    net = CenterHeatmapNet(num_classes=3, stride=stride)
    centers_px, class_ids = net.decode(
        HeatmapOutput(heatmap=heatmap, offset=offset, class_logits=class_logits, stride=stride)
    )
    assert torch.allclose(centers_px[0], torch.tensor([cx, cy]), atol=1e-4)
    assert int(class_ids[0]) == 2


def test_heatmap_loss_offset_term_zero_on_matched_offset():
    """Offset L1 is exactly 0 when the predicted offset at the GT cell equals
    the loss's own target offset; all terms are finite and non-negative."""
    stride, hw = 4, 16
    cx, cy = 41.0, 23.0
    cell_x, cell_y = round(cx / stride), round(cy / stride)
    off = torch.full((1, 2, hw, hw), 0.5)
    off[0, 0, cell_y, cell_x] = cx / stride - cell_x + 0.5
    off[0, 1, cell_y, cell_x] = cy / stride - cell_y + 0.5
    out = HeatmapOutput(
        heatmap=torch.zeros((1, 1, hw, hw)),
        offset=off,
        class_logits=torch.zeros((1, 3, hw, hw)),
        stride=stride,
    )
    terms = HeatmapLoss()(out, torch.tensor([[cx, cy]]), torch.tensor([0]))
    assert float(terms.offset) == 0.0
    for t in (terms.total, terms.heatmap, terms.class_):
        assert torch.isfinite(t) and float(t) >= 0.0


class TestMultiHeatmapDecodeDedup:
    """A flat heatmap plateau (equal adjacent scores) must not decode to several
    duplicate detections of one object (the max-pool '==' NMS doesn't break the
    tie; the post-decode center dedup does)."""

    def _plateau_output(self, hw=16, stride=4):
        heatmap = torch.full((1, 1, hw, hw), -10.0)
        heatmap[0, 0, 4:6, 4:6] = 5.0   # 2x2 plateau, one object
        offset = torch.full((1, 2, hw, hw), 0.5)
        class_logits = torch.zeros((1, 3, hw, hw))
        return HeatmapOutput(heatmap=heatmap, offset=offset, class_logits=class_logits, stride=stride)

    def test_without_dedup_plateau_yields_duplicates(self):
        net = MultiHeatmapNet(num_classes=3, stride=4)
        dec = net.decode(self._plateau_output(), max_objects=5, dedup_radius=0.0)
        assert int((dec.scores[0] > 0.1).sum()) == 4  # the artifact

    def test_dedup_collapses_plateau_to_one(self):
        net = MultiHeatmapNet(num_classes=3, stride=4)
        dec = net.decode(self._plateau_output(), max_objects=5)  # default 2*stride
        assert int((dec.scores[0] > 0.1).sum()) == 1
