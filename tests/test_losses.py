"""Tests for CenterPredictionLoss correctness."""
from __future__ import annotations

import torch

from models.types import ModelType
from utils.losses import CenterPredictionLoss


def _make_pred(batch=2, max_objects=4, num_classes=3, requires_grad=True):
    raw = torch.randn(batch, max_objects, 3 + num_classes, requires_grad=requires_grad)
    centers = torch.sigmoid(raw[..., :2])
    conf = torch.sigmoid(raw[..., 2:3])
    cls = raw[..., 3:]
    pred = torch.cat([centers, conf, cls], dim=-1)
    return raw, pred


class TestLossGradients:
    def test_gradient_flows_to_predictions(self):
        raw, pred = _make_pred()
        loss_fn = CenterPredictionLoss(model_type=ModelType.center_localization_and_class_id)
        gt_centers = [
            torch.tensor([[0.5, 0.5]]),
            torch.tensor([[0.3, 0.7], [0.6, 0.2]]),
        ]
        gt_classes = [torch.tensor([0]), torch.tensor([1, 2])]
        loss = loss_fn(pred, gt_centers, gt_classes)
        loss.backward()
        assert raw.grad is not None
        assert raw.grad.abs().sum() > 0

    def test_handles_image_with_no_objects(self):
        raw, pred = _make_pred()
        loss_fn = CenterPredictionLoss(model_type=ModelType.center_localization_and_class_id)
        gt_centers = [torch.zeros((0, 2)), torch.tensor([[0.4, 0.4]])]
        gt_classes = [torch.zeros((0,), dtype=torch.long), torch.tensor([1])]
        loss = loss_fn(pred, gt_centers, gt_classes)
        loss.backward()
        assert torch.isfinite(loss)
        assert raw.grad is not None
        assert raw.grad.abs().sum() > 0

    def test_no_constant_inflation_on_unmatched_gts(self):
        """Pre-fix bug: BCE(zeros, ones) added a constant ~100 per missed
        detection, with no gradient. Ensure the loss is now bounded by a
        sane value (well under 100) on a typical small batch."""
        raw, pred = _make_pred()
        loss_fn = CenterPredictionLoss(model_type=ModelType.center_localization_and_class_id)
        gt_centers = [
            torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]),
            torch.tensor([[0.4, 0.4]]),
        ]
        gt_classes = [torch.tensor([0, 1, 2]), torch.tensor([0])]
        loss = loss_fn(pred, gt_centers, gt_classes)
        # Random predictions on a 4-slot model with 3+1 GTs shouldn't
        # produce a >50 loss; the buggy version did.
        assert float(loss) < 10.0

    def test_class_does_not_override_geometric_match(self):
        """Two GTs ~38 px apart; each prediction sits exactly on its matching
        GT but its class logits point at the OTHER GT's class. With the small
        default class_match_weight, matching must follow geometry, so the
        coordinate loss (isolated via lambda_class=lambda_conf=0) is ~0. A large
        weight lets class flip the assignment and the coordinate loss rises."""
        # pred0 @ (0.40, 0.40) favors class 1; pred1 @ (0.55, 0.40) favors class 0.
        pred = torch.tensor([[
            [0.40, 0.40, 0.9, -2.0, 2.0, -2.0],
            [0.55, 0.40, 0.9, 2.0, -2.0, -2.0],
        ]])
        gt_centers = [torch.tensor([[0.40, 0.40], [0.55, 0.40]])]
        gt_classes = [torch.tensor([0, 1])]  # GT0=class0, GT1=class1

        coord_only = dict(lambda_class=0.0, lambda_conf=0.0)
        loss_default = CenterPredictionLoss(
            model_type=ModelType.center_localization_and_class_id, **coord_only,
        )(pred, gt_centers, gt_classes)
        loss_class_dominated = CenterPredictionLoss(
            model_type=ModelType.center_localization_and_class_id,
            class_match_weight=1.0, **coord_only,
        )(pred, gt_centers, gt_classes)

        # Geometry-respecting match -> predictions coincide with GT -> ~0 coord.
        assert float(loss_default) < 1e-4
        # Class-dominated match flips the pairing -> nonzero coordinate loss.
        assert float(loss_class_dominated) > float(loss_default)

    def test_lambda_class_weights_class_term(self):
        raw, pred = _make_pred()
        gt_centers = [torch.tensor([[0.5, 0.5]])] * 2
        gt_classes = [torch.tensor([0])] * 2

        loss_default = CenterPredictionLoss(
            model_type=ModelType.center_localization_and_class_id,
            lambda_class=1.0,
        )(pred.clone(), gt_centers, gt_classes)
        loss_heavy_class = CenterPredictionLoss(
            model_type=ModelType.center_localization_and_class_id,
            lambda_class=10.0,
        )(pred.clone(), gt_centers, gt_classes)
        assert float(loss_heavy_class) > float(loss_default)
