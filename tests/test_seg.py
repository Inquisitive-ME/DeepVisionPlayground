"""Tests for the segmentation model, loss, and metrics."""
from __future__ import annotations

import pytest
import torch

from models.seg_net import ShapeSegNet
from utils.metrics import evaluate_segmentation, segmentation_confusion
from utils.seg_loss import SegLoss


class TestShapeSegNet:
    @pytest.mark.parametrize("stride, out", [(1, 64), (2, 32), (4, 16)])
    def test_output_shape(self, stride, out):
        net = ShapeSegNet(num_classes=3, stride=stride)
        logits = net(torch.randn(2, 3, 64, 64))
        assert logits.shape == (2, 4, out, out)  # 3 classes + background

    def test_decode_argmax_shape(self):
        net = ShapeSegNet(num_classes=3, stride=1)
        logits = net(torch.randn(2, 3, 64, 64))
        labels = net.decode(logits)
        assert labels.shape == (2, 64, 64)
        assert labels.dtype == torch.long
        assert int(labels.max()) <= 3

    def test_gradient_flows(self):
        net = ShapeSegNet(num_classes=3, stride=2)
        x = torch.randn(2, 3, 64, 64)
        loss = net(x).square().mean()
        loss.backward()
        grads = [p.grad for p in net.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in grads)


class TestSegLoss:
    def test_zero_on_perfect_prediction(self):
        target = torch.randint(0, 4, (2, 16, 16))
        # Confident logits placing all mass on the true class -> CE ~ 0.
        logits = torch.full((2, 4, 16, 16), -20.0)
        logits.scatter_(1, target.unsqueeze(1), 20.0)
        assert float(SegLoss()(logits, target)) < 1e-3

    def test_higher_when_wrong(self):
        target = torch.zeros((2, 16, 16), dtype=torch.long)
        right = torch.full((2, 4, 16, 16), -20.0)
        right[:, 0] = 20.0
        wrong = torch.full((2, 4, 16, 16), -20.0)
        wrong[:, 1] = 20.0  # confidently predicts the wrong class everywhere
        assert float(SegLoss()(wrong, target)) > float(SegLoss()(right, target))

    def test_downsamples_target_for_strided_logits(self):
        # Logits at half resolution; loss must resize the (B,H,W) target, not crash.
        target = torch.randint(0, 4, (2, 32, 32))
        logits = torch.randn(2, 4, 16, 16)
        loss = SegLoss()(logits, target)
        assert torch.isfinite(loss)

    def test_dice_term_is_optional_and_finite(self):
        target = torch.randint(0, 4, (2, 16, 16))
        logits = torch.randn(2, 4, 16, 16, requires_grad=True)
        loss = SegLoss(lambda_dice=1.0)(logits, target)
        loss.backward()
        assert torch.isfinite(loss) and logits.grad is not None


class TestSegMetrics:
    def test_perfect_prediction(self):
        gt = torch.randint(0, 4, (2, 32, 32))
        conf = segmentation_confusion(gt, gt, num_classes_incl_bg=4)
        m = evaluate_segmentation(conf, class_names=("RECT", "CIRCLE", "TRIANGLE", "bg"))
        assert m.miou == pytest.approx(1.0)
        assert m.pixel_acc == pytest.approx(1.0)
        assert "seg/iou/bg" in m.to_dict()

    def test_known_iou(self):
        # GT all class 0 (16 px). Predict 12 as class 0, 4 as class 1.
        gt = torch.zeros(16, dtype=torch.long)
        pred = torch.tensor([0] * 12 + [1] * 4)
        conf = segmentation_confusion(pred, gt, num_classes_incl_bg=4)
        m = evaluate_segmentation(conf)
        # class0: TP=12, union = 12(gt) + 12(pred) - 12 = 12 -> IoU 1.0? no:
        # gt0=16, pred0=12, tp0=12 -> union=16+12-12=16 -> IoU 12/16 = 0.75
        assert m.per_class_iou[0] == pytest.approx(0.75)
        # class1: tp=0, gt1=0, pred1=4 -> union=4 -> IoU 0.0
        assert m.per_class_iou[1] == pytest.approx(0.0)
        assert m.pixel_acc == pytest.approx(0.75)

    def test_confusion_accumulates_across_batches(self):
        gt = torch.randint(0, 4, (32, 32))
        total = segmentation_confusion(gt, gt, 4) + segmentation_confusion(gt, gt, 4)
        assert int(total.sum()) == 2 * 32 * 32
        assert evaluate_segmentation(total).miou == pytest.approx(1.0)
