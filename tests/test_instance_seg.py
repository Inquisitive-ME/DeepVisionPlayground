"""Tests for instance segmentation: model, offset grouping decode, loss, metric."""
from __future__ import annotations

import pytest
import torch

from models.instance_seg_net import InstanceSegNet, InstanceSegOutput
from utils.metrics import evaluate_instance_segmentation
from utils.seg_loss import InstanceSegLoss


class TestInstanceSegNet:
    def test_output_shapes(self):
        net = InstanceSegNet(num_classes=3, stride=2)
        out = net(torch.randn(2, 3, 64, 64))
        assert out.semantic_logits.shape == (2, 4, 32, 32)  # num_classes + bg
        assert out.heatmap.shape == (2, 1, 32, 32)
        assert out.offset.shape == (2, 2, 32, 32)
        assert out.stride == 2

    def test_decode_separates_two_same_class_blobs(self):
        """Two class-0 blobs with a center peak each must decode to two distinct
        instances, each covering its blob. (Zero offset -> nearest-center.)"""
        net = InstanceSegNet(num_classes=3, stride=2)
        hw = 16
        sem = torch.full((1, 4, hw, hw), -10.0)
        sem[0, 3] = 10.0  # background everywhere
        for cols in (slice(2, 6), slice(10, 14)):
            sem[0, 3, 4:8, cols] = -10.0
            sem[0, 0, 4:8, cols] = 10.0  # class 0 blob
        hm = torch.full((1, 1, hw, hw), -10.0)
        hm[0, 0, 5, 3] = 10.0   # left center
        hm[0, 0, 5, 12] = 10.0  # right center
        out = InstanceSegOutput(
            semantic_logits=sem, heatmap=hm, offset=torch.zeros((1, 2, hw, hw)), stride=2,
        )

        _, instances, scores = net.decode(out, max_objects=8, score_threshold=0.3)
        ids = [i for i in instances.unique().tolist() if i != 0]
        assert len(ids) == 2
        assert scores[0].numel() == 2
        left_id = instances[0, 5, 3].item()
        right_id = instances[0, 5, 12].item()
        assert left_id != 0 and right_id != 0 and left_id != right_id
        assert (instances[0, 4:8, 2:6] == left_id).all()
        assert (instances[0, 4:8, 10:14] == right_id).all()


def test_instance_seg_loss_runs_and_backprops():
    net = InstanceSegNet(num_classes=3, stride=2)
    out = net(torch.randn(2, 3, 64, 64))
    class_map = torch.randint(0, 4, (2, 64, 64))
    instance_map = torch.randint(0, 3, (2, 64, 64))
    centers = [torch.tensor([[20.0, 20.0], [40.0, 40.0]]),
               torch.tensor([[10.0, 10.0], [40.0, 40.0]])]
    terms = InstanceSegLoss()(out, class_map, instance_map, centers)
    terms.total.backward()
    for t in (terms.total, terms.semantic, terms.heatmap, terms.offset):
        assert torch.isfinite(t)


class TestInstanceMetric:
    def _two_instances(self):
        gt = torch.zeros(8, 8, dtype=torch.long)
        gt[0:4, 0:4] = 1
        gt[0:4, 4:8] = 2
        return gt

    def test_perfect(self):
        gt = self._two_instances()
        m = evaluate_instance_segmentation([(gt.clone(), gt.clone(), torch.tensor([0.9, 0.8]))])
        assert m.mean_iou == pytest.approx(1.0)
        assert m.recall_at[0.5] == pytest.approx(1.0)
        assert m.pq == pytest.approx(1.0)
        assert m.mask_ap[0.5] == pytest.approx(1.0)
        assert m.n_gt == 2 and m.n_pred == 2

    def test_partial_overlap_known_iou(self):
        gt = self._two_instances()
        pred = gt.clone()
        pred[3, 0:4] = 0  # drop a row of instance 1 -> IoU 12/16 = 0.75
        m = evaluate_instance_segmentation([(pred, gt)])  # scores optional
        assert m.mean_iou == pytest.approx(0.875)         # (0.75 + 1.0) / 2
        assert m.recall_at[0.75] == pytest.approx(1.0)
        assert m.pq == pytest.approx(0.875)               # both IoU > 0.5

    def test_missed_instance(self):
        gt = self._two_instances()
        pred = torch.zeros(8, 8, dtype=torch.long)
        pred[0:4, 0:4] = 1  # only instance 1 predicted
        m = evaluate_instance_segmentation([(pred, gt, torch.tensor([0.9]))])
        assert m.mean_iou == pytest.approx(0.5)           # (1.0 + 0) / 2 GT
        assert m.recall_at[0.5] == pytest.approx(0.5)
        assert m.pq == pytest.approx(1.0 / 1.5)           # 1 TP, 1 FN
