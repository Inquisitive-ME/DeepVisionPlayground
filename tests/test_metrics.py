"""Tests for evaluation metrics, including per-class breakdowns."""
from __future__ import annotations

import pytest
import torch

from utils.metrics import evaluate_multi_object, evaluate_single_object


class TestSingleObject:
    def test_perfect_predictions(self):
        preds = torch.tensor([[0.5, 0.5, 1.0, 0.0, 0.0], [0.3, 0.7, 0.0, 1.0, 0.0]])
        gt = torch.tensor([[0.5, 0.5], [0.3, 0.7]])
        gt_classes = torch.tensor([0, 1])
        m = evaluate_single_object(
            preds, gt, image_size=(256, 256),
            gt_classes=gt_classes, has_classes=True,
        )
        assert m.mean_center_px == 0.0
        assert m.accuracy == 1.0

    def test_pixel_offset_is_in_pixels(self):
        # 10/256 in normalized space -> 10 px in image space.
        preds = torch.tensor([[0.5 + 10 / 256, 0.5, 1.0, 0.0, 0.0]])
        gt = torch.tensor([[0.5, 0.5]])
        m = evaluate_single_object(preds, gt, image_size=(256, 256))
        assert m.mean_center_px == pytest.approx(10.0, abs=1e-3)

    def test_pearson_perfect_predictions_is_one(self):
        preds = torch.tensor([
            [0.2, 0.3, 1.0, 0.0, 0.0],
            [0.4, 0.5, 0.0, 1.0, 0.0],
            [0.7, 0.8, 0.0, 0.0, 1.0],
        ])
        gt = torch.tensor([[0.2, 0.3], [0.4, 0.5], [0.7, 0.8]])
        m = evaluate_single_object(preds, gt, image_size=(256, 256))
        assert m.pearson_cx == pytest.approx(1.0, abs=1e-6)
        assert m.pearson_cy == pytest.approx(1.0, abs=1e-6)

    def test_pearson_constant_prediction_is_zero(self):
        """Regression to the mean: Pearson should be 0 for a constant predictor.

        This is the failure mode the metric was added to expose: a model that
        always predicts (0.5, 0.5) gets a low-ish mean_center_px (~70 on a
        256x256 canvas) but Pearson 0, which lights it up immediately.
        """
        preds = torch.tensor([[0.5, 0.5, 1.0, 0.0, 0.0]] * 4)
        gt = torch.tensor([[0.2, 0.3], [0.4, 0.5], [0.7, 0.8], [0.1, 0.9]])
        m = evaluate_single_object(preds, gt, image_size=(256, 256))
        assert m.pearson_cx == 0.0
        assert m.pearson_cy == 0.0


class TestMultiObjectGlobals:
    def test_perfect_predictions_score_full(self):
        preds = torch.tensor([[
            [0.50, 0.50, 0.9, 1.0, 0.0, 0.0],
            [0.30, 0.70, 0.8, 0.0, 1.0, 0.0],
            [0.10, 0.10, 0.05, 0.0, 0.0, 1.0],
        ]])
        gt_centers = [torch.tensor([[0.50, 0.50], [0.30, 0.70]])]
        gt_classes = [torch.tensor([0, 1])]
        m = evaluate_multi_object(
            preds, gt_centers, image_size=(256, 256),
            gt_classes_list=gt_classes, has_classes=True,
        )
        assert m.mean_matched_center_px == 0.0
        assert m.precision_at[2] == 1.0
        assert m.recall_at[2] == 1.0
        assert m.matched_class_accuracy == 1.0
        assert m.cardinality_error == 0.0

    def test_empty_gt_with_false_positives(self):
        preds = torch.tensor([[
            [0.5, 0.5, 0.9, 1.0, 0.0, 0.0],
            [0.3, 0.7, 0.05, 0.0, 1.0, 0.0],
        ]])
        gt_centers = [torch.zeros((0, 2))]
        gt_classes = [torch.zeros((0,), dtype=torch.long)]
        m = evaluate_multi_object(
            preds, gt_centers, image_size=(256, 256),
            gt_classes_list=gt_classes, has_classes=True,
        )
        assert m.cardinality_error == 1.0
        for t in m.precision_at:
            assert m.precision_at[t] == 0.0


class TestMultiObjectPerClass:
    """Build a deliberately mixed batch and check per-class accounting.

    Layout:
      Image 0:
        GT: [(0.5, 0.5, RECT=0), (0.3, 0.7, CIRC=1)]
        Predictions kept (conf > 0.5):
          - (0.50, 0.50, RECT)   correct match
          - (0.30, 0.70, TRI)    location matches CIRC GT, but class is wrong
      Image 1:
        GT: [(0.4, 0.4, TRI=2)]
        Predictions kept:
          - (0.40, 0.40, TRI)    correct match
          - (0.10, 0.10, RECT)   FP — no GT to match

    Expected at threshold T=4 px on a 256x256 image:
      RECT: TP=1 (img0), FP=1 (img1 unmatched), FN=0  -> P=0.5, R=1.0
      CIRC: TP=0,         FP=0,                 FN=1 (location matched but class wrong)
                                                            -> P undefined-as-0, R=0.0
      TRI:  TP=1 (img1),  FP=1 (img0 wrong-class match), FN=0
                                                            -> P=0.5, R=1.0
    """
    @pytest.fixture
    def metrics(self):
        preds = torch.tensor([[
            [0.50, 0.50, 0.9, 5.0, 0.0, 0.0],
            [0.30, 0.70, 0.8, 0.0, 0.0, 5.0],
            [0.10, 0.10, 0.05, 1.0, 0.0, 0.0],
            [0.20, 0.20, 0.1, 1.0, 0.0, 0.0],
            [0.40, 0.40, 0.2, 1.0, 0.0, 0.0],
        ], [
            [0.40, 0.40, 0.9, 0.0, 0.0, 5.0],
            [0.10, 0.10, 0.7, 5.0, 0.0, 0.0],
            [0.0, 0.0, 0.05, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.05, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.05, 0.0, 0.0, 0.0],
        ]])
        return evaluate_multi_object(
            preds,
            gt_centers_list=[
                torch.tensor([[0.5, 0.5], [0.3, 0.7]]),
                torch.tensor([[0.4, 0.4]]),
            ],
            image_size=(256, 256),
            gt_classes_list=[
                torch.tensor([0, 1]),
                torch.tensor([2]),
            ],
            has_classes=True,
            class_names=("RECT", "CIRC", "TRI"),
        )

    def test_rect_precision_and_recall(self, metrics):
        assert metrics.per_class_precision_at[0][4] == pytest.approx(0.5)
        assert metrics.per_class_recall_at[0][4] == pytest.approx(1.0)

    def test_circ_recall_zero_due_to_class_mismatch(self, metrics):
        assert metrics.per_class_recall_at[1][4] == pytest.approx(0.0)

    def test_tri_precision_and_recall(self, metrics):
        assert metrics.per_class_precision_at[2][4] == pytest.approx(0.5)
        assert metrics.per_class_recall_at[2][4] == pytest.approx(1.0)

    def test_class_names_propagate_to_tags(self, metrics):
        d = metrics.to_dict()
        assert "multi/RECT/precision@4px" in d
        assert "multi/CIRC/recall@4px" in d
        assert "multi/TRI/mean_center_px" in d

    def test_n_gt_and_n_pred_counts(self, metrics):
        assert metrics.per_class_n_gt == {0: 1, 1: 1, 2: 1}
        # After conf-threshold: img0 keeps 2 (RECT, TRI), img1 keeps 2 (TRI, RECT).
        assert metrics.per_class_n_pred == {0: 2, 1: 0, 2: 2}
