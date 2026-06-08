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

    def test_map_center_is_confidence_threshold_independent(self):
        """map_center is now a confidence-integrated AP, so it must not change
        when the at-threshold confidence cutoff changes (that cutoff only
        affects the diagnostic precision_at/recall_at)."""
        preds = torch.tensor([[
            [0.50, 0.50, 0.90, 1.0, 0.0, 0.0],
            [0.30, 0.70, 0.40, 0.0, 1.0, 0.0],   # correct but mid confidence
            [0.10, 0.10, 0.20, 0.0, 0.0, 1.0],   # spurious, low confidence
        ]])
        gt_centers = [torch.tensor([[0.50, 0.50], [0.30, 0.70]])]
        m_lo = evaluate_multi_object(
            preds, gt_centers, image_size=(256, 256), confidence_threshold=0.1,
        )
        m_hi = evaluate_multi_object(
            preds, gt_centers, image_size=(256, 256), confidence_threshold=0.5,
        )
        assert m_lo.map_center == pytest.approx(m_hi.map_center)
        # The at-threshold recall SHOULD differ (0.5 drops the 0.40 detection).
        assert m_hi.recall_at[2] < m_lo.recall_at[2]

    def test_map_center_tie_order_independent(self):
        """One TP and one far FP at the SAME confidence must give the same AP
        regardless of input order (pessimistic tie-break: 0.5, not 1.0)."""
        tp = [0.50, 0.50, 0.7, 1.0, 0.0, 0.0]   # on the GT
        fp = [0.05, 0.05, 0.7, 1.0, 0.0, 0.0]   # far from the GT, same conf
        gt = [torch.tensor([[0.50, 0.50]])]
        m_tp_first = evaluate_multi_object(torch.tensor([[tp, fp]]), gt, image_size=(256, 256))
        m_fp_first = evaluate_multi_object(torch.tensor([[fp, tp]]), gt, image_size=(256, 256))
        assert m_tp_first.map_center == pytest.approx(m_fp_first.map_center)
        assert m_tp_first.map_center == pytest.approx(0.5)

    def test_map_center_perfect_is_one(self):
        preds = torch.tensor([[
            [0.50, 0.50, 0.9, 1.0, 0.0, 0.0],
            [0.30, 0.70, 0.9, 0.0, 1.0, 0.0],
        ]])
        gt_centers = [torch.tensor([[0.50, 0.50], [0.30, 0.70]])]
        m = evaluate_multi_object(preds, gt_centers, image_size=(256, 256))
        assert m.map_center == pytest.approx(1.0)

    def test_map_center_penalizes_false_positives(self):
        """A model that nails both GTs but also emits many higher-confidence
        spurious detections: matched_* metrics look perfect, but the AP must
        drop because the false positives outrank the true positives."""
        real = [
            [0.50, 0.50, 0.60, 1.0, 0.0, 0.0],
            [0.30, 0.70, 0.60, 0.0, 1.0, 0.0],
        ]
        spurious = [
            [0.05 + 0.1 * i, 0.05, 0.90, 1.0, 0.0, 0.0] for i in range(8)
        ]
        preds = torch.tensor([real + spurious])
        gt_centers = [torch.tensor([[0.50, 0.50], [0.30, 0.70]])]
        m = evaluate_multi_object(
            preds, gt_centers, image_size=(256, 256), confidence_threshold=0.1,
        )
        # Matched-only localization is blind to the false positives.
        assert m.median_matched_center_px == 0.0
        # The AP sees the higher-confidence false positives and drops well below 1.
        assert m.map_center < 0.5

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


class TestSizeAndCountBuckets:
    """Bucketed metrics let us see "where the model fails" at a glance.

    Build a batch with two GTs of distinct sizes and confirm the recall
    of the small one is 100% (we pair it with a co-located prediction)
    while the large one is 0% (the matching prediction is far off).
    """

    def test_size_buckets_route_correctly(self):
        preds = torch.tensor([[
            [0.50, 0.50, 0.9, 5.0, 0.0, 0.0],   # near GT0 (small)
            [0.10, 0.10, 0.9, 0.0, 5.0, 0.0],   # far from GT1 (large)
        ]])
        gt_centers = [torch.tensor([[0.50, 0.50], [0.30, 0.70]])]
        gt_classes = [torch.tensor([0, 1])]
        # Sizes: 25 px (small bucket "xs") and 100 px (medium bucket "md").
        gt_sizes = [torch.tensor([25.0, 100.0])]

        m = evaluate_multi_object(
            preds, gt_centers, image_size=(256, 256),
            gt_classes_list=gt_classes, has_classes=True,
            gt_sizes_list=gt_sizes,
        )
        # The small GT got matched within 4 px → recall@4=1.0 in xs bucket.
        assert m.by_size_recall_at["xs"][4] == pytest.approx(1.0)
        # The medium GT was matched far away → recall@4=0 in md bucket.
        assert m.by_size_recall_at["md"][4] == pytest.approx(0.0)
        # n_gt counts route correctly.
        assert m.by_size_n_gt["xs"] == 1
        assert m.by_size_n_gt["md"] == 1
        assert m.by_size_n_gt["sm"] == 0

    def test_count_buckets_route_correctly(self):
        # Two images, one with 2 GTs (n2 bucket), one with 4 GTs (n3-5 bucket).
        preds = torch.tensor([
            [
                [0.5, 0.5, 0.9, 5.0, 0.0, 0.0],
                [0.3, 0.7, 0.9, 0.0, 5.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.1, 0.1, 0.9, 5.0, 0.0, 0.0],
                [0.3, 0.3, 0.9, 0.0, 5.0, 0.0],
                [0.5, 0.5, 0.9, 0.0, 0.0, 5.0],
                [0.7, 0.7, 0.9, 5.0, 0.0, 0.0],
            ],
        ])
        gt_centers = [
            torch.tensor([[0.5, 0.5], [0.3, 0.7]]),
            torch.tensor([[0.1, 0.1], [0.3, 0.3], [0.5, 0.5], [0.7, 0.7]]),
        ]
        gt_classes = [torch.tensor([0, 1]), torch.tensor([0, 1, 2, 0])]
        m = evaluate_multi_object(
            preds, gt_centers, image_size=(256, 256),
            gt_classes_list=gt_classes, has_classes=True,
        )
        assert m.by_count_n_gt["n2"] == 2
        assert m.by_count_n_gt["n3-5"] == 4
        # Both images had near-perfect predictions, so recall@4 in both
        # populated buckets should be 1.
        assert m.by_count_recall_at["n2"][4] == pytest.approx(1.0)
        assert m.by_count_recall_at["n3-5"][4] == pytest.approx(1.0)
