"""Evaluation metrics for center-based shape detectors.

Two model families are supported:

- Single-object: prediction tensor is shape (B, 2 [+ num_classes]), GT is one
  center per image. Reported metrics are pixel-distance and classification
  accuracy.

- Multi-object: prediction tensor is shape (B, max_objects, 3 [+ num_classes])
  with sigmoid'd confidence in slot 2. GT is a list[Tensor] of variable
  length. Predictions above ``confidence_threshold`` are matched to GT with
  the Hungarian algorithm; reported metrics are mean center distance,
  classification accuracy on matched pairs, cardinality error, and a center-
  mAP-style score over a sweep of pixel thresholds.

All metrics are reported in PIXEL space, given the dataset's ``image_size``.
This keeps numbers interpretable across image resolutions ("how many pixels
off was the center?") and matches the stated benchmarking goal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

DEFAULT_PIXEL_THRESHOLDS: tuple[int, ...] = (2, 4, 8, 16)


@dataclass
class SingleObjectMetrics:
    mean_center_px: float = 0.0
    median_center_px: float = 0.0
    accuracy: float = 0.0
    n_images: int = 0

    def to_dict(self) -> dict[str, float]:
        return {
            "single/mean_center_px": self.mean_center_px,
            "single/median_center_px": self.median_center_px,
            "single/accuracy": self.accuracy,
        }


@dataclass
class MultiObjectMetrics:
    mean_matched_center_px: float = 0.0
    median_matched_center_px: float = 0.0
    matched_class_accuracy: float = 0.0
    cardinality_error: float = 0.0
    mean_conf_matched: float = 0.0
    mean_conf_unmatched: float = 0.0
    precision_at: dict[int, float] = field(default_factory=dict)
    recall_at: dict[int, float] = field(default_factory=dict)
    map_center: float = 0.0  # average over thresholds of (P*R) ... see below
    # Per-class breakdowns. Keys are class indices (0..num_classes-1); the
    # outer dict for the *_at fields is keyed by pixel threshold.
    per_class_precision_at: dict[int, dict[int, float]] = field(default_factory=dict)
    per_class_recall_at: dict[int, dict[int, float]] = field(default_factory=dict)
    per_class_mean_center_px: dict[int, float] = field(default_factory=dict)
    per_class_n_gt: dict[int, int] = field(default_factory=dict)
    per_class_n_pred: dict[int, int] = field(default_factory=dict)
    class_names: tuple[str, ...] = ()
    n_images: int = 0
    n_gt: int = 0
    n_pred: int = 0

    def _class_tag(self, class_idx: int) -> str:
        if self.class_names and 0 <= class_idx < len(self.class_names):
            return self.class_names[class_idx]
        return f"class_{class_idx}"

    def to_dict(self) -> dict[str, float]:
        out: dict[str, float] = {
            "multi/mean_matched_center_px": self.mean_matched_center_px,
            "multi/median_matched_center_px": self.median_matched_center_px,
            "multi/matched_class_accuracy": self.matched_class_accuracy,
            "multi/cardinality_error": self.cardinality_error,
            "multi/mean_conf_matched": self.mean_conf_matched,
            "multi/mean_conf_unmatched": self.mean_conf_unmatched,
            "multi/map_center": self.map_center,
        }
        for t, p in self.precision_at.items():
            out[f"multi/precision@{t}px"] = p
        for t, r in self.recall_at.items():
            out[f"multi/recall@{t}px"] = r
        for c, by_t in self.per_class_precision_at.items():
            tag = self._class_tag(c)
            for t, p in by_t.items():
                out[f"multi/{tag}/precision@{t}px"] = p
        for c, by_t in self.per_class_recall_at.items():
            tag = self._class_tag(c)
            for t, r in by_t.items():
                out[f"multi/{tag}/recall@{t}px"] = r
        for c, d in self.per_class_mean_center_px.items():
            tag = self._class_tag(c)
            out[f"multi/{tag}/mean_center_px"] = d
        return out


def _normalized_centers_to_pixels(
    centers: torch.Tensor | np.ndarray, image_size: tuple[int, int]
) -> np.ndarray:
    """Convert (..., 2) normalized centers to pixel coordinates."""
    arr = centers.detach().cpu().numpy() if isinstance(centers, torch.Tensor) else np.asarray(centers)
    w, h = image_size
    scaled = arr.copy().astype(np.float64)
    scaled[..., 0] *= w
    scaled[..., 1] *= h
    return scaled


def evaluate_single_object(
    preds: torch.Tensor,
    gt_centers: torch.Tensor,
    image_size: tuple[int, int],
    gt_classes: torch.Tensor | None = None,
    has_classes: bool = False,
) -> SingleObjectMetrics:
    """Evaluate a single-object center predictor on one batch.

    Args:
        preds: (B, 2 [+ num_classes]) — sigmoided centers + class logits.
        gt_centers: (B, 2) — normalized GT centers.
        gt_classes: (B,) long — GT class indices (only used if has_classes).
    """
    pred_px = _normalized_centers_to_pixels(preds[:, :2], image_size)
    gt_px = _normalized_centers_to_pixels(gt_centers, image_size)
    distances = np.linalg.norm(pred_px - gt_px, axis=-1)

    accuracy = 0.0
    if has_classes and gt_classes is not None and preds.shape[-1] > 2:
        pred_classes = preds[:, 2:].argmax(dim=-1).detach().cpu().numpy()
        gt_class_arr = gt_classes.detach().cpu().numpy()
        accuracy = float((pred_classes == gt_class_arr).mean())

    return SingleObjectMetrics(
        mean_center_px=float(distances.mean()) if distances.size else 0.0,
        median_center_px=float(np.median(distances)) if distances.size else 0.0,
        accuracy=accuracy,
        n_images=int(distances.size),
    )


def _hungarian_match_pixels(
    pred_centers_px: np.ndarray,
    gt_centers_px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise Euclidean cost in pixels, Hungarian assignment."""
    if pred_centers_px.size == 0 or gt_centers_px.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    cost = np.linalg.norm(
        pred_centers_px[:, None, :] - gt_centers_px[None, :, :], axis=-1
    )
    pred_idx, gt_idx = linear_sum_assignment(cost)
    return pred_idx, gt_idx


def evaluate_multi_object(
    preds: torch.Tensor,
    gt_centers_list: list[torch.Tensor],
    image_size: tuple[int, int],
    gt_classes_list: list[torch.Tensor] | None = None,
    has_classes: bool = False,
    confidence_threshold: float = 0.5,
    pixel_thresholds: tuple[int, ...] = DEFAULT_PIXEL_THRESHOLDS,
    num_classes: int | None = None,
    class_names: tuple[str, ...] = (),
) -> MultiObjectMetrics:
    """Evaluate a multi-object center predictor on one batch.

    Args:
        preds: (B, max_objects, 3 [+ num_classes]); slot 2 is sigmoid'd conf.
        gt_centers_list: per-image list of (num_objects_i, 2) normalized centers.
        gt_classes_list: per-image list of (num_objects_i,) long class indices.
        num_classes: number of distinct classes; required for per-class
            counters when has_classes=True. Inferred from preds[..., 3:]
            shape if not given.
        class_names: optional human-readable names; used only for log tags.

    Per-class counters (when has_classes):
        For each class c, with class-aware accounting on the SAME Hungarian
        match used for the global metric:
          - TP[c]: pred_class == c == gt_class AND distance <= threshold
          - FP[c]: kept-pred with pred_class == c that does NOT contribute
                   to a TP[c] (class mismatch, distance miss, or unmatched)
          - FN[c]: GT with gt_class == c not matched by a same-class pred
                   within threshold
        Per-class mean_center_px is computed over class-correct matched pairs
        (so it answers "when we got the class right, how close was it?").
    """
    batch_size = preds.shape[0]
    image_diag = math.hypot(image_size[0], image_size[1])
    if num_classes is None and has_classes and preds.shape[-1] > 3:
        num_classes = int(preds.shape[-1] - 3)

    distances: list[float] = []
    correct_class = 0
    total_matched = 0
    matched_confs: list[float] = []
    unmatched_confs: list[float] = []
    cardinality_err: list[int] = []
    n_gt_total = 0
    n_pred_total = 0
    # Global per-threshold counters.
    tp_at = {t: 0 for t in pixel_thresholds}
    fp_at = {t: 0 for t in pixel_thresholds}
    fn_at = {t: 0 for t in pixel_thresholds}
    # Per-class per-threshold counters (only populated when has_classes).
    classes = list(range(num_classes)) if (has_classes and num_classes) else []
    pc_tp_at: dict[int, dict[int, int]] = {c: {t: 0 for t in pixel_thresholds} for c in classes}
    pc_fp_at: dict[int, dict[int, int]] = {c: {t: 0 for t in pixel_thresholds} for c in classes}
    pc_fn_at: dict[int, dict[int, int]] = {c: {t: 0 for t in pixel_thresholds} for c in classes}
    pc_distances: dict[int, list[float]] = {c: [] for c in classes}
    pc_n_gt: dict[int, int] = {c: 0 for c in classes}
    pc_n_pred: dict[int, int] = {c: 0 for c in classes}

    for b in range(batch_size):
        confs = preds[b, :, 2].detach().cpu().numpy()
        keep = confs > confidence_threshold
        pred_centers_px = _normalized_centers_to_pixels(preds[b, :, :2], image_size)
        kept_pred_centers = pred_centers_px[keep]
        kept_pred_confs = confs[keep]
        kept_pred_classes: np.ndarray | None = None
        if has_classes and preds.shape[-1] > 3:
            kept_pred_classes = (
                preds[b, :, 3:].argmax(dim=-1).detach().cpu().numpy()[keep]
            )

        gt_centers = gt_centers_list[b]
        gt_centers_px = _normalized_centers_to_pixels(gt_centers, image_size)
        n_gt = gt_centers_px.shape[0] if gt_centers_px.ndim == 2 else 0
        n_pred = int(keep.sum())
        n_gt_total += n_gt
        n_pred_total += n_pred
        cardinality_err.append(abs(n_pred - n_gt))

        unmatched_confs.extend(confs[~keep].tolist())

        gt_classes_arr: np.ndarray | None = None
        if has_classes and gt_classes_list is not None:
            gt_classes_arr = gt_classes_list[b].detach().cpu().numpy()
            for c in classes:
                pc_n_gt[c] += int((gt_classes_arr == c).sum())
            if kept_pred_classes is not None:
                for c in classes:
                    pc_n_pred[c] += int((kept_pred_classes == c).sum())

        if n_pred == 0 or n_gt == 0:
            for t in pixel_thresholds:
                fp_at[t] += n_pred
                fn_at[t] += n_gt
            # Unmatched preds contribute to FP[pred_class] for all thresholds;
            # unmatched GTs contribute to FN[gt_class].
            if kept_pred_classes is not None:
                for c in classes:
                    cnt = int((kept_pred_classes == c).sum())
                    if cnt > 0:
                        for t in pixel_thresholds:
                            pc_fp_at[c][t] += cnt
            if gt_classes_arr is not None:
                for c in classes:
                    cnt = int((gt_classes_arr == c).sum())
                    if cnt > 0:
                        for t in pixel_thresholds:
                            pc_fn_at[c][t] += cnt
            continue

        pred_idx, gt_idx = _hungarian_match_pixels(kept_pred_centers, gt_centers_px)
        if pred_idx.size == 0:
            for t in pixel_thresholds:
                fp_at[t] += n_pred
                fn_at[t] += n_gt
            continue

        matched_dist = np.linalg.norm(
            kept_pred_centers[pred_idx] - gt_centers_px[gt_idx], axis=-1
        )
        distances.extend(matched_dist.tolist())
        matched_confs.extend(kept_pred_confs[pred_idx].tolist())
        total_matched += len(pred_idx)

        if has_classes and kept_pred_classes is not None and gt_classes_arr is not None:
            correct_class += int(
                (kept_pred_classes[pred_idx] == gt_classes_arr[gt_idx]).sum()
            )

        # Global TP/FP/FN by threshold.
        for t in pixel_thresholds:
            tp = int((matched_dist <= t).sum())
            tp_at[t] += tp
            fp_at[t] += n_pred - tp
            fn_at[t] += n_gt - tp

        # Per-class TP/FP/FN by threshold. A match counts as TP[c] only when
        # both classes equal c AND distance <= threshold; otherwise the
        # predicted slot is FP[pred_class] and the GT slot is FN[gt_class].
        if classes and kept_pred_classes is not None and gt_classes_arr is not None:
            matched_pred_classes = kept_pred_classes[pred_idx]
            matched_gt_classes = gt_classes_arr[gt_idx]
            class_correct = matched_pred_classes == matched_gt_classes

            # Track which preds and GTs got matched at all.
            matched_pred_mask = np.zeros(n_pred, dtype=bool)
            matched_pred_mask[pred_idx] = True
            matched_gt_mask = np.zeros(n_gt, dtype=bool)
            matched_gt_mask[gt_idx] = True

            # Class-correct distance pool, used for per_class_mean_center_px.
            for d, pc in zip(matched_dist[class_correct], matched_pred_classes[class_correct]):
                if pc in pc_distances:
                    pc_distances[int(pc)].append(float(d))

            for t in pixel_thresholds:
                # TP[c]: class-correct match within threshold.
                tp_mask = class_correct & (matched_dist <= t)
                for pc in matched_pred_classes[tp_mask]:
                    pc_tp_at[int(pc)][t] += 1

                # Among matched preds: anything that didn't TP at this t
                # contributes to FP for its own predicted class.
                non_tp_matched_preds = matched_pred_classes[~tp_mask]
                for pc in non_tp_matched_preds:
                    pc_fp_at[int(pc)][t] += 1
                # Unmatched preds contribute FP for their predicted class at every t.
                for pc in kept_pred_classes[~matched_pred_mask]:
                    pc_fp_at[int(pc)][t] += 1

                # Among matched GTs: anything that didn't TP at this t
                # contributes to FN for its own GT class.
                non_tp_matched_gts = matched_gt_classes[~tp_mask]
                for gc in non_tp_matched_gts:
                    pc_fn_at[int(gc)][t] += 1
                # Unmatched GTs contribute FN for their GT class.
                for gc in gt_classes_arr[~matched_gt_mask]:
                    pc_fn_at[int(gc)][t] += 1

    precision_at = {
        t: (tp_at[t] / (tp_at[t] + fp_at[t])) if (tp_at[t] + fp_at[t]) > 0 else 0.0
        for t in pixel_thresholds
    }
    recall_at = {
        t: (tp_at[t] / (tp_at[t] + fn_at[t])) if (tp_at[t] + fn_at[t]) > 0 else 0.0
        for t in pixel_thresholds
    }
    # "Center mAP" here is the average of P*R across thresholds — a single
    # scalar that goes up only when both precision and recall improve. (This
    # is not COCO mAP; we don't have IoUs without bboxes. Rename if/when we
    # add bbox heads.)
    map_center = (
        float(np.mean([precision_at[t] * recall_at[t] for t in pixel_thresholds]))
        if pixel_thresholds else 0.0
    )

    per_class_precision_at: dict[int, dict[int, float]] = {}
    per_class_recall_at: dict[int, dict[int, float]] = {}
    per_class_mean_center_px: dict[int, float] = {}
    for c in classes:
        per_class_precision_at[c] = {
            t: (pc_tp_at[c][t] / (pc_tp_at[c][t] + pc_fp_at[c][t]))
            if (pc_tp_at[c][t] + pc_fp_at[c][t]) > 0 else 0.0
            for t in pixel_thresholds
        }
        per_class_recall_at[c] = {
            t: (pc_tp_at[c][t] / (pc_tp_at[c][t] + pc_fn_at[c][t]))
            if (pc_tp_at[c][t] + pc_fn_at[c][t]) > 0 else 0.0
            for t in pixel_thresholds
        }
        per_class_mean_center_px[c] = (
            float(np.mean(pc_distances[c])) if pc_distances[c] else 0.0
        )

    _ = image_diag  # reserved for future per-image diag normalization

    return MultiObjectMetrics(
        mean_matched_center_px=float(np.mean(distances)) if distances else 0.0,
        median_matched_center_px=float(np.median(distances)) if distances else 0.0,
        matched_class_accuracy=(correct_class / total_matched) if total_matched > 0 else 0.0,
        cardinality_error=float(np.mean(cardinality_err)) if cardinality_err else 0.0,
        mean_conf_matched=float(np.mean(matched_confs)) if matched_confs else 0.0,
        mean_conf_unmatched=float(np.mean(unmatched_confs)) if unmatched_confs else 0.0,
        precision_at=precision_at,
        recall_at=recall_at,
        map_center=map_center,
        per_class_precision_at=per_class_precision_at,
        per_class_recall_at=per_class_recall_at,
        per_class_mean_center_px=per_class_mean_center_px,
        per_class_n_gt=pc_n_gt,
        per_class_n_pred=pc_n_pred,
        class_names=class_names,
        n_images=batch_size,
        n_gt=n_gt_total,
        n_pred=n_pred_total,
    )
