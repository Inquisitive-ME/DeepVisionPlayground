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

# Size buckets in pixels for "shape size" — matched against
# max(bbox_w, bbox_h). Anything larger than the last bucket falls into "huge".
DEFAULT_SIZE_BUCKETS: tuple[tuple[int, int], ...] = (
    (0, 30), (30, 60), (60, 120), (120, 1_000_000),
)
# Bucket label for each size bucket; aligned with DEFAULT_SIZE_BUCKETS.
DEFAULT_SIZE_BUCKET_NAMES: tuple[str, ...] = ("xs", "sm", "md", "lg")

# Bucket boundaries for "num GT objects in this image."
DEFAULT_COUNT_BUCKETS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 6), (6, 11), (11, 21), (21, 1_000_000),
)
DEFAULT_COUNT_BUCKET_NAMES: tuple[str, ...] = (
    "n0", "n1", "n2", "n3-5", "n6-10", "n11-20", "n21+",
)


def _which_bucket(value: float, buckets: tuple[tuple[int, int], ...]) -> int:
    """Return the index of the half-open bucket containing value, or len(buckets) if none."""
    for i, (lo, hi) in enumerate(buckets):
        if lo <= value < hi:
            return i
    return len(buckets)


@dataclass
class SingleObjectMetrics:
    mean_center_px: float = 0.0
    median_center_px: float = 0.0
    accuracy: float = 0.0
    pearson_cx: float = 0.0
    pearson_cy: float = 0.0
    n_images: int = 0

    def to_dict(self) -> dict[str, float]:
        return {
            "single/mean_center_px": self.mean_center_px,
            "single/median_center_px": self.median_center_px,
            "single/accuracy": self.accuracy,
            "single/pearson_cx": self.pearson_cx,
            "single/pearson_cy": self.pearson_cy,
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
    map_center: float = 0.0  # confidence-integrated center-AP ... see below
    confidence_threshold: float = 0.5  # cutoff used for the at-threshold P/R
    pearson_cx: float = 0.0  # over class-agnostic Hungarian matched pairs
    pearson_cy: float = 0.0
    # Per-class breakdowns. Keys are class indices (0..num_classes-1); the
    # outer dict for the *_at fields is keyed by pixel threshold.
    per_class_precision_at: dict[int, dict[int, float]] = field(default_factory=dict)
    per_class_recall_at: dict[int, dict[int, float]] = field(default_factory=dict)
    per_class_mean_center_px: dict[int, float] = field(default_factory=dict)
    per_class_n_gt: dict[int, int] = field(default_factory=dict)
    per_class_n_pred: dict[int, int] = field(default_factory=dict)
    class_names: tuple[str, ...] = ()
    # Per-size-bucket and per-num-objects-bucket breakdowns. Keys are
    # bucket names (e.g. "xs", "n3-5"); values are the per-bucket
    # mean center pixel error and recall at each pixel threshold over
    # only the GT objects in that bucket.
    by_size_mean_center_px: dict[str, float] = field(default_factory=dict)
    by_size_recall_at: dict[str, dict[int, float]] = field(default_factory=dict)
    by_size_n_gt: dict[str, int] = field(default_factory=dict)
    by_count_mean_center_px: dict[str, float] = field(default_factory=dict)
    by_count_recall_at: dict[str, dict[int, float]] = field(default_factory=dict)
    by_count_n_gt: dict[str, int] = field(default_factory=dict)
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
            "multi/confidence_threshold": self.confidence_threshold,
            "multi/pearson_cx": self.pearson_cx,
            "multi/pearson_cy": self.pearson_cy,
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
        for tag, d in self.by_size_mean_center_px.items():
            out[f"multi/by_size/{tag}/mean_center_px"] = d
        for tag, by_t in self.by_size_recall_at.items():
            for t, r in by_t.items():
                out[f"multi/by_size/{tag}/recall@{t}px"] = r
        for tag, n in self.by_size_n_gt.items():
            out[f"multi/by_size/{tag}/n_gt"] = float(n)
        for tag, d in self.by_count_mean_center_px.items():
            out[f"multi/by_count/{tag}/mean_center_px"] = d
        for tag, by_t in self.by_count_recall_at.items():
            for t, r in by_t.items():
                out[f"multi/by_count/{tag}/recall@{t}px"] = r
        for tag, n in self.by_count_n_gt.items():
            out[f"multi/by_count/{tag}/n_gt"] = float(n)
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


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient. Returns 0.0 when degenerate.

    A constant predictor gives std=0 and corrcoef=NaN; we report 0 then so
    TensorBoard scalars stay numeric.
    """
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


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

    Reports Pearson cx / cy alongside pixel distance because that's the
    metric that actually catches "model is regressing to the mean": a model
    that always predicts (0.5, 0.5) gives mean_center_px around 70 px on a
    256x256 canvas (close to the floor) AND pearson 0, where a partially
    converged model gives something like 50 px AND pearson 0.6+.
    """
    pred_px = _normalized_centers_to_pixels(preds[:, :2], image_size)
    gt_px = _normalized_centers_to_pixels(gt_centers, image_size)
    distances = np.linalg.norm(pred_px - gt_px, axis=-1)

    accuracy = 0.0
    if has_classes and gt_classes is not None and preds.shape[-1] > 2:
        pred_classes = preds[:, 2:].argmax(dim=-1).detach().cpu().numpy()
        gt_class_arr = gt_classes.detach().cpu().numpy()
        accuracy = float((pred_classes == gt_class_arr).mean())

    pearson_cx = _pearson(pred_px[:, 0], gt_px[:, 0])
    pearson_cy = _pearson(pred_px[:, 1], gt_px[:, 1])

    return SingleObjectMetrics(
        mean_center_px=float(distances.mean()) if distances.size else 0.0,
        median_center_px=float(np.median(distances)) if distances.size else 0.0,
        accuracy=accuracy,
        pearson_cx=pearson_cx,
        pearson_cy=pearson_cy,
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


def _voc_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """All-points (VOC2010+) average precision under a precision-recall curve.

    Recall is extended to 1.0 with precision 0, so undetected GT (recall that
    never reaches 1) is correctly penalized. Precision is made monotonically
    non-increasing before integrating over the recall steps.
    """
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _center_average_precision(
    ap_images: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    total_gt: int,
    pixel_thresholds: tuple[int, ...],
) -> float:
    """Confidence-integrated center-AP, averaged over pixel thresholds.

    For each pixel threshold, every prediction (across all images, at ALL
    confidences) is ranked by confidence and greedily matched to the nearest
    unmatched GT within the threshold; the resulting TP/FP sequence gives a
    PR curve whose area is the AP. Averaging the per-image-independent ranking
    over the whole val set and integrating over confidence makes the score
    independent of any single confidence cutoff — so models whose confidences
    live on different scales (e.g. focal heatmap peaks ~0.3 vs BCE slots ~1.0)
    are directly comparable.

    Each ``ap_images`` entry is ``(confidences, pred_centers_px, gt_centers_px)``
    for one image, using ALL predicted slots (not just thresholded ones).
    """
    if total_gt == 0:
        return 0.0
    aps: list[float] = []
    for t in pixel_thresholds:
        scored: list[tuple[float, int]] = []  # (confidence, is_tp)
        for confs, pred_px, gt_px in ap_images:
            n_pred = confs.shape[0]
            if n_pred == 0:
                continue
            n_gt = gt_px.shape[0] if gt_px.ndim == 2 else 0
            matched = np.zeros(n_gt, dtype=bool)
            # Stable sort so equal-confidence predictions are processed in a
            # deterministic order (e.g. the zero-confidence padding slots the
            # heatmap top-K decode emits, which would otherwise make the AP
            # depend on an arbitrary spatial tie-break).
            for i in np.argsort(-confs, kind="stable"):
                if n_gt == 0:
                    scored.append((float(confs[i]), 0))
                    continue
                dist = np.linalg.norm(pred_px[i] - gt_px, axis=-1)
                dist_avail = np.where(matched, np.inf, dist)
                j = int(np.argmin(dist_avail))
                if dist_avail[j] <= t:
                    matched[j] = True
                    scored.append((float(confs[i]), 1))
                else:
                    scored.append((float(confs[i]), 0))
        if not scored:
            aps.append(0.0)
            continue
        # Rank by confidence desc; within a confidence tie put false positives
        # before true positives (is_tp ascending) — the standard pessimistic
        # convention, so ties never optimistically inflate precision and the
        # AP is order-independent.
        scored.sort(key=lambda e: (-e[0], e[1]))
        tp = fp = 0
        recalls: list[float] = []
        precisions: list[float] = []
        for _conf, is_tp in scored:
            tp += is_tp
            fp += 1 - is_tp
            recalls.append(tp / total_gt)
            precisions.append(tp / (tp + fp))
        aps.append(_voc_ap(np.asarray(recalls), np.asarray(precisions)))
    return float(np.mean(aps)) if aps else 0.0


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
    gt_sizes_list: list[torch.Tensor] | None = None,
    size_buckets: tuple[tuple[int, int], ...] = DEFAULT_SIZE_BUCKETS,
    size_bucket_names: tuple[str, ...] = DEFAULT_SIZE_BUCKET_NAMES,
    count_buckets: tuple[tuple[int, int], ...] = DEFAULT_COUNT_BUCKETS,
    count_bucket_names: tuple[str, ...] = DEFAULT_COUNT_BUCKET_NAMES,
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
    # (pred, gt) pixel-coordinate pairs across all images, for Pearson cx/cy.
    pearson_pairs_x: list[tuple[float, float]] = []
    pearson_pairs_y: list[tuple[float, float]] = []
    n_gt_total = 0
    n_pred_total = 0
    # Per-image (confidences, pred_centers_px, gt_centers_px) over ALL slots,
    # used to compute the confidence-integrated AP (map_center) after the loop.
    ap_images: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
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

    # Per-bucket counters: shape size and num-objects-per-image.
    n_size_buckets = len(size_bucket_names)
    n_count_buckets = len(count_bucket_names)
    size_distances: list[list[float]] = [[] for _ in range(n_size_buckets)]
    size_tp_at: list[dict[int, int]] = [{t: 0 for t in pixel_thresholds} for _ in range(n_size_buckets)]
    size_n_gt_arr: list[int] = [0] * n_size_buckets
    count_distances: list[list[float]] = [[] for _ in range(n_count_buckets)]
    count_tp_at: list[dict[int, int]] = [{t: 0 for t in pixel_thresholds} for _ in range(n_count_buckets)]
    count_n_gt_arr: list[int] = [0] * n_count_buckets

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
        # Stash ALL slots (unfiltered by confidence) for the AP computation.
        ap_images.append((confs, pred_centers_px, gt_centers_px))
        n_gt_total += n_gt
        n_pred_total += n_pred
        cardinality_err.append(abs(n_pred - n_gt))

        # Bucket assignments for this image's GTs. size bucket is per-GT;
        # count bucket is per-image (everyone in this image shares it).
        gt_sizes_arr: np.ndarray | None = None
        if gt_sizes_list is not None and b < len(gt_sizes_list):
            gt_sizes_arr = gt_sizes_list[b].detach().cpu().numpy()
        size_bucket_per_gt = (
            [
                min(_which_bucket(float(gt_sizes_arr[i]), size_buckets), n_size_buckets - 1)
                for i in range(n_gt)
            ]
            if gt_sizes_arr is not None and n_gt > 0
            else [0] * n_gt
        )
        count_bucket = min(_which_bucket(n_gt, count_buckets), n_count_buckets - 1)
        for sb in size_bucket_per_gt:
            size_n_gt_arr[sb] += 1
        count_n_gt_arr[count_bucket] += n_gt

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

        matched_pred_xy = kept_pred_centers[pred_idx]
        matched_gt_xy = gt_centers_px[gt_idx]
        matched_dist = np.linalg.norm(matched_pred_xy - matched_gt_xy, axis=-1)
        distances.extend(matched_dist.tolist())
        matched_confs.extend(kept_pred_confs[pred_idx].tolist())
        total_matched += len(pred_idx)
        # Stash matched pairs so the loop can compute Pearson at the end
        # over all matched predictions, not per-batch (per-batch averages
        # of correlations don't combine cleanly).
        for px, gx in zip(matched_pred_xy[:, 0], matched_gt_xy[:, 0]):
            pearson_pairs_x.append((float(px), float(gx)))
        for py, gy in zip(matched_pred_xy[:, 1], matched_gt_xy[:, 1]):
            pearson_pairs_y.append((float(py), float(gy)))

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

        # Bucketed TP and distance pools for each matched GT.
        for k, gi in enumerate(gt_idx.tolist()):
            sb = size_bucket_per_gt[gi]
            d = float(matched_dist[k])
            size_distances[sb].append(d)
            count_distances[count_bucket].append(d)
            for t in pixel_thresholds:
                if d <= t:
                    size_tp_at[sb][t] += 1
                    count_tp_at[count_bucket][t] += 1

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
    # "Center mAP" here is a confidence-integrated average precision: for each
    # pixel threshold we rank every prediction by confidence, build a PR curve,
    # take its area (AP), and average over thresholds. Because it integrates
    # over the confidence axis, it does NOT depend on confidence_threshold, so
    # models whose confidence scores live on different scales (focal heatmap
    # peaks vs BCE slot confidences) are directly comparable. (Still center-
    # distance based, not IoU — not COCO mAP.)
    map_center = _center_average_precision(ap_images, n_gt_total, pixel_thresholds)

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

    by_size_mean_center_px: dict[str, float] = {}
    by_size_recall_at: dict[str, dict[int, float]] = {}
    by_size_n_gt: dict[str, int] = {}
    for i, name in enumerate(size_bucket_names):
        by_size_n_gt[name] = size_n_gt_arr[i]
        by_size_mean_center_px[name] = (
            float(np.mean(size_distances[i])) if size_distances[i] else 0.0
        )
        by_size_recall_at[name] = {
            t: (size_tp_at[i][t] / size_n_gt_arr[i]) if size_n_gt_arr[i] > 0 else 0.0
            for t in pixel_thresholds
        }

    by_count_mean_center_px: dict[str, float] = {}
    by_count_recall_at: dict[str, dict[int, float]] = {}
    by_count_n_gt: dict[str, int] = {}
    for i, name in enumerate(count_bucket_names):
        by_count_n_gt[name] = count_n_gt_arr[i]
        by_count_mean_center_px[name] = (
            float(np.mean(count_distances[i])) if count_distances[i] else 0.0
        )
        by_count_recall_at[name] = {
            t: (count_tp_at[i][t] / count_n_gt_arr[i]) if count_n_gt_arr[i] > 0 else 0.0
            for t in pixel_thresholds
        }

    if pearson_pairs_x:
        px_arr = np.array([p[0] for p in pearson_pairs_x])
        gx_arr = np.array([p[1] for p in pearson_pairs_x])
        py_arr = np.array([p[0] for p in pearson_pairs_y])
        gy_arr = np.array([p[1] for p in pearson_pairs_y])
        pearson_cx = _pearson(px_arr, gx_arr)
        pearson_cy = _pearson(py_arr, gy_arr)
    else:
        pearson_cx = pearson_cy = 0.0

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
        confidence_threshold=confidence_threshold,
        pearson_cx=pearson_cx,
        pearson_cy=pearson_cy,
        per_class_precision_at=per_class_precision_at,
        per_class_recall_at=per_class_recall_at,
        per_class_mean_center_px=per_class_mean_center_px,
        per_class_n_gt=pc_n_gt,
        per_class_n_pred=pc_n_pred,
        class_names=class_names,
        by_size_mean_center_px=by_size_mean_center_px,
        by_size_recall_at=by_size_recall_at,
        by_size_n_gt=by_size_n_gt,
        by_count_mean_center_px=by_count_mean_center_px,
        by_count_recall_at=by_count_recall_at,
        by_count_n_gt=by_count_n_gt,
        n_images=batch_size,
        n_gt=n_gt_total,
        n_pred=n_pred_total,
    )
