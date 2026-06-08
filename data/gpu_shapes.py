"""GPU-side synthetic shape generator.

The CPU dataset (``synthetic_shapes_dataset.ShapeDataset``) renders shapes
through PIL one image at a time, which is exactly the bottleneck that pinned
the 3090 at ~10% utilization in our earlier audit. This module replaces the
PIL pipeline with batched torch ops that run on the GPU:

- Sample every numeric parameter (positions, sizes, colors, classes, optional
  rotation angles) on the CPU with cheap Python / NumPy.
- Push those parameters to the GPU as small tensors.
- Fill backgrounds, then rasterize ellipses (axis-aligned) and polygons
  (rectangles + triangles, optionally rotated) with vectorized point-in-shape
  masks.
- Return ``(images, annotations)`` matching the shape that
  ``ShapeDataset.collate_function`` produces, so training scripts only need
  to swap their loader.

Scope (intentional, kept tight to start):

- Solid backgrounds only. Texture / noise backgrounds and add_noise=True are
  not supported yet.
- Filled shapes only — no outlines.
- Ellipses are axis-aligned even when ``rotate_shapes`` is True (matches the
  CPU dataset).
- Rectangles and triangles can be rotated; rotation is computed on the CPU
  before rendering so the GPU kernel doesn't need to know. Like the CPU
  dataset, rotations are retried to stay in-bounds and ``max_overlap`` is
  enforced, so the two paths produce comparable distributions.

Anything outside that scope, fall back to the CPU dataset for now.

Known CPU/GPU rendering differences (small, but present — keep them in mind
when comparing ``--gpu-data`` runs against CPU runs):

- Circles: this renderer fills the exact analytic disc, which is a few percent
  SMALLER than PIL's inclusive-bbox ``draw.ellipse`` (largest at small sizes,
  ~10% area at the 20-px floor, <3% by 90 px). Rectangles/triangles match PIL.
- Batch count: ``len(loader) == ceil(num_images / batch_size)`` and every batch
  is full, so an epoch emits up to ``batch_size - 1`` more images than
  ``num_images`` (the CPU DataLoader instead emits exactly ``num_images`` with a
  smaller final batch). Throughput is reported off the actual emitted count.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from data.annotations import BoundingBox, ShapeType, validate_shape_size_range


@dataclass
class _ShapeBatchPlan:
    """All the per-shape parameters for a batch, ready to push to GPU.

    Each field is shape (B, N_max, ...) with ``valid`` masking out unused
    slots; that lets every batch be a fixed-size tensor regardless of how
    many shapes individual images happened to want.
    """
    bg_colors: torch.Tensor          # (B, 3) uint8 0..255
    valid: torch.Tensor              # (B, N_max) bool
    is_ellipse: torch.Tensor         # (B, N_max) bool
    poly_vertices: torch.Tensor      # (B, N_max, 4, 2) float — pad triangles to 4 verts via repeat
    poly_v_count: torch.Tensor       # (B, N_max) long — number of real vertices (3 or 4)
    ellipse_center: torch.Tensor     # (B, N_max, 2) float
    ellipse_radii: torch.Tensor      # (B, N_max, 2) float
    colors: torch.Tensor             # (B, N_max, 3) uint8
    annotations: list[list[dict]]    # CPU side annotation dicts


def _sample_color_avoiding(
    rng: random.Random,
    bg: tuple[int, int, int],
    threshold: float = 50.0,
) -> tuple[int, int, int]:
    for _ in range(10):
        c = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        if math.hypot(*[a - b for a, b in zip(c, bg)]) > threshold:
            return c
    return c


def _rotate_points(points: list[tuple[float, float]],
                   center: tuple[float, float],
                   angle_rad: float) -> list[tuple[float, float]]:
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    out = []
    for x, y in points:
        dx, dy = x - center[0], y - center[1]
        out.append((center[0] + dx * cos_a - dy * sin_a,
                    center[1] + dx * sin_a + dy * cos_a))
    return out


def _rotate_in_bounds(points: list[tuple[float, float]],
                      center: tuple[float, float],
                      w: int, h: int,
                      rng: random.Random,
                      max_attempts: int = 10) -> list[tuple[float, float]]:
    """Rotate ``points`` about ``center`` by a random angle that keeps the
    rotated shape inside [0, w] x [0, h].

    Mirrors ``ShapeDataset.rotate_shape_points`` on the CPU path: try a few
    angles with a shrinking range and accept the first whose bounding box is
    fully in-bounds, otherwise fall back to the unrotated shape. Without this
    the GPU path produced ~25% off-canvas (clipped) rotated shapes, so the two
    pipelines were not comparable for rotation studies.
    """
    max_angle = 2 * math.pi
    for _ in range(max_attempts):
        angle = rng.uniform(0, max_angle)
        rotated = _rotate_points(points, center, angle)
        xs, ys = zip(*rotated)
        if min(xs) >= 0 and min(ys) >= 0 and max(xs) <= w and max(ys) <= h:
            return rotated
        max_angle *= 0.25
    return points


def _bbox_overlap_ratio(a: tuple[int, int, int, int],
                        b: tuple[int, int, int, int]) -> float:
    """Intersection area over the smaller box area (same measure the CPU
    dataset's ``compute_overlap_ratio`` uses). Boxes are (x0, y0, x1, y1)."""
    x_left = max(a[0], b[0])
    y_top = max(a[1], b[1])
    x_right = min(a[2], b[2])
    y_bottom = min(a[3], b[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    inter = (x_right - x_left) * (y_bottom - y_top)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    min_area = min(area_a, area_b)
    return inter / min_area if min_area > 0 else 0.0


def _is_overlapping(box: tuple[int, int, int, int],
                    existing: list[tuple[int, int, int, int]],
                    max_overlap: float) -> bool:
    return any(_bbox_overlap_ratio(box, e) > max_overlap for e in existing)


def plan_batch(
    *,
    batch_size: int,
    image_size: tuple[int, int],
    num_shapes_range: tuple[int, int],
    shape_size_range: tuple[int, int],
    shape_types: tuple[ShapeType, ...],
    rotate_shapes: bool,
    rng: random.Random,
    max_overlap: float = 0.6,
) -> _ShapeBatchPlan:
    """Sample annotations on CPU and pack into fixed-size tensors.

    Tensors live on CPU here; ``GpuShapeLoader`` moves them to GPU before
    rasterization. ``max_overlap`` rejects placements whose intersection-over-
    min-area exceeds the threshold (matching the CPU dataset), so the GPU path
    produces the same shape-density distribution.
    """
    n_max = max(num_shapes_range[1], 1)
    w, h = image_size

    bg_colors = np.zeros((batch_size, 3), dtype=np.uint8)
    valid = np.zeros((batch_size, n_max), dtype=bool)
    is_ellipse = np.zeros((batch_size, n_max), dtype=bool)
    poly_vertices = np.zeros((batch_size, n_max, 4, 2), dtype=np.float32)
    poly_v_count = np.zeros((batch_size, n_max), dtype=np.int64)
    ellipse_center = np.zeros((batch_size, n_max, 2), dtype=np.float32)
    ellipse_radii = np.zeros((batch_size, n_max, 2), dtype=np.float32)
    colors = np.zeros((batch_size, n_max, 3), dtype=np.uint8)
    annotations: list[list[dict]] = []

    for b in range(batch_size):
        bg = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        bg_colors[b] = bg

        n = rng.randint(*num_shapes_range)
        anns: list[dict] = []
        # Boxes placed so far in this image, for overlap rejection. Slots fill
        # contiguously; a shape that can't find a non-overlapping spot in 10
        # tries is dropped (matching the CPU dataset), so an image may end up
        # with fewer than n shapes.
        existing_boxes: list[tuple[int, int, int, int]] = []
        slot = 0
        for _ in range(n):
            shape = rng.choice(shape_types)
            color = _sample_color_avoiding(rng, bg)
            min_size, max_size = shape_size_range
            sw = rng.randint(min_size, min(max_size, w // 2))
            sh = rng.randint(min_size, min(max_size, h // 2))

            placed_box: tuple[int, int, int, int] | None = None
            for _try in range(10):
                x0 = rng.randint(0, w - sw)
                y0 = rng.randint(0, h - sh)
                cand = (x0, y0, x0 + sw, y0 + sh)
                if not _is_overlapping(cand, existing_boxes, max_overlap):
                    placed_box = cand
                    existing_boxes.append(cand)
                    break
            if placed_box is None:
                continue
            x0, y0, x1, y1 = placed_box
            cx_px = (x0 + x1) / 2
            cy_px = (y0 + y1) / 2

            i = slot
            valid[b, i] = True
            colors[b, i] = color

            if shape is ShapeType.CIRCLE:
                is_ellipse[b, i] = True
                ellipse_center[b, i] = (cx_px, cy_px)
                ellipse_radii[b, i] = ((x1 - x0) / 2, (y1 - y0) / 2)
                bbox = BoundingBox(x0, y0, x1, y1)
            elif shape is ShapeType.RECTANGLE:
                corners: list[tuple[float, float]] = [
                    (float(x0), float(y0)), (float(x1), float(y0)),
                    (float(x1), float(y1)), (float(x0), float(y1)),
                ]
                if rotate_shapes:
                    corners = _rotate_in_bounds(corners, (cx_px, cy_px), w, h, rng)
                xs, ys = zip(*corners)
                bbox = BoundingBox(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                poly_vertices[b, i] = corners
                poly_v_count[b, i] = 4
            elif shape is ShapeType.TRIANGLE:
                p1 = (float(x0 + rng.randint(0, sw)), float(y0))
                p2 = (float(x0), float(y1))
                p3 = (float(x1), float(y1))
                centroid = ((p1[0] + p2[0] + p3[0]) / 3.0, (p1[1] + p2[1] + p3[1]) / 3.0)
                pts: list[tuple[float, float]] = [p1, p2, p3]
                if rotate_shapes:
                    pts = _rotate_in_bounds(pts, centroid, w, h, rng)
                xs, ys = zip(*pts)
                bbox = BoundingBox(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                # Pad to 4 vertices by repeating the last point so the polygon
                # rasterizer always sees a fixed-size buffer; the redundant
                # edge has zero length and contributes nothing.
                poly_vertices[b, i, :3] = pts
                poly_vertices[b, i, 3] = pts[2]
                poly_v_count[b, i] = 3
                cx_px = centroid[0]
                cy_px = centroid[1]
            else:
                raise ValueError(f"unsupported shape: {shape}")

            anns.append({
                "shape": shape.value,
                "bbox": bbox,
                "center": (cx_px / w, cy_px / h),
                "color": color,
            })
            slot += 1
        annotations.append(anns)

    return _ShapeBatchPlan(
        bg_colors=torch.from_numpy(bg_colors),
        valid=torch.from_numpy(valid),
        is_ellipse=torch.from_numpy(is_ellipse),
        poly_vertices=torch.from_numpy(poly_vertices),
        poly_v_count=torch.from_numpy(poly_v_count),
        ellipse_center=torch.from_numpy(ellipse_center),
        ellipse_radii=torch.from_numpy(ellipse_radii),
        colors=torch.from_numpy(colors),
        annotations=annotations,
    )


def _pixel_grid(image_size: tuple[int, int], device: torch.device) -> torch.Tensor:
    """Returns (H, W, 2) tensor with pixel coordinates as floats [0, W-1]."""
    w, h = image_size
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack([xx, yy], dim=-1)  # (H, W, 2)


def render_batch(
    plan: _ShapeBatchPlan,
    image_size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Rasterize a planned batch on ``device``. Returns (B, 3, H, W) float32 in [0, 1]."""
    plan_d = _ShapeBatchPlan(
        bg_colors=plan.bg_colors.to(device),
        valid=plan.valid.to(device),
        is_ellipse=plan.is_ellipse.to(device),
        poly_vertices=plan.poly_vertices.to(device),
        poly_v_count=plan.poly_v_count.to(device),
        ellipse_center=plan.ellipse_center.to(device),
        ellipse_radii=plan.ellipse_radii.to(device),
        colors=plan.colors.to(device),
        annotations=plan.annotations,
    )

    w, h = image_size
    B, N_max = plan_d.valid.shape
    # Initialize canvas with solid background colors.
    canvas = plan_d.bg_colors[:, :, None, None].expand(B, 3, h, w).contiguous().float() / 255.0

    grid = _pixel_grid(image_size, device)  # (H, W, 2)

    # Compose shape by shape; later shapes paint over earlier ones, matching
    # the CPU dataset's draw order. N_max is small (typically 1-3), so this
    # Python loop is cheap.
    for n in range(N_max):
        slot_valid = plan_d.valid[:, n]                # (B,)
        if not slot_valid.any():
            continue
        is_ell = plan_d.is_ellipse[:, n]               # (B,)
        color = plan_d.colors[:, n].float() / 255.0    # (B, 3)

        # Ellipse mask: ((x - cx) / rx)^2 + ((y - cy) / ry)^2 <= 1
        cx = plan_d.ellipse_center[:, n, 0:1, None].unsqueeze(-1)  # (B, 1, 1, 1)
        cy = plan_d.ellipse_center[:, n, 1:2, None].unsqueeze(-1)
        rx = plan_d.ellipse_radii[:, n, 0:1, None].unsqueeze(-1).clamp_min(1e-3)
        ry = plan_d.ellipse_radii[:, n, 1:2, None].unsqueeze(-1).clamp_min(1e-3)
        gx = grid[..., 0].unsqueeze(0).unsqueeze(0)    # (1, 1, H, W)
        gy = grid[..., 1].unsqueeze(0).unsqueeze(0)
        ell_dist = ((gx - cx) / rx) ** 2 + ((gy - cy) / ry) ** 2
        ell_mask = ell_dist.squeeze(1) <= 1.0          # (B, H, W)

        # Polygon mask via signed-cross-product test on each edge. Vertex
        # buffer is always 4 long; for triangles we duplicated the last
        # vertex so the dummy edge has zero cross and doesn't break the
        # all-same-sign test.
        verts = plan_d.poly_vertices[:, n]             # (B, 4, 2)
        v_next = torch.roll(verts, shifts=-1, dims=1)  # (B, 4, 2)
        edge = v_next - verts                          # (B, 4, 2)
        # diff: pixel relative to each vertex -> (B, 4, H, W, 2)
        diff = grid[None, None] - verts[:, :, None, None, :]
        cross = edge[:, :, None, None, 0] * diff[..., 1] - edge[:, :, None, None, 1] * diff[..., 0]
        # all >= 0 OR all <= 0 means inside (CCW or CW polygon)
        all_pos = (cross >= 0).all(dim=1)
        all_neg = (cross <= 0).all(dim=1)
        poly_mask = all_pos | all_neg                  # (B, H, W)

        # Pick mask per image based on whether this slot is an ellipse.
        chosen = torch.where(is_ell[:, None, None], ell_mask, poly_mask)
        chosen = chosen & slot_valid[:, None, None]

        # Composite: where chosen, replace with this slot's color.
        canvas = torch.where(
            chosen[:, None],          # (B, 1, H, W)
            color[:, :, None, None],  # (B, 3, 1, 1)
            canvas,
        )

    return canvas


class GpuShapeLoader(Iterable):
    """Yields (images, annotations) batches generated entirely on the GPU.

    Drop-in replacement for ``torch.utils.data.DataLoader`` for the subset
    of options ``ShapeDataset`` exposes — call as ``for imgs, anns in loader:``
    and the train script keeps working.

    ``len(loader) == num_images // batch_size`` (no partial last batch).
    """

    def __init__(
        self,
        *,
        batch_size: int,
        num_images: int,
        image_size: tuple[int, int] = (256, 256),
        num_shapes_range: tuple[int, int] = (0, 3),  # matches ShapeDataset's default
        shape_size_range: tuple[int, int] = (20, 90),
        shape_types: tuple[ShapeType, ...] = tuple(ShapeType),
        rotate_shapes: bool = False,
        max_overlap: float = 0.6,
        device: torch.device | str = "cuda",
        seed: int | None = None,
        reseed_each_epoch: bool = False,
    ):
        if num_shapes_range[0] < 0 or num_shapes_range[1] < num_shapes_range[0]:
            raise ValueError(f"bad num_shapes_range={num_shapes_range}")
        if reseed_each_epoch and seed is None:
            # A fixed val set needs a fixed seed; without one the reseed would
            # silently degrade to a different val set every epoch.
            raise ValueError("reseed_each_epoch=True requires a seed")
        validate_shape_size_range(image_size, shape_size_range)
        self.batch_size = batch_size
        self.num_images = num_images
        self.image_size = image_size
        self.num_shapes_range = num_shapes_range
        self.shape_size_range = shape_size_range
        self.shape_types = shape_types
        self.rotate_shapes = rotate_shapes
        self.max_overlap = max_overlap
        self.device = torch.device(device)
        # Keep the seed so a val loader can re-derive the SAME dataset every
        # epoch. Training loaders leave reseed_each_epoch=False so they keep
        # drawing fresh data per epoch (the intended "infinite data" behavior).
        self._seed = seed
        self.reseed_each_epoch = reseed_each_epoch
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        # Round up so num_images < batch_size still yields one full batch.
        # The actual emitted image count per epoch is len(self) * batch_size,
        # which can slightly exceed num_images. That's fine for training and
        # the train script reports throughput off the actual counts.
        return max(1, math.ceil(self.num_images / self.batch_size))

    def __iter__(self):
        # A fixed-seed val loader re-derives the same images every epoch so
        # the validation metric is computed on a stable dataset (otherwise the
        # val curve is a moving target — see the per-idx reseed on the CPU
        # ShapeDataset). Train loaders keep advancing the RNG for fresh data.
        if self.reseed_each_epoch:
            self._rng = random.Random(self._seed)
        for _ in range(len(self)):
            plan = plan_batch(
                batch_size=self.batch_size,
                image_size=self.image_size,
                num_shapes_range=self.num_shapes_range,
                shape_size_range=self.shape_size_range,
                shape_types=self.shape_types,
                rotate_shapes=self.rotate_shapes,
                rng=self._rng,
                max_overlap=self.max_overlap,
            )
            images = render_batch(plan, self.image_size, self.device)
            yield images, plan.annotations
