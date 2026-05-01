"""Synthetic curved-line dataset.

Generates images with smooth polynomial curves running top-to-bottom and
optional polygon "blob" distractors. Ground truth is one set of polynomial
coefficients per curve, with x expressed as a function of y so vertical
curves don't blow up::

    x(y) = c[0] + c[1] * y + c[2] * y**2 + ...

Conventions mirror ``synthetic_shapes_dataset``: per-instance RNGs so a
fixed ``seed=`` produces reproducible images even with parallel workers,
optional ``transform=`` for converting PIL → tensor, and a
``collate_function`` that batches images and ragged ground-truth lists.
Rendering is done with Pillow only — no opencv dependency.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

if TYPE_CHECKING:
    import torch

rgb_color_type = Tuple[int, int, int]


@dataclass
class LineAnnotation:
    """One polynomial curve in pixel coordinates: x(y) = sum(coeffs[i] * y**i)."""
    coeffs: Tuple[float, ...]
    color: rgb_color_type


def _random_color(rng: np.random.Generator) -> rgb_color_type:
    arr = rng.integers(0, 256, size=3, dtype=np.int64)
    return (int(arr[0]), int(arr[1]), int(arr[2]))


def _generate_polynomial(
    image_size: tuple[int, int],
    x_top: float,
    np_rng: np.random.Generator,
    degree: int = 2,
    max_curvature: float = 0.0004,
    x_bottom_margin: float = 40.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_vals, y_vals, coeffs) for a single curve.

    ``x_top`` pins x at y=0; the bottom is clamped within ``x_bottom_margin``
    pixels of x_top so curves stay roughly vertical and don't lash across
    the canvas.
    """
    coeffs = np_rng.uniform(-max_curvature, max_curvature, degree + 1)
    coeffs[0] = x_top
    y = np.linspace(0, image_size[1] - 1, num=image_size[1])
    x = np.polyval(coeffs[::-1], y)
    x[-1] = float(np.clip(x[-1], x_top - x_bottom_margin, x_top + x_bottom_margin))
    return x.astype(np.int32), y.astype(np.int32), coeffs


def _draw_blob(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    radius: int,
    color: rgb_color_type,
    rng: random.Random,
) -> None:
    """Draw an irregular polygon "blob" centered at (cx, cy)."""
    num_sides = rng.randint(3, 12)
    angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
    points = [
        (
            int(cx + radius * np.cos(a) * rng.uniform(0.4, 2.0)),
            int(cy + radius * np.sin(a) * rng.uniform(0.4, 2.0)),
        )
        for a in angles
    ]
    draw.polygon(points, fill=color)


class LinesDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 1000,
        image_size: tuple[int, int] = (256, 256),
        num_lines_range: tuple[int, int] = (1, 3),
        num_blobs_per_line: int = 20,
        blob_size_range: tuple[int, int] = (4, 8),
        line_thickness: int = 2,
        min_spacing: int = 30,
        max_curvature: float = 0.0004,
        transform: Optional[Callable[[Any], Any]] = None,
        seed: Optional[int] = None,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_lines_range = num_lines_range
        self.num_blobs_per_line = num_blobs_per_line
        self.blob_size_range = blob_size_range
        self.line_thickness = line_thickness
        self.min_spacing = min_spacing
        self.max_curvature = max_curvature
        self.transform = transform
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[Any, list[LineAnnotation]]:
        img, annotations = self.generate_image()
        if self.transform:
            img = self.transform(img)
        return img, annotations

    def generate_image(self) -> tuple[Image.Image, list[LineAnnotation]]:
        rng = self._rng
        np_rng = self._np_rng
        w, h = self.image_size

        bg_color = _random_color(np_rng)
        img = Image.new("RGB", self.image_size, bg_color)
        draw = ImageDraw.Draw(img)

        num_lines = rng.randint(*self.num_lines_range)
        annotations: list[LineAnnotation] = []
        for i in range(num_lines):
            spacing = (w - 2 * self.min_spacing) / max(num_lines, 1)
            x_top = self.min_spacing + i * spacing + spacing * 0.5
            x_vals, y_vals, coeffs = _generate_polynomial(
                self.image_size, x_top=x_top, np_rng=np_rng,
                max_curvature=self.max_curvature,
            )
            line_color = _random_color(np_rng)

            # Render the curve as a polyline so thickness > 1 actually shows.
            points = [
                (int(x), int(y))
                for x, y in zip(x_vals, y_vals)
                if 0 <= x < w and 0 <= y < h
            ]
            if len(points) >= 2:
                draw.line(points, fill=line_color, width=self.line_thickness)

            min_blob, max_blob = self.blob_size_range
            for _ in range(self.num_blobs_per_line):
                radius = rng.randint(min_blob, max_blob)
                idx = rng.randrange(len(x_vals))
                blob_x = int(np.clip(x_vals[idx] + rng.randint(-5, 5), 0, w - 1))
                blob_y = int(np.clip(y_vals[idx] + rng.randint(-5, 5), 0, h - 1))
                blob_color = _random_color(np_rng)
                _draw_blob(draw, blob_x, blob_y, radius, blob_color, rng)

            annotations.append(LineAnnotation(coeffs=tuple(coeffs.tolist()), color=line_color))

        return img, annotations

    @staticmethod
    def collate_function(
        batch: list[tuple[torch.Tensor, list[LineAnnotation]]],
    ) -> tuple[torch.Tensor, list[list[dict[str, Any]]]]:
        images, annotations = zip(*batch)
        images_t = default_collate(images)
        ann_dicts = [
            [{"coeffs": list(a.coeffs), "color": list(a.color)} for a in ann]
            for ann in annotations
        ]
        return images_t, ann_dicts


if __name__ == "__main__":  # pragma: no cover - visual smoke
    import matplotlib.pyplot as plt

    ds = LinesDataset(num_samples=4, seed=0, num_lines_range=(2, 4))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (img, anns) in zip(axes, (ds[i] for i in range(4))):
        ax.imshow(np.array(img))
        ax.set_title(f"{len(anns)} lines")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
