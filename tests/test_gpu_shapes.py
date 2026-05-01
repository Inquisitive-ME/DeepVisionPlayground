"""Tests for the GPU-side batched shape renderer."""
from __future__ import annotations

import pytest
import torch

from data.annotations import ShapeType
from data.gpu_shapes import GpuShapeLoader

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"


def test_loader_produces_correct_batch_shape():
    loader = GpuShapeLoader(
        batch_size=4, num_images=8, image_size=(64, 64),
        num_shapes_range=(1, 1), shape_size_range=(20, 30),
        shape_types=(ShapeType.RECTANGLE,), rotate_shapes=False,
        device=DEVICE, seed=0,
    )
    assert len(loader) == 2
    for imgs, anns in loader:
        assert imgs.shape == (4, 3, 64, 64)
        assert imgs.dtype == torch.float32
        assert imgs.min() >= 0.0 and imgs.max() <= 1.0
        assert len(anns) == 4
        for ann_list in anns:
            assert len(ann_list) == 1
            assert "center" in ann_list[0] and "shape" in ann_list[0]


@pytest.mark.parametrize("shape_type", [
    ShapeType.RECTANGLE, ShapeType.CIRCLE, ShapeType.TRIANGLE,
])
def test_pixel_at_center_matches_shape_color(shape_type):
    """Sample center -> rendered pixel should equal the annotation color.

    Catches off-by-one errors in the rasterizer and pairing bugs between
    annotations and the rendered tensor.
    """
    loader = GpuShapeLoader(
        batch_size=4, num_images=4, image_size=(64, 64),
        num_shapes_range=(1, 1), shape_size_range=(20, 30),
        shape_types=(shape_type,), rotate_shapes=False,
        device=DEVICE, seed=42,
    )
    imgs, anns = next(iter(loader))
    for i in range(4):
        cx, cy = anns[i][0]["center"]
        col = anns[i][0]["color"]
        px, py = int(cx * 64), int(cy * 64)
        actual = (imgs[i, :, py, px].cpu().numpy() * 255).astype(int)
        diff = sum(abs(int(actual[c]) - col[c]) for c in range(3))
        # Allow up to 3 of total RGB delta for float -> uint8 round-trip.
        assert diff <= 3, f"shape={shape_type}: pixel {actual.tolist()} != color {col}"


def test_seeded_runs_are_reproducible():
    a = GpuShapeLoader(
        batch_size=4, num_images=4, image_size=(32, 32),
        num_shapes_range=(1, 2), shape_size_range=(8, 14),
        shape_types=tuple(ShapeType), rotate_shapes=True,
        device=DEVICE, seed=99,
    )
    b = GpuShapeLoader(
        batch_size=4, num_images=4, image_size=(32, 32),
        num_shapes_range=(1, 2), shape_size_range=(8, 14),
        shape_types=tuple(ShapeType), rotate_shapes=True,
        device=DEVICE, seed=99,
    )
    imgs_a, _ = next(iter(a))
    imgs_b, _ = next(iter(b))
    assert torch.allclose(imgs_a, imgs_b)


def test_small_dataset_still_yields_a_batch():
    """num_images < batch_size used to produce zero batches, breaking val."""
    loader = GpuShapeLoader(
        batch_size=64, num_images=8, image_size=(32, 32),
        num_shapes_range=(1, 1), shape_size_range=(8, 14),
        device=DEVICE, seed=0,
    )
    assert len(loader) == 1
    imgs, anns = next(iter(loader))
    assert imgs.shape[0] == 64


def test_multi_shape_per_image():
    loader = GpuShapeLoader(
        batch_size=2, num_images=2, image_size=(64, 64),
        num_shapes_range=(2, 3), shape_size_range=(10, 16),
        device=DEVICE, seed=0,
    )
    _, anns = next(iter(loader))
    counts = [len(ann_list) for ann_list in anns]
    assert all(2 <= c <= 3 for c in counts)
