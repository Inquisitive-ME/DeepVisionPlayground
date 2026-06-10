"""Tests for segmentation label-map (mask) generation on both data paths.

The mask is rendered with the same shapes, in the same draw order, as the image
— so it matches pixel-for-pixel (including overlaps). Background is the class
index just past the real shape classes (``len(ShapeType)``).
"""
from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.annotations import ShapeOutline, ShapeType
from data.gpu_shapes import GpuShapeLoader
from data.synthetic_shapes_dataset import ShapeDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BG = len(ShapeType)  # background class index


def test_gpu_loader_emits_masks_with_right_class_at_centers():
    loader = GpuShapeLoader(
        batch_size=4, num_images=4, image_size=(64, 64),
        num_shapes_range=(1, 1), shape_size_range=(20, 28),
        shape_types=tuple(ShapeType), rotate_shapes=False,
        device=DEVICE, seed=0, with_masks=True,
    )
    batch = next(iter(loader))
    assert len(batch) == 3, "with_masks should yield (images, annotations, masks)"
    images, anns, masks = batch
    assert masks.shape == (4, 64, 64)
    assert masks.dtype == torch.long
    assert int(masks.min()) >= 0 and int(masks.max()) <= BG  # only valid classes
    assert (masks == BG).any(), "some background should remain on a 64px canvas"
    for i in range(4):
        cx, cy = anns[i][0]["center"]
        px, py = int(cx * 64), int(cy * 64)
        assert int(masks[i, py, px]) == anns[i][0]["shape"]  # class at the center


def test_gpu_no_masks_keeps_two_tuple():
    loader = GpuShapeLoader(
        batch_size=2, num_images=2, image_size=(32, 32),
        num_shapes_range=(1, 1), shape_size_range=(8, 14),
        device=DEVICE, seed=0,  # with_masks defaults False
    )
    batch = next(iter(loader))
    assert len(batch) == 2


def test_cpu_dataset_emits_mask_with_right_class():
    ds = ShapeDataset(
        num_images=4, seed=0, image_size=(64, 64),
        num_shapes_range=(1, 1), shape_size_range=(20, 28),
        shape_types=tuple(ShapeType), shape_outline=ShapeOutline.FILL,
        rotate_shapes=False, transform=transforms.ToTensor(), with_masks=True,
    )
    img, ann, mask = ds[0]
    assert mask.shape == (64, 64)
    assert mask.dtype == torch.long
    cx, cy = ann[0].center
    px, py = int(cx * 64), int(cy * 64)
    assert int(mask[py, px]) == ann[0].shape.value


def test_cpu_masks_collate_through_dataloader():
    ds = ShapeDataset(
        num_images=6, seed=1, image_size=(48, 48),
        num_shapes_range=(0, 3), shape_size_range=(10, 20),
        shape_outline=ShapeOutline.FILL, transform=transforms.ToTensor(),
        with_masks=True,
    )
    loader = DataLoader(ds, batch_size=3, collate_fn=ShapeDataset.collate_function)
    images, anns, masks = next(iter(loader))
    assert images.shape == (3, 3, 48, 48)
    assert masks.shape == (3, 48, 48)
    assert masks.dtype == torch.long
    assert int(masks.max()) <= BG


def test_outlined_shape_mask_matches_image():
    """Masks now support outlined shapes (CPU): the mask draws the same outline,
    so it matches the image (interior stays background, not filled)."""
    ds = ShapeDataset(
        num_images=1, seed=0, image_size=(64, 64),
        num_shapes_range=(1, 1), shape_size_range=(24, 30),
        shape_types=(ShapeType.RECTANGLE,), shape_outline=ShapeOutline.THIN,
        rotate_shapes=False, transform=transforms.ToTensor(), with_masks=True,
    )
    img, ann, mask = ds[0]
    bg_pix = img[:, mask == BG]            # (3, Nbg)
    bgcol = bg_pix[:, 0:1]
    fg = img[:, mask != BG]
    # mask foreground <=> image differs from the (solid) background colour.
    assert bool((fg != bgcol).any(0).all())
    # The rectangle interior (its centre) is NOT filled in an outlined mask.
    x0, y0, x1, y1 = ann[0].bbox
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    assert int(mask[cy, cx]) == BG


def test_with_masks_rejects_fixed_dataset():
    # The fixed-dataset cache doesn't persist masks, so the combination would
    # return a None mask and crash collate; reject it up front.
    with pytest.raises(NotImplementedError):
        ShapeDataset(
            image_size=(64, 64), shape_outline=ShapeOutline.FILL,
            with_masks=True, fixed_dataset=True,
        )


def test_cpu_default_no_masks_keeps_two_tuple():
    ds = ShapeDataset(
        num_images=2, seed=0, image_size=(32, 32),
        num_shapes_range=(1, 1), shape_size_range=(8, 14),
        transform=transforms.ToTensor(),  # with_masks defaults False
    )
    item = ds[0]
    assert len(item) == 2
