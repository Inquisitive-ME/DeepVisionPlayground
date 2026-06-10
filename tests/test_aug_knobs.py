"""Tests for the blur and color-threshold data knobs."""
from __future__ import annotations

import pytest
import torch
from torchvision import transforms

from data.annotations import ShapeOutline, ShapeType
from data.dataset_config import DatasetConfig, build_gpu_loader
from data.synthetic_shapes_dataset import ShapeDataset, color_distance, select_shape_color


def _ds(**kw):
    base = dict(
        num_images=1, seed=0, image_size=(64, 64),
        num_shapes_range=(2, 2), shape_size_range=(18, 28),
        shape_types=(ShapeType.RECTANGLE,), shape_outline=ShapeOutline.FILL,
        rotate_shapes=False, transform=transforms.ToTensor(),
    )
    base.update(kw)
    return ShapeDataset(**base)


class TestBlur:
    def test_blur_changes_image(self):
        sharp, _ = _ds(blur=0.0)[0]
        blurred, _ = _ds(blur=2.0)[0]
        assert not torch.allclose(sharp, blurred)

    def test_blur_keeps_mask_crisp(self):
        ds = _ds(blur=2.0, with_masks=True)
        _, _, mask = ds[0]
        # The label is integral (not blurred): only valid class ids appear.
        assert set(mask.unique().tolist()) <= set(range(len(ShapeType) + 1))

    def test_gpu_rejects_blur(self):
        with pytest.raises(NotImplementedError):
            build_gpu_loader(
                DatasetConfig(blur=1.0), batch_size=2, num_images=2,
                image_size=(64, 64), seed=0, device="cpu",
            )


class TestColorThreshold:
    def test_select_shape_color_respects_threshold(self):
        import random
        bg = (128, 128, 128)
        rng = random.Random(0)
        # 10 tries per call, so with threshold 100 almost every sample clears it.
        far = sum(
            color_distance(select_shape_color(bg, threshold=100, rng=rng), bg) > 100
            for _ in range(50)
        )
        assert far >= 45

    def test_color_threshold_propagates(self):
        ds = _ds(color_threshold=200.0)
        assert ds.color_threshold == 200.0

    def test_dataset_config_round_trips_new_knobs(self):
        dc = DatasetConfig.from_dict({"blur": 1.5, "color_threshold": 90.0})
        assert dc.blur == 1.5 and dc.color_threshold == 90.0
