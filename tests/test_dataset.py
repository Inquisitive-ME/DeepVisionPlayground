"""Tests for synthetic dataset generators and helpers."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.annotations import (
    Annotation,
    AnnotationEncoder,
    BackgroundType,
    BoundingBox,
    ShapeOutline,
    ShapeType,
)
from data.synthetic_lines_dataset import LineAnnotation, LinesDataset
from data.synthetic_shapes_dataset import (
    ShapeDataset,
    compute_overlap_ratio,
    is_overlapping,
    seed_worker,
)


class TestShapeDataset:
    def test_seeded_runs_are_reproducible(self, tiny_image_size):
        a = ShapeDataset(
            num_images=2, image_size=tiny_image_size, seed=42,
            background=BackgroundType.RANDOM, add_noise=True,
        )
        b = ShapeDataset(
            num_images=2, image_size=tiny_image_size, seed=42,
            background=BackgroundType.RANDOM, add_noise=True,
        )
        img_a, ann_a = a[0]
        img_b, ann_b = b[0]
        assert np.array_equal(np.array(img_a), np.array(img_b))
        assert len(ann_a) == len(ann_b)

    def test_unseeded_runs_differ(self, tiny_image_size):
        a = ShapeDataset(num_images=2, image_size=tiny_image_size)
        b = ShapeDataset(num_images=2, image_size=tiny_image_size)
        # Without a seed both instances default to fresh entropy.
        assert not np.array_equal(np.array(a[0][0]), np.array(b[0][0]))

    def test_seeded_index_is_stable_across_calls(self, tiny_image_size):
        """idx -> image must not drift when the dataset is re-iterated.

        This used to break val: _rng advanced on every __getitem__ so epoch
        N+1 saw different images at the same idx, hiding training progress.
        """
        ds = ShapeDataset(num_images=4, image_size=tiny_image_size, seed=7)
        first = np.array(ds[2][0])
        # Pull other indices in between to advance the underlying RNG.
        for i in (0, 1, 3, 0):
            _ = ds[i]
        second = np.array(ds[2][0])
        assert np.array_equal(first, second)

    def test_collate_function_shapes(self, tiny_image_size):
        ds = ShapeDataset(
            num_images=4, image_size=tiny_image_size, seed=0,
            transform=transforms.ToTensor(),
        )
        loader = DataLoader(ds, batch_size=2, collate_fn=ShapeDataset.collate_function)
        images, annotations = next(iter(loader))
        assert images.shape == (2, 3, tiny_image_size[1], tiny_image_size[0])
        assert isinstance(annotations, list) and len(annotations) == 2
        for ann_list in annotations:
            for a in ann_list:
                assert "center" in a and "shape" in a and "bbox" in a

    def test_fixed_dataset_length_mismatch_raises(self, tiny_image_size):
        with tempfile.TemporaryDirectory() as tmp:
            save = os.path.join(tmp, "ds")
            ShapeDataset(
                num_images=3, image_size=tiny_image_size,
                fixed_dataset=True, save_location=save,
            )
            # On reload, requesting a different num_images must error,
            # not silently return a wrong __len__.
            with pytest.raises(ValueError, match="num_images=10"):
                ShapeDataset(
                    num_images=10, image_size=tiny_image_size,
                    fixed_dataset=True, save_location=save,
                )

    def test_fixed_dataset_round_trip(self, tiny_image_size):
        with tempfile.TemporaryDirectory() as tmp:
            save = os.path.join(tmp, "ds")
            ShapeDataset(
                num_images=4, image_size=tiny_image_size, seed=7,
                fixed_dataset=True, save_location=save,
            )
            reloaded = ShapeDataset(
                num_images=4, image_size=tiny_image_size,
                fixed_dataset=True, save_location=save,
            )
            assert len(reloaded) == 4
            img, anns = reloaded[0]
            assert img.size == tiny_image_size
            for a in anns:
                assert isinstance(a, Annotation)
                assert isinstance(a.bbox, BoundingBox)
                assert isinstance(a.shape, ShapeType)

    def test_get_classes_matches_shape_types(self, tiny_image_size):
        ds = ShapeDataset(
            num_images=1, image_size=tiny_image_size, seed=0,
            shape_types=(ShapeType.RECTANGLE, ShapeType.CIRCLE),
        )
        assert ds.get_classes() == ["RECTANGLE", "CIRCLE"]


class TestIsOverlapping:
    def test_pairwise_threshold(self):
        b0 = BoundingBox(0, 0, 100, 100)
        b1 = BoundingBox(50, 50, 150, 150)  # intersection / min_area = 0.25
        assert compute_overlap_ratio(b0, b1) == pytest.approx(0.25)
        assert is_overlapping(b1, [b0], max_overlap_ratio=0.2) is True
        assert is_overlapping(b1, [b0], max_overlap_ratio=0.3) is False

    def test_disjoint_pairs_dont_accumulate(self):
        # Pre-fix bug: summing per-pair ratios across unrelated boxes wrongly
        # tripped the threshold. Each pair-ratio here is 0, so no overlap.
        b1 = BoundingBox(0, 0, 10, 10)
        b2 = BoundingBox(50, 50, 60, 60)
        b3 = BoundingBox(100, 100, 110, 110)
        new_box = BoundingBox(200, 200, 210, 210)
        assert is_overlapping(new_box, [b1, b2, b3], max_overlap_ratio=0.3) is False


class TestAnnotationSerialization:
    def test_round_trip(self):
        import json
        ann = Annotation(
            shape=ShapeType.CIRCLE, bbox=BoundingBox(1, 2, 3, 4),
            center=(0.5, 0.5), color=(10, 20, 30),
        )
        serialized = json.dumps([ann], cls=AnnotationEncoder)
        loaded = [Annotation.from_dict(d) for d in json.loads(serialized)]
        assert loaded[0].shape is ShapeType.CIRCLE
        assert loaded[0].bbox == ann.bbox
        assert tuple(loaded[0].center) == ann.center
        assert tuple(loaded[0].color) == ann.color


class TestSeedWorker:
    def test_seeded_dataset_is_deterministic_across_workers(self, tiny_image_size):
        """With seed set, idx -> image is deterministic regardless of worker.

        This relies on the per-idx reseeding inside ShapeDataset.__getitem__,
        so seed_worker has nothing to do here.
        """
        ds = ShapeDataset(
            num_images=8, image_size=tiny_image_size, seed=42,
            num_shapes_range=(1, 1), shape_outline=ShapeOutline.FILL,
            transform=transforms.ToTensor(),
        )
        a = list(DataLoader(
            ds, batch_size=2, num_workers=2, collate_fn=ShapeDataset.collate_function,
        ))
        b = list(DataLoader(
            ds, batch_size=2, num_workers=0, collate_fn=ShapeDataset.collate_function,
        ))
        # Same indices, same images, regardless of worker count.
        for ba, bb in zip(a, b):
            assert torch.equal(ba[0], bb[0])

    def test_unseeded_workers_diverge_with_seed_worker(self, tiny_image_size):
        """When seed=None, each worker shares an initial RNG state.

        Without seed_worker they emit duplicate images; with it they diverge.
        """
        ds = ShapeDataset(
            num_images=8, image_size=tiny_image_size,
            num_shapes_range=(1, 1), shape_outline=ShapeOutline.FILL,
            transform=transforms.ToTensor(),
        )
        without_init = list(DataLoader(
            ds, batch_size=2, num_workers=2, collate_fn=ShapeDataset.collate_function,
        ))
        with_init = list(DataLoader(
            ds, batch_size=2, num_workers=2, collate_fn=ShapeDataset.collate_function,
            worker_init_fn=seed_worker,
        ))
        assert torch.equal(without_init[0][0], without_init[1][0])
        assert not torch.equal(with_init[0][0], with_init[1][0])


class TestLinesDataset:
    def test_seeded_runs_are_reproducible(self, tiny_image_size):
        a = LinesDataset(num_samples=2, image_size=tiny_image_size, seed=3)
        b = LinesDataset(num_samples=2, image_size=tiny_image_size, seed=3)
        img_a, ann_a = a[0]
        img_b, ann_b = b[0]
        assert np.array_equal(np.array(img_a), np.array(img_b))
        assert len(ann_a) == len(ann_b)

    def test_annotation_type(self, tiny_image_size):
        ds = LinesDataset(num_samples=1, image_size=tiny_image_size, seed=0)
        _, anns = ds[0]
        assert all(isinstance(a, LineAnnotation) for a in anns)
        assert all(len(a.coeffs) == 3 for a in anns)  # degree=2 -> 3 coeffs

    def test_collate(self, tiny_image_size):
        ds = LinesDataset(
            num_samples=4, image_size=tiny_image_size, seed=0,
            transform=transforms.ToTensor(),
        )
        loader = DataLoader(ds, batch_size=2, collate_fn=LinesDataset.collate_function)
        imgs, anns = next(iter(loader))
        assert imgs.shape == (2, 3, tiny_image_size[1], tiny_image_size[0])
        assert all("coeffs" in a[0] for a in anns)
