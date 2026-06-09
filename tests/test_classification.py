"""Tests for the whole-image shape-classification task."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from data.annotations import ShapeOutline, ShapeType
from data.dataset_config import DatasetConfig
from data.synthetic_shapes_dataset import ShapeDataset
from models.encoders import EncodeType
from models.shape_classifier import ShapeClassifier
from scripts.run_training import RunConfig, check_single_shape_tasks, evaluate_classification


def test_single_shape_task_rejects_empty_images():
    """A single-object task whose distribution can yield a 0-shape image must
    error clearly up front, not IndexError deep in training."""
    cfg = RunConfig(
        task="classification",
        train=DatasetConfig(num_shapes_range=(0, 3)),
        val=DatasetConfig(num_shapes_range=(0, 3)),
    )
    with pytest.raises(ValueError):
        check_single_shape_tasks(cfg)


def test_single_shape_task_ok_and_multi_unaffected():
    check_single_shape_tasks(RunConfig(
        task="classification",
        train=DatasetConfig(num_shapes_range=(1, 1)),
        val=DatasetConfig(num_shapes_range=(1, 1)),
    ))
    # Multi-object tasks legitimately allow empty images.
    check_single_shape_tasks(RunConfig(
        task="multi",
        train=DatasetConfig(num_shapes_range=(0, 3)),
        val=DatasetConfig(num_shapes_range=(0, 3)),
    ))


def test_classifier_output_shape():
    model = ShapeClassifier(num_classes=3, encoder_type=EncodeType.simple_gn_gap)
    out = model(torch.randn(4, 3, 64, 64))
    assert out.shape == (4, 3)


def test_classifier_gradient_flows():
    model = ShapeClassifier(num_classes=3, encoder_type=EncodeType.simple_gn_gap)
    loss = model(torch.randn(2, 3, 64, 64)).square().mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


class _AlwaysClassZero(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.full((x.shape[0], 3), -10.0)
        out[:, 0] = 10.0
        return out


def test_evaluate_classification_known_answer():
    # All shapes are rectangles (class 0); a model that always predicts class 0
    # must score accuracy 1.0.
    ds = ShapeDataset(
        num_images=6, seed=0, image_size=(32, 32),
        num_shapes_range=(1, 1), shape_size_range=(8, 14),
        shape_types=(ShapeType.RECTANGLE,), shape_outline=ShapeOutline.FILL,
        transform=transforms.ToTensor(),
    )
    loader = DataLoader(ds, batch_size=3, collate_fn=ShapeDataset.collate_function)
    metrics = evaluate_classification(_AlwaysClassZero(), loader, torch.device("cpu"))
    assert metrics["classification/accuracy"] == 1.0
