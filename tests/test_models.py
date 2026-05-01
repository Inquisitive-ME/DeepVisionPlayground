"""Tests for model architectures and encoders."""
from __future__ import annotations

import pytest
import torch

from models.encoders import EncodeType, encoder
from models.multiple_center_predictor import CenterPredictor
from models.simple_center_net import SimpleCenterNet
from models.types import ModelType


@pytest.mark.parametrize("encoder_type, expected_features", [
    (EncodeType.simple_gap, 128),
    (EncodeType.resnet18, 512),
    (EncodeType.resnet34, 512),
])
def test_encoder_output_shape(encoder_type, expected_features):
    model, sz = encoder(encoder_type)
    assert sz == expected_features
    out = model(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, expected_features)


def test_simple_encoder_keeps_spatial_dims():
    model, sz = encoder(EncodeType.simple)
    assert sz == 128 * 16 * 16
    out = model(torch.randn(1, 3, 256, 256))
    assert out.shape == (1, 128, 16, 16)


def test_simple_center_net_output_shape_with_classes():
    model = SimpleCenterNet(
        num_classes=3,
        encoder_type=EncodeType.simple_gap,
        model_type=ModelType.center_localization_and_class_id,
    )
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 5)  # 2 centers + 3 class logits


def test_simple_center_net_centers_only_mode():
    model = SimpleCenterNet(
        num_classes=3,
        encoder_type=EncodeType.simple_gap,
        model_type=ModelType.center_localization,
    )
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 2)


def test_center_predictor_centers_and_confidence_in_unit_range():
    model = CenterPredictor(
        num_classes=3,
        model_type=ModelType.center_localization_and_class_id,
        encoder_type=EncodeType.simple_gap,
        max_objects=4,
    )
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 4, 6)  # (cx, cy, conf, 3 class logits)
    assert (out[..., :2] >= 0).all() and (out[..., :2] <= 1).all()
    assert (out[..., 2] >= 0).all() and (out[..., 2] <= 1).all()


def test_center_predictor_with_hidden_dims():
    model = CenterPredictor(
        num_classes=3,
        model_type=ModelType.center_localization_and_class_id,
        encoder_type=EncodeType.simple_gap,
        max_objects=4,
        hidden_dims=[64, 64],
    )
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 4, 6)


def test_unknown_model_type_raises():
    with pytest.raises(ValueError):
        SimpleCenterNet(
            num_classes=3,
            encoder_type=EncodeType.simple_gap,
            model_type="not_a_model_type",  # type: ignore[arg-type]
        )
