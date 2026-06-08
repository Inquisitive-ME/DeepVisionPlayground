"""Tests for model architectures and encoders."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

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
    """The simple encoder flattens at the end so it can plug into either
    SimpleCenterNet (which then no-op-flattens again) or CenterPredictor
    (which expects a 2-D tensor straight into its FC layers)."""
    model, sz = encoder(EncodeType.simple)
    assert sz == 128 * 16 * 16
    out = model(torch.randn(1, 3, 256, 256))
    assert out.shape == (1, 128 * 16 * 16)


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


def test_center_predictor_confidence_in_unit_range():
    """Confidence is sigmoided (the loss BCEs it). Centers are raw — see
    the SimpleCenterNet sigmoid-revert commit for why; same logic here."""
    model = CenterPredictor(
        num_classes=3,
        model_type=ModelType.center_localization_and_class_id,
        encoder_type=EncodeType.simple_gap,
        max_objects=4,
    )
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 4, 6)  # (cx, cy, conf, 3 class logits)
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


@pytest.mark.parametrize("encoder_type, expected_features", [
    (EncodeType.simple_gn, 128 * 16 * 16),
    (EncodeType.simple_gn_gap, 128),
])
def test_groupnorm_encoder_output_shape(encoder_type, expected_features):
    model, sz = encoder(encoder_type)
    assert sz == expected_features
    out = model(torch.randn(2, 3, 256, 256))
    assert out.shape == (2, expected_features)


def test_center_predictor_default_path_has_no_batchnorm():
    """The regression head + the default encoder must avoid BatchNorm, whose
    frozen eval stats bias distribution-shift measurements. GroupNorm is used
    instead (per-sample, mode-invariant)."""
    model = CenterPredictor(
        num_classes=3,
        model_type=ModelType.center_localization_and_class_id,
        encoder_type=EncodeType.simple_gn_gap,
        max_objects=4,
        hidden_dims=[64, 64],
    )
    assert not any(
        isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in model.modules()
    )
    assert any(isinstance(m, nn.GroupNorm) for m in model.modules())


def test_center_predictor_eval_matches_train_mode():
    """With dropout neutralized, train- and eval-mode outputs must match — they
    would diverge with BatchNorm (batch stats vs frozen running stats)."""
    model = CenterPredictor(
        num_classes=3,
        model_type=ModelType.center_localization_and_class_id,
        encoder_type=EncodeType.simple_gn_gap,
        max_objects=4,
        hidden_dims=[64, 64],
    )
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    x = torch.randn(2, 3, 64, 64)
    model.train()
    out_train = model(x)
    model.eval()
    out_eval = model(x)
    assert torch.allclose(out_train, out_eval, atol=1e-5)


def test_unknown_model_type_raises():
    with pytest.raises(ValueError):
        SimpleCenterNet(
            num_classes=3,
            encoder_type=EncodeType.simple_gap,
            model_type="not_a_model_type",  # type: ignore[arg-type]
        )
