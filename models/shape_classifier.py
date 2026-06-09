"""Whole-image shape classifier: predict the class of the single shape.

The other half of the project's stated vision ("classify ... shapes"), and the
simplest task in the suite — no localization, just a class per image. Reuses the
shared encoders + a linear head; with a ``*_gap`` (global-average-pooled,
position-invariant) encoder this is the textbook classification setup, which
demonstrates that those encoders — useless for localization — are exactly right
here.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.encoders import EncodeType, encoder


class ShapeClassifier(nn.Module):
    def __init__(self, num_classes: int, encoder_type: EncodeType) -> None:
        super().__init__()
        self.features, features_out_size = encoder(encoder_type)
        self.head = nn.Linear(features_out_size, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))  # (B, num_classes) logits
