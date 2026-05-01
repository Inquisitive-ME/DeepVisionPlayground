from enum import Enum, auto

import torch.nn as nn
import torchvision.models as models


class EncodeType(Enum):
    simple = auto()
    simple_gap = auto()
    resnet18 = auto()
    resnet34 = auto()


def encoder(encoder_type: EncodeType) -> tuple[nn.Module, int]:
    if encoder_type is EncodeType.simple:
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 256 -> 128
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
        )
        features_out_size = 128 * 16 * 16
    elif encoder_type is EncodeType.simple_gap:
        # Same conv stack, but global-average-pooled to a 128-d vector. This
        # is the right baseline when comparing to ResNet — it removes the
        # 8M-parameter FC head you'd otherwise get from flattening 16x16x128.
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        features_out_size = 128
    elif encoder_type is EncodeType.resnet18:
        model = models.resnet18(weights=None)
        features_out_size = model.fc.in_features
        model.fc = nn.Identity()
    elif encoder_type is EncodeType.resnet34:
        model = models.resnet34(weights=None)
        features_out_size = model.fc.in_features
        model.fc = nn.Identity()
    else:
        raise ValueError(f"unknown encoder type: {encoder_type}")

    return model, features_out_size
