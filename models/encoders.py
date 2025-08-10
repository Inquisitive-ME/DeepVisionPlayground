from enum import Enum, auto

import torch.nn as nn
import torchvision.models as models


class EncodeType(Enum):
    simple = auto()
    resnet18 = auto()
    resnet34 = auto()


def encoder(encoder_type: EncodeType):
    if encoder_type is EncodeType.simple:
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 256x256 -> 128x128
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 128x128 -> 64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
        )
        features_out_size = 128 * 16 * 16
    elif encoder_type is EncodeType.resnet34:
        model = models.resnet34(weights=None)
        model.fc = nn.Identity()
        features_out_size = 512
    elif encoder_type is EncodeType.resnet18:
        model = models.resnet18(pretrained=False)
        # Ensure model.fc is a Linear layer with in_features attribute
        if hasattr(model.fc, 'in_features'):
            features_out_size = model.fc.in_features
        else:
            features_out_size = 512  # Default size for resnet18
        model.fc = nn.Identity()
    else:
        assert False, "unknown encoder type"

    return model, features_out_size
