from enum import Enum, auto

import torch
import torch.nn as nn
import torchvision.models as models


class ModelType(Enum):
    center_localization = auto()
    center_localization_and_class_id = auto()

class EncodeType(Enum):
    simple = auto()
    resnet34 = auto()

# Define a simple CNN that outputs two numbers (x, y center coordinates)
class SimpleCenterNet(nn.Module):
    def __init__(self,
                 num_objects: int,
                 encoder_type: EncodeType,
                 model_type: ModelType) -> None:
        super(SimpleCenterNet, self).__init__()
        if encoder_type is EncodeType.simple:
            self.features = nn.Sequential(
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
            self.features = models.resnet34(weights=None)
            self.features.fc = nn.Identity()
            features_out_size = 512
        else:
            assert False, "unknown encoder type"

        self.flatten = nn.Flatten()

        if model_type is ModelType.center_localization:
            output_size = 2
        elif model_type is ModelType.center_localization_and_class_id:
            output_size = 2 + num_objects
        else:
            assert False, "unknown encoder type"
        self.fc = nn.Sequential(
            nn.Linear(features_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)  # Predict x, y coordinates, and one hot encoded label
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
