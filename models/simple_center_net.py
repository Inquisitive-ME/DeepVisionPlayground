from enum import Enum, auto

import torch
import torch.nn as nn

from models.encoders import EncodeType, encoder


class ModelType(Enum):
    center_localization = auto()
    center_localization_and_class_id = auto()


# Define a simple CNN that outputs two numbers (x, y center coordinates)
class SimpleCenterNet(nn.Module):
    def __init__(self,
                 num_objects: int,
                 encoder_type: EncodeType,
                 model_type: ModelType) -> None:
        super(SimpleCenterNet, self).__init__()
        self.features, features_out_size = encoder(encoder_type)

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
            # Predict x, y coordinates, and one hot encoded label
            nn.Linear(256, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
