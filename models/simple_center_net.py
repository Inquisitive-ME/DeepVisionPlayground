import torch
import torch.nn as nn

from models.encoders import EncodeType, encoder
from models.types import ModelType


class SimpleCenterNet(nn.Module):
    """Single-object center predictor: outputs (cx, cy) and optionally class logits.

    Use ``EncodeType.simple_gap`` (or any ResNet) to get a flat feature vector
    and avoid the 8M-parameter dense head that ``EncodeType.simple`` produces.
    """

    def __init__(self,
                 num_classes: int,
                 encoder_type: EncodeType,
                 model_type: ModelType) -> None:
        super().__init__()
        self.model_type = model_type
        self.features, features_out_size = encoder(encoder_type)

        if model_type is ModelType.center_localization:
            output_size = 2
        elif model_type is ModelType.center_localization_and_class_id:
            output_size = 2 + num_classes
        else:
            raise ValueError(f"unknown model type: {model_type}")

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        # GT centers are normalized to [0, 1]; sigmoid keeps the regression
        # well-conditioned and removes the need to clamp at inference time.
        if self.model_type is ModelType.center_localization_and_class_id:
            centers = torch.sigmoid(x[..., :2])
            class_logits = x[..., 2:]
            return torch.cat([centers, class_logits], dim=-1)
        return torch.sigmoid(x)
