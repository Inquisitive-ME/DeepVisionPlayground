import torch
import torch.nn as nn

from models.encoders import EncodeType, encoder
from models.types import ModelType


class SimpleCenterNet(nn.Module):
    """Single-object center predictor: outputs (cx, cy) and optionally class logits.

    Centers are produced as raw regression outputs — no sigmoid. With targets
    in [0, 1], an earlier version sigmoided the output thinking it was tidier,
    but that strangles the gradient at init (sigmoid output near 0.5 gives
    very small gradient on MSE for a target distribution centered near 0.5),
    and on this synthetic-shapes task it turned a model that converged in a
    few epochs into one that crawled to Pearson ~0.4 over 100 epochs. Raw
    regression matches the original working config.
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
        return self.head(x)
