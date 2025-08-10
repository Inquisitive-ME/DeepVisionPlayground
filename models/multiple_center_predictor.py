from enum import Enum, auto

import torch
import torch.nn as nn

from models.encoders import EncodeType, encoder


class ModelType(Enum):
    center_localization = auto()
    center_localization_and_class_id = auto()


class CenterPredictor(nn.Module):
    def __init__(self,
                 num_classes: int,
                 model_type: ModelType,
                 encoder_type: EncodeType = EncodeType.resnet34,
                 max_objects: int = 10):
        super().__init__()
        if model_type is ModelType.center_localization_and_class_id:
            self.output_shape = (max_objects, (3 + num_classes))
        elif model_type is ModelType.center_localization:
            self.output_shape = (max_objects, 3)
        else:
            assert False, "unknown model type"
        self.backbone, features_out_size = encoder(encoder_type=encoder_type)
        # Predict (cx, cy, p, one-hot-encoding for num classes)
        self.fc_output = nn.Linear(features_out_size, self.output_shape[0] * self.output_shape[1])

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc_output(x)
        x = x.view(-1, * self.output_shape)  # Output shape: (batch, max_objects, 3 + num_classes)
        x[:, :, 2] = torch.sigmoid(x[:, :, 2])
        return x

    @staticmethod
    def filter_predictions(pred, confidence_threshold=0.3):
        """
        Filters out low-confidence predictions from the model's output.

        Args:
            pred: Tensor of shape (batch_size, max_objects, 3) -> (cx, cy, confidence)
            confidence_threshold: Minimum confidence required to keep a prediction.

        Returns:
            List of tensors, where each tensor is (num_valid_objects, 3) -> (cx, cy, confidence).
        """
        batch_size, max_objects, _ = pred.shape
        filtered_predictions = []

        for b in range(batch_size):
            valid_mask = pred[b, :, 2] > confidence_threshold  # Select based on confidence

            valid_preds = pred[b][valid_mask]
            filtered_predictions.append(valid_preds)

        return filtered_predictions
