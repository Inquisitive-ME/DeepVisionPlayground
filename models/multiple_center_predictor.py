import torch
import torch.nn as nn

from models.encoders import EncodeType, encoder
from models.types import ModelType


class CenterPredictor(nn.Module):
    def __init__(self,
                 num_classes: int,
                 model_type: ModelType,
                 encoder_type: EncodeType = EncodeType.resnet34,
                 max_objects: int = 5,
                 hidden_dims: list[int] | None = None
                 ):
        super().__init__()
        if model_type is ModelType.center_localization_and_class_id:
            self.output_shape = (max_objects, (3 + num_classes))
        elif model_type is ModelType.center_localization:
            self.output_shape = (max_objects, 3)
        else:
            assert False, "unknown model type"
        self.backbone, features_out_size = encoder(encoder_type=encoder_type)

        # Add hidden layers if specified
        if hidden_dims:
            layers = []
            current_size = features_out_size
            for dim in hidden_dims:
                layers.extend([
                    nn.Linear(current_size, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
                current_size = dim
            self.hidden_layers: nn.Sequential | nn.Identity = nn.Sequential(*layers)
            final_size = current_size
        else:
            self.hidden_layers = nn.Identity()
            final_size = features_out_size

        # Predict (cx, cy, p, one-hot-encoding for num classes)
        self.fc_output = nn.Linear(final_size, self.output_shape[0] * self.output_shape[1])
        self.max_objects = max_objects

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.hidden_layers(x)
        x = self.fc_output(x)
        x = x.view(-1, *self.output_shape)  # (batch, max_objects, 3 [+ num_classes])
        centers = torch.sigmoid(x[..., :2])  # GT is normalized to [0, 1]
        confidence = torch.sigmoid(x[..., 2:3])
        if x.shape[-1] > 3:
            class_logits = x[..., 3:]
            return torch.cat([centers, confidence, class_logits], dim=-1)
        return torch.cat([centers, confidence], dim=-1)

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
