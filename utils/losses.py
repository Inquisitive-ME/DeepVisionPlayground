import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from models.multiple_center_predictor import ModelType


class CenterPredictionLoss(nn.Module):
    def __init__(self,
                 model_type: ModelType,
                 lambda_conf=1.0,
                 ):
        super().__init__()
        self.lambda_conf = lambda_conf
        self.model_type = model_type

    def forward(self, pred, target_centers, target_classes=None):
        """
        pred: Tensor (batch, max_objects, 3) -> (cx, cy, p)
        target: List of tensors [(num_objects, 2), (num_objects, 2), ...] -> (cx, cy)
        """
        batch_size, max_objects, _ = pred.shape
        device = pred.device

        total_coord_loss = torch.tensor(0.0, device=device)
        total_conf_loss = torch.tensor(0.0, device=device)

        for b in range(batch_size):
            pred_centers = pred[b, :, :2]  # (max_objects, 2)
            pred_confs = pred[b, :, 2]  # (max_objects,)
            if self.model_type == ModelType.center_localization_and_class_id:
                pred_classes = pred[b, :, 3:]

            target_centers_torch = target_centers[b].to(device)  # (num_objects, 2)
            num_objects = target_centers_torch.shape[0]
            if self.model_type == ModelType.center_localization_and_class_id:
                target_classes_torch = target_classes[b].to(device)

            if num_objects == 0:
                # If no ground truth objects exist, all confidence should be 0
                conf_loss = F.binary_cross_entropy(pred_confs, torch.zeros_like(pred_confs))
                total_conf_loss += conf_loss
                continue

            # Compute pairwise distances (cost matrix)
            cost_matrix = torch.cdist(pred_centers, target_centers_torch) - (0.1 * pred_confs[:, None])

            # Solve optimal assignment (Hungarian Algorithm)
            pred_indices, gt_indices = linear_sum_assignment(cost_matrix.cpu().detach().numpy())

            # Convert indices to tensors
            pred_indices = torch.tensor(pred_indices, dtype=torch.long, device=device)
            gt_indices = torch.tensor(gt_indices, dtype=torch.long, device=device)

            # Compute coordinate loss for matched pairs
            matched_pred = pred_centers[pred_indices]  # (min(max_objects, num_objects), 2)
            matched_target = target_centers_torch[gt_indices]  # (min(max_objects, num_objects), 2)
            coord_loss = F.mse_loss(matched_pred, matched_target)

            # Compute object class identification loss
            if self.model_type is ModelType.center_localization_and_class_id:
                match_pred_classes = pred_classes[pred_indices]
                matched_target_classes = target_classes_torch[gt_indices]
                # match_pred_classes = match_pred_classes.permute(1, 0)
                class_loss = F.cross_entropy(match_pred_classes, matched_target_classes)
                coord_loss = coord_loss + class_loss

            # Compute confidence loss for matched predictions
            matched_confs = pred_confs[pred_indices]  # (min(max_objects, num_objects),)
            conf_targets = torch.ones_like(matched_confs)  # They should have high confidence
            conf_loss = F.binary_cross_entropy(matched_confs, conf_targets)

            # Handle unmatched predictions (False Positives → Confidence Should be Low)
            unmatched_mask = torch.ones(max_objects, dtype=torch.bool, device=device)
            unmatched_mask[pred_indices] = False  # Mark matched predictions
            unmatched_confs = pred_confs[unmatched_mask]  # Remaining unmatched predictions
            if unmatched_confs.numel() > 0:
                conf_loss += F.binary_cross_entropy(unmatched_confs, torch.zeros_like(unmatched_confs))

            # Handle unmatched ground truth objects (Missed Detections → Should have been predicted)
            unmatched_gt_mask = torch.ones(num_objects, dtype=torch.bool, device=device)
            unmatched_gt_mask[gt_indices] = False  # Mark matched ground truth objects
            num_missed = int(unmatched_gt_mask.sum().item())
            if num_missed > 0:
                conf_loss += F.binary_cross_entropy(torch.zeros(num_missed, device=device),
                                                    torch.ones(num_missed, device=device))

            total_coord_loss += coord_loss
            total_conf_loss += conf_loss

        # Normalize by batch size
        return total_coord_loss / batch_size + self.lambda_conf * (total_conf_loss / batch_size)
