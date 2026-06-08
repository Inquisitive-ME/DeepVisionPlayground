import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from models.types import ModelType


class CenterPredictionLoss(nn.Module):
    def __init__(self,
                 model_type: ModelType,
                 lambda_conf: float = 1.0,
                 lambda_class: float = 1.0,
                 class_match_weight: float = 0.1,
                 ):
        super().__init__()
        self.lambda_conf = lambda_conf
        self.lambda_class = lambda_class
        # Weight of the class term INSIDE the Hungarian matching cost — kept
        # separate from lambda_class (which weights the supervised class loss).
        # Centers are normalized to [0, 1], so the geometric cdist between
        # distinct objects is ~0.1 while the class cost (-prob) spans ~1.0.
        # If the class term were weighted at lambda_class=1.0 it would dominate
        # matching and flip spatially-correct assignments; a small weight lets
        # class only break ties between near-coincident objects.
        self.class_match_weight = class_match_weight
        self.model_type = model_type

    def forward(self, pred, target_centers, target_classes=None):
        """
        pred: Tensor (batch, max_objects, 3 [+num_classes]) -> (cx, cy, p, [class_logits])
        target_centers: List of tensors [(num_objects, 2), ...] -> (cx, cy) per image
        target_classes: List of long tensors [(num_objects,), ...] (no padding) per image
        """
        batch_size, max_objects, _ = pred.shape
        device = pred.device
        with_class = self.model_type is ModelType.center_localization_and_class_id

        total_coord_loss = torch.tensor(0.0, device=device)
        total_class_loss = torch.tensor(0.0, device=device)
        total_conf_loss = torch.tensor(0.0, device=device)

        for b in range(batch_size):
            pred_centers = pred[b, :, :2]  # (max_objects, 2)
            pred_confs = pred[b, :, 2]  # (max_objects,)
            pred_classes = pred[b, :, 3:] if with_class else None

            target_centers_torch = target_centers[b].to(device)  # (num_objects, 2)
            num_objects = target_centers_torch.shape[0]
            target_classes_torch = (
                target_classes[b].to(device).long()
                if with_class and target_classes is not None
                else None
            )

            if num_objects == 0:
                # No ground-truth objects → all confidences should be 0.
                total_conf_loss = total_conf_loss + F.binary_cross_entropy(
                    pred_confs, torch.zeros_like(pred_confs)
                )
                continue

            # Cost matrix is purely the geometric assignment cost; confidence
            # does NOT bias matching (otherwise the model is rewarded for
            # asserting high confidence on every slot).
            cost_matrix = torch.cdist(pred_centers, target_centers_torch)
            if with_class and pred_classes is not None and target_classes_torch is not None:
                # Lower cost for predictions that already place mass on the
                # correct class (DETR-style class cost: -prob[target]).
                pred_probs = F.softmax(pred_classes, dim=-1)
                class_cost = -pred_probs[:, target_classes_torch]
                cost_matrix = cost_matrix + self.class_match_weight * class_cost

            pred_indices_np, gt_indices_np = linear_sum_assignment(
                cost_matrix.detach().cpu().numpy()
            )
            pred_indices = torch.as_tensor(pred_indices_np, dtype=torch.long, device=device)
            gt_indices = torch.as_tensor(gt_indices_np, dtype=torch.long, device=device)

            matched_pred = pred_centers[pred_indices]
            matched_target = target_centers_torch[gt_indices]
            # Smooth-L1 (Huber) instead of MSE: when matching is noisy
            # early in training a few wildly wrong pairs get matched
            # together, and MSE squares those errors and dominates the
            # gradient. Smooth-L1 is linear in the tail, so a wrong pair
            # contributes a bounded gradient and the model isn't dragged
            # toward "predict the dataset mean" as aggressively. The
            # behavior on tight matches (within the L1/L2 transition) is
            # the same as MSE.
            total_coord_loss = total_coord_loss + F.smooth_l1_loss(
                matched_pred, matched_target,
            )

            if with_class and pred_classes is not None and target_classes_torch is not None:
                matched_pred_classes = pred_classes[pred_indices]
                matched_target_classes = target_classes_torch[gt_indices]
                total_class_loss = total_class_loss + F.cross_entropy(
                    matched_pred_classes, matched_target_classes
                )

            # Matched predictions should have high confidence; unmatched slots
            # (extras beyond the GT count) should have low confidence. Both
            # terms have real gradients on `pred_confs`.
            matched_confs = pred_confs[pred_indices]
            conf_loss = F.binary_cross_entropy(
                matched_confs, torch.ones_like(matched_confs)
            )

            unmatched_mask = torch.ones(max_objects, dtype=torch.bool, device=device)
            unmatched_mask[pred_indices] = False
            unmatched_confs = pred_confs[unmatched_mask]
            if unmatched_confs.numel() > 0:
                conf_loss = conf_loss + F.binary_cross_entropy(
                    unmatched_confs, torch.zeros_like(unmatched_confs)
                )

            total_conf_loss = total_conf_loss + conf_loss

        return (
            total_coord_loss / batch_size
            + self.lambda_class * (total_class_loss / batch_size)
            + self.lambda_conf * (total_conf_loss / batch_size)
        )
