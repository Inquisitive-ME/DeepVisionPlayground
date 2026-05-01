import os
import time
from typing import cast

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from matplotlib import patches
from torch.utils.data import DataLoader
from torchvision import transforms

from data.annotations import BackgroundType, ShapeOutline, ShapeType
from data.synthetic_shapes_dataset import ShapeDataset
from models.multiple_center_predictor import CenterPredictor
from models.types import ModelType
from utils.losses import CenterPredictionLoss
from utils.metrics import evaluate_multi_object
from utils.training_logger import TrainingLogger

VAL_SEED = 1234
EVAL_EVERY = 1  # epochs between full validation passes

transform = transforms.ToTensor()
image_size = (256, 256)

train_dataset = ShapeDataset(
    num_images=1000,
    image_size=image_size,
    num_shapes_range=(0, 3),
    shape_types=tuple(ShapeType),
    shape_size_range=(20, 90),
    background=BackgroundType.SOLID,
    shape_outline=ShapeOutline.FILL,
    add_noise=False,
    fixed_dataset=False,
    transform=transform,
    rotate_shapes=False,
    max_overlap=0.6,
)

# Reproducible held-out validation set: same num_shapes / size / background
# as training so we measure generalization on a frozen sample of the
# training distribution. The seed pins the exact images, so any change in
# the metric across runs is attributable to the model, not the data.
val_dataset = ShapeDataset(
    num_images=200,
    image_size=image_size,
    num_shapes_range=(0, 3),
    shape_types=tuple(ShapeType),
    shape_size_range=(20, 90),
    background=BackgroundType.SOLID,
    shape_outline=ShapeOutline.FILL,
    add_noise=False,
    fixed_dataset=False,
    transform=transform,
    rotate_shapes=False,
    max_overlap=0.6,
    seed=VAL_SEED,
)

train_loader = DataLoader(
    train_dataset, batch_size=100, shuffle=True, collate_fn=ShapeDataset.collate_function
)
val_loader = DataLoader(
    val_dataset, batch_size=100, shuffle=False, collate_fn=ShapeDataset.collate_function
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_type = ModelType.center_localization_and_class_id
num_classes = len(train_dataset.get_classes())
save_model_path = "center_predictor.pth"
model = CenterPredictor(
    num_classes=num_classes,
    model_type=model_type,
    max_objects=5,
    hidden_dims=[1024, 1024, 1024],
).to(device)
if os.path.exists(save_model_path):
    print("loading model: ", save_model_path)
    checkpoint = torch.load(save_model_path, map_location=device)
    model_state = model.state_dict()
    filtered_checkpoint = {
        k: v for k, v in checkpoint.items()
        if k in model_state and model_state[k].shape == v.shape
    }
    model.load_state_dict(filtered_checkpoint, strict=False)

optimizer = optim.Adam(model.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=150)
num_epochs = 10_000
center_prediction_loss = CenterPredictionLoss(model_type=model_type)


def build_targets(annotations: list[list[dict]], device: torch.device):
    """Convert the dataloader's per-image annotation dicts into the
    (centers, classes) tensor lists that the loss / metrics expect."""
    centers_per_image = []
    classes_per_image = []
    for ann in annotations:
        if ann:
            centers_per_image.append(
                torch.tensor([o['center'] for o in ann], dtype=torch.float32, device=device)
            )
            classes_per_image.append(
                torch.tensor([o['shape'] for o in ann], dtype=torch.long, device=device)
            )
        else:
            centers_per_image.append(torch.zeros((0, 2), dtype=torch.float32, device=device))
            classes_per_image.append(torch.zeros((0,), dtype=torch.long, device=device))
    return centers_per_image, classes_per_image


def evaluate(model, loader, device, image_size, has_classes: bool):
    """Run the held-out val set through the model and aggregate metrics
    across all batches into a single MultiObjectMetrics."""
    model.eval()
    total = None
    n_images = 0
    val_loss_sum = 0.0
    with torch.no_grad():
        for images, annotations in loader:
            images = images.to(device)
            centers, classes = build_targets(annotations, device)
            outputs = model(images)
            val_loss = center_prediction_loss(outputs, centers, classes)
            val_loss_sum += float(val_loss) * images.size(0)
            batch_metrics = evaluate_multi_object(
                outputs, centers, image_size,
                gt_classes_list=classes, has_classes=has_classes,
            )
            n_images += batch_metrics.n_images
            if total is None:
                total = batch_metrics
            else:
                # Weighted-by-batch aggregation. Batches are equal-sized so a
                # simple mean is fine; storing a rolling weighted sum would
                # be needed for variable batch sizes.
                w_old = total.n_images
                w_new = batch_metrics.n_images
                total_w = w_old + w_new
                for f in (
                    "mean_matched_center_px", "median_matched_center_px",
                    "matched_class_accuracy", "cardinality_error",
                    "mean_conf_matched", "mean_conf_unmatched", "map_center",
                ):
                    setattr(
                        total, f,
                        (getattr(total, f) * w_old + getattr(batch_metrics, f) * w_new) / total_w
                    )
                for t in batch_metrics.precision_at:
                    total.precision_at[t] = (
                        total.precision_at[t] * w_old + batch_metrics.precision_at[t] * w_new
                    ) / total_w
                    total.recall_at[t] = (
                        total.recall_at[t] * w_old + batch_metrics.recall_at[t] * w_new
                    ) / total_w
                total.n_images = total_w
                total.n_gt += batch_metrics.n_gt
                total.n_pred += batch_metrics.n_pred
    if total is None:
        total_dict = {}
    else:
        total_dict = total.to_dict()
    total_dict["val/loss"] = val_loss_sum / max(n_images, 1)
    return total_dict


run_name = "train_multi_center"
hparams = {
    "model": "CenterPredictor",
    "encoder": "resnet34",
    "max_objects": 5,
    "hidden_dims": "1024x3",
    "batch_size": 100,
    "lr": 1e-5,
    "num_epochs": num_epochs,
    "num_classes": num_classes,
    "image_size": f"{image_size[0]}x{image_size[1]}",
}

with TrainingLogger(root="runs", run_name=run_name) as logger:
    print(f"TensorBoard log dir: {logger.run_dir}")
    print("  → tensorboard --logdir runs")

    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for images, annotations in train_loader:
            images = images.to(device)
            centers_per_image, classes_per_image = build_targets(annotations, device)

            optimizer.zero_grad()
            model_outputs = model(images)
            loss = center_prediction_loss(model_outputs, centers_per_image, classes_per_image)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dataset)
        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(epoch_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        logger.log_scalar("train/loss", epoch_loss, step=epoch)
        logger.log_scalar("train/lr", new_lr, step=epoch)

        log_extras = ""
        if (epoch + 1) % EVAL_EVERY == 0:
            val_metrics = evaluate(
                model, val_loader, device, image_size,
                has_classes=(model_type is ModelType.center_localization_and_class_id),
            )
            logger.log_metrics(val_metrics, step=epoch)
            log_extras = (
                f" Val loss: {val_metrics['val/loss']:.4f}"
                f" mean_px: {val_metrics['multi/mean_matched_center_px']:.2f}"
                f" mAP_center: {val_metrics['multi/map_center']:.3f}"
                f" cls_acc: {val_metrics['multi/matched_class_accuracy']:.3f}"
            )
            if val_metrics["val/loss"] < best_val_loss:
                best_val_loss = val_metrics["val/loss"]

        if new_lr != prev_lr:
            print(f"Learning rate reduced from {prev_lr:.8f} to {new_lr:.8f} at epoch {epoch + 1}")
        epoch_time_s = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{num_epochs},"
            f" Loss: {epoch_loss:.4f} Time: {epoch_time_s:.2f}s,"
            f" Remaining: {epoch_time_s * (num_epochs - 1 - epoch) / 60:.2f}min"
            f"{log_extras}"
        )

    print("Training complete.")
    final_val_metrics = evaluate(
        model, val_loader, device, image_size,
        has_classes=(model_type is ModelType.center_localization_and_class_id),
    )
    logger.log_hparams(hparams, final_val_metrics)

torch.save(model.state_dict(), save_model_path)
print("saved model to:", save_model_path)

# Visualize predictions for a few samples from the val dataset (deterministic).
model.eval()
num_samples_to_show = 5
with torch.no_grad():
    data_iter = iter(val_dataset)
    for sample_num in range(num_samples_to_show):
        image, annotations = next(data_iter)

        image = cast("torch.Tensor", image).to(device)
        model_outputs = model(image.unsqueeze(0))
        model_outputs = model.filter_predictions(model_outputs, confidence_threshold=0.5)[0]

        image_np = image.detach().cpu().permute(1, 2, 0).numpy()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image_np)

        label: str | None = 'Predicted'
        for mout in model_outputs:
            pred = mout.cpu().numpy()
            pred_center = pred[:2]
            conf = pred[2]
            if model_type is ModelType.center_localization_and_class_id:
                pred_shape_class = ShapeType(int(torch.argmax(mout[3:], dim=0).cpu().numpy()))
            pred_center_image_location = val_dataset.convert_center_to_image_coordinates(pred_center)
            ax.scatter(*pred_center_image_location, marker='x', color='red', s=100, label=label)
            label = None
            ax.text(*pred_center_image_location, f"{conf:.2f}")
            if model_type is ModelType.center_localization_and_class_id:
                ax.text(
                    pred_center_image_location[0], pred_center_image_location[1] + 10,
                    pred_shape_class.name,
                    color='red', fontsize=8, backgroundcolor='black',
                )

        label = 'Ground Truth'
        for ann in annotations:
            ax.scatter(
                *val_dataset.convert_center_to_image_coordinates(ann.center),
                marker='o', color='green', s=100, label=label,
            )
            label = None
            ax.text(
                ann.bbox.x_min, ann.bbox.y_min - 5, ann.shape.name,
                color='yellow', fontsize=8, backgroundcolor='black',
            )
            rect = patches.Rectangle(
                (ann.bbox.x_min, ann.bbox.y_min),
                ann.bbox.x_max - ann.bbox.x_min,
                ann.bbox.y_max - ann.bbox.y_min,
                linewidth=2, edgecolor='red', facecolor='none',
            )
            ax.add_patch(rect)

        plt.title(f"Sample {sample_num}")
        plt.legend()
        plt.show()
