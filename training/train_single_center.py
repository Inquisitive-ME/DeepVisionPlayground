from typing import cast

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import patches
from torch.utils.data import DataLoader
from torchvision import transforms

from data.annotations import BackgroundType, ShapeOutline, ShapeType
from data.synthetic_shapes_dataset import ShapeDataset, seed_worker
from models.encoders import EncodeType
from models.simple_center_net import SimpleCenterNet
from models.types import ModelType
from utils.metrics import evaluate_single_object
from utils.perf import configure_for_speed, pick_device
from utils.training_logger import TrainingLogger

configure_for_speed()

VAL_SEED = 1234
EVAL_EVERY = 1

transform = transforms.ToTensor()
image_size = (256, 256)

train_dataset = ShapeDataset(
    num_images=1000,
    image_size=image_size,
    num_shapes_range=(1, 1),
    shape_types=tuple(ShapeType),
    shape_size_range=(20, 128),
    background=BackgroundType.SOLID,
    shape_outline=ShapeOutline.FILL,
    add_noise=False,
    fixed_dataset=False,
    transform=transform,
    rotate_shapes=True,
)

val_dataset = ShapeDataset(
    num_images=200,
    image_size=image_size,
    num_shapes_range=(1, 1),
    shape_types=tuple(ShapeType),
    shape_size_range=(20, 128),
    background=BackgroundType.SOLID,
    shape_outline=ShapeOutline.FILL,
    add_noise=False,
    fixed_dataset=False,
    transform=transform,
    rotate_shapes=True,
    seed=VAL_SEED,
)

device = pick_device()
print(device)
pin_memory = device.type == "cuda"

train_loader = DataLoader(
    train_dataset,
    batch_size=100,
    shuffle=True,
    collate_fn=ShapeDataset.collate_function,
    num_workers=4,
    worker_init_fn=seed_worker,
    persistent_workers=True,
    pin_memory=pin_memory,
)
# Val loader runs single-process so a fixed seed= on the dataset gives
# byte-identical outputs across runs without any extra plumbing.
val_loader = DataLoader(
    val_dataset,
    batch_size=100,
    shuffle=False,
    collate_fn=ShapeDataset.collate_function,
    num_workers=0,
    pin_memory=pin_memory,
)
model_type = ModelType.center_localization_and_class_id
model = SimpleCenterNet(
    num_classes=len(train_dataset.get_classes()),
    model_type=model_type,
    encoder_type=EncodeType.simple_gap,
).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, patience=15,
)
mse_loss = torch.nn.MSELoss()
num_epochs = 100


def build_targets(annotations, device):
    centers = [ann[0]['center'] for ann in annotations]
    object_classes = [ann[0]['shape'] for ann in annotations]
    return (
        torch.tensor(centers, dtype=torch.float32, device=device),
        torch.tensor(object_classes, dtype=torch.long, device=device),
    )


def compute_loss(model_outputs, centers_tensor, classes_tensor, model_type):
    if model_type is ModelType.center_localization:
        return mse_loss(model_outputs, centers_tensor)
    if model_type is ModelType.center_localization_and_class_id:
        center_predictions = model_outputs[:, :2]
        class_predictions = model_outputs[:, 2:]
        class_loss = F.cross_entropy(class_predictions, classes_tensor)
        return mse_loss(center_predictions, centers_tensor) + class_loss
    raise ValueError(f"Unsupported model type: {model_type}")


def evaluate(model, loader, device, image_size, model_type):
    model.eval()
    total_loss = 0.0
    n_images = 0
    distances_sum = 0.0
    distances_n = 0
    correct = 0
    with torch.no_grad():
        for images, annotations in loader:
            images = images.to(device, non_blocking=True)
            centers, classes = build_targets(annotations, device)
            outputs = model(images)
            loss = compute_loss(outputs, centers, classes, model_type)
            total_loss += float(loss) * images.size(0)
            n_images += images.size(0)
            batch = evaluate_single_object(
                outputs, centers, image_size,
                gt_classes=classes,
                has_classes=(model_type is ModelType.center_localization_and_class_id),
            )
            distances_sum += batch.mean_center_px * batch.n_images
            distances_n += batch.n_images
            correct += int(batch.accuracy * batch.n_images)
    return {
        "val/loss": total_loss / max(n_images, 1),
        "single/mean_center_px": distances_sum / max(distances_n, 1),
        "single/accuracy": correct / max(n_images, 1),
    }


run_name = "train_single_center"
hparams = {
    "model": "SimpleCenterNet",
    "encoder": "simple_gap",
    "batch_size": 100,
    "lr": 1e-4,
    "num_epochs": num_epochs,
    "image_size": f"{image_size[0]}x{image_size[1]}",
}

with TrainingLogger(root="runs", run_name=run_name) as logger:
    print(f"TensorBoard log dir: {logger.run_dir}")
    print("  → tensorboard --logdir runs")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, annotations in train_loader:
            images = images.to(device, non_blocking=True)
            centers_tensor, classes_tensor = build_targets(annotations, device)
            optimizer.zero_grad()
            model_outputs = model(images)
            loss = compute_loss(model_outputs, centers_tensor, classes_tensor, model_type)
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
            val_metrics = evaluate(model, val_loader, device, image_size, model_type)
            logger.log_metrics(val_metrics, step=epoch)
            log_extras = (
                f" Val loss: {val_metrics['val/loss']:.4f}"
                f" mean_px: {val_metrics['single/mean_center_px']:.2f}"
                f" cls_acc: {val_metrics['single/accuracy']:.3f}"
            )

        if new_lr != prev_lr:
            print(f"Learning rate reduced from {prev_lr:.6f} to {new_lr:.6f} at epoch {epoch + 1}")
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}{log_extras}")

    print("Training complete.")
    final_val_metrics = evaluate(model, val_loader, device, image_size, model_type)
    logger.log_hparams(hparams, final_val_metrics)

# Visualize predictions on the deterministic val dataset.
model.eval()
num_samples_to_show = 5
with torch.no_grad():
    data_iter = iter(val_dataset)
    for sample_num in range(num_samples_to_show):
        image, ann = next(data_iter)
        gt_center = ann[0].center
        image = cast("torch.Tensor", image).to(device)
        model_outputs = model(image.unsqueeze(0))
        pred_center = model_outputs.cpu().numpy()[0, :2]
        if model_type is ModelType.center_localization_and_class_id:
            pred_shape_class = ShapeType(int(torch.argmax(model_outputs[0, 2:], dim=0).cpu().numpy()))
        image_np = image.detach().cpu().permute(1, 2, 0).numpy()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image_np)
        ax.scatter(
            *val_dataset.convert_center_to_image_coordinates(pred_center),
            marker='x', color='red', s=100, label='Predicted',
        )
        ax.scatter(
            *val_dataset.convert_center_to_image_coordinates(gt_center),
            marker='o', color='green', s=100, label='Ground Truth',
        )
        ax.text(
            ann[0].bbox[0], ann[0].bbox[1] - 5, ann[0].shape.name,
            color='yellow', fontsize=8, backgroundcolor='black',
        )
        if model_type is ModelType.center_localization_and_class_id:
            ax.text(
                ann[0].bbox[0], ann[0].bbox[1] + 10, pred_shape_class.name,
                color='red', fontsize=8, backgroundcolor='black',
            )
        rect = patches.Rectangle(
            (ann[0].bbox[0], ann[0].bbox[1]),
            ann[0].bbox[2] - ann[0].bbox[0],
            ann[0].bbox[3] - ann[0].bbox[1],
            linewidth=2, edgecolor='red', facecolor='none',
        )
        ax.add_patch(rect)
        plt.title(f"Sample {sample_num}")
        plt.legend()
        plt.show()
