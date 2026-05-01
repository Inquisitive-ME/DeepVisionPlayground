"""Configurable training driver for single- and multi-center predictors.

Run from the repo root:

    python -m scripts.run_training --task single --encoder simple_gap \
        --epochs 50 --batch-size 100 --image-size 256

Why this exists: we want to compare encoders, batch sizes, and image
sizes by changing one CLI flag, not by hand-editing the script. It also
writes a structured ``results.json`` next to the TensorBoard run dir, so
downstream analysis can read final metrics + throughput without
re-parsing TensorBoard event files.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from data.annotations import BackgroundType, ShapeOutline, ShapeType
from data.gpu_shapes import GpuShapeLoader
from data.synthetic_shapes_dataset import ShapeDataset, seed_worker
from models.encoders import EncodeType
from models.multiple_center_predictor import CenterPredictor
from models.simple_center_net import SimpleCenterNet
from models.types import ModelType
from utils.losses import CenterPredictionLoss
from utils.metrics import evaluate_multi_object, evaluate_single_object
from utils.perf import configure_for_speed, pick_device
from utils.training_logger import TrainingLogger


@dataclass
class RunConfig:
    task: str  # "single" or "multi"
    encoder: str
    epochs: int
    batch_size: int
    image_size: int
    lr: float
    num_train_images: int
    num_val_images: int
    max_objects: int
    hidden_dims: tuple[int, ...]
    num_shapes_min: int
    num_shapes_max: int
    seed: int
    val_seed: int
    num_workers: int
    gpu_data: bool


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=("single", "multi"), default="single")
    p.add_argument(
        "--encoder",
        choices=("simple", "simple_bn", "simple_gap", "simple_bn_gap", "resnet18", "resnet34"),
        default="simple_bn",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-train-images", type=int, default=1000)
    p.add_argument("--num-val-images", type=int, default=200)
    p.add_argument("--max-objects", type=int, default=5,
                   help="Multi-object predictor's slot count")
    p.add_argument("--hidden-dims", type=str, default="",
                   help="Comma-separated hidden FC dims, e.g. 1024,1024")
    p.add_argument("--num-shapes-min", type=int, default=0)
    p.add_argument("--num-shapes-max", type=int, default=3)
    p.add_argument("--seed", type=int, default=0,
                   help="Train RNG seed (0 = unseeded fresh entropy)")
    p.add_argument("--val-seed", type=int, default=1234)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--gpu-data", action="store_true",
        help=(
            "Generate batches directly on the GPU with data.gpu_shapes "
            "instead of CPU PIL workers. Way higher GPU utilization."
        ),
    )
    args = p.parse_args()

    hd = tuple(int(x) for x in args.hidden_dims.split(",") if x) if args.hidden_dims else ()
    return RunConfig(
        task=args.task,
        encoder=args.encoder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        lr=args.lr,
        num_train_images=args.num_train_images,
        num_val_images=args.num_val_images,
        max_objects=args.max_objects,
        hidden_dims=hd,
        num_shapes_min=args.num_shapes_min,
        num_shapes_max=args.num_shapes_max,
        seed=args.seed,
        val_seed=args.val_seed,
        num_workers=args.num_workers,
        gpu_data=args.gpu_data,
    )


def build_targets_multi(annotations, device):
    centers, classes = [], []
    for ann in annotations:
        if ann:
            centers.append(torch.tensor([o["center"] for o in ann], dtype=torch.float32, device=device))
            classes.append(torch.tensor([o["shape"] for o in ann], dtype=torch.long, device=device))
        else:
            centers.append(torch.zeros((0, 2), dtype=torch.float32, device=device))
            classes.append(torch.zeros((0,), dtype=torch.long, device=device))
    return centers, classes


def build_targets_single(annotations, device):
    centers = [a[0]["center"] for a in annotations]
    classes = [a[0]["shape"] for a in annotations]
    return (
        torch.tensor(centers, dtype=torch.float32, device=device),
        torch.tensor(classes, dtype=torch.long, device=device),
    )


def evaluate_multi(model, loader, device, image_size, class_names):
    model.eval()
    all_outputs: list[torch.Tensor] = []
    all_centers: list[torch.Tensor] = []
    all_classes: list[torch.Tensor] = []
    with torch.no_grad():
        for images, anns in loader:
            if images.device != device:
                images = images.to(device, non_blocking=True)
            centers, classes = build_targets_multi(anns, device)
            out = model(images)
            all_outputs.append(out.detach().cpu())
            all_centers.extend(c.detach().cpu() for c in centers)
            all_classes.extend(c.detach().cpu() for c in classes)
    stacked = torch.cat(all_outputs, dim=0) if all_outputs else torch.empty((0, 0, 0))
    metrics = evaluate_multi_object(
        stacked, all_centers, image_size,
        gt_classes_list=all_classes if all_classes else None,
        has_classes=True,
        class_names=class_names,
    )
    return metrics.to_dict()


def evaluate_single(model, loader, device, image_size):
    model.eval()
    distances_sum = 0.0
    distances_n = 0
    correct = 0
    n_images = 0
    with torch.no_grad():
        for images, anns in loader:
            if images.device != device:
                images = images.to(device, non_blocking=True)
            centers, classes = build_targets_single(anns, device)
            out = model(images)
            batch = evaluate_single_object(
                out, centers, image_size, gt_classes=classes, has_classes=True,
            )
            distances_sum += batch.mean_center_px * batch.n_images
            distances_n += batch.n_images
            correct += int(batch.accuracy * batch.n_images)
            n_images += images.size(0)
    return {
        "single/mean_center_px": distances_sum / max(distances_n, 1),
        "single/accuracy": correct / max(n_images, 1),
    }


def main() -> None:
    cfg = parse_args()
    configure_for_speed()
    device = pick_device()
    print(f"device={device}, config={cfg}")

    img_size = (cfg.image_size, cfg.image_size)
    encoder_type = EncodeType[cfg.encoder]
    model_type = ModelType.center_localization_and_class_id
    transform = transforms.ToTensor()

    train_kwargs: dict[str, Any] = dict(
        image_size=img_size,
        shape_types=tuple(ShapeType),
        shape_size_range=(20, 90),
        background=BackgroundType.SOLID,
        shape_outline=ShapeOutline.FILL,
        add_noise=False,
        rotate_shapes=False,
        max_overlap=0.6,
        transform=transform,
    )

    if cfg.task == "single":
        train_kwargs["num_shapes_range"] = (1, 1)
        train_kwargs["shape_size_range"] = (20, 128)
        train_kwargs["rotate_shapes"] = True
    else:
        train_kwargs["num_shapes_range"] = (cfg.num_shapes_min, cfg.num_shapes_max)

    train_seed = cfg.seed if cfg.seed != 0 else None

    if cfg.gpu_data:
        if device.type != "cuda":
            raise RuntimeError("--gpu-data requires CUDA")
        if train_kwargs["background"] is not BackgroundType.SOLID:
            raise NotImplementedError("GpuShapeLoader only supports SOLID backgrounds")
        gpu_kwargs = dict(
            image_size=img_size,
            num_shapes_range=train_kwargs["num_shapes_range"],
            shape_size_range=train_kwargs["shape_size_range"],
            shape_types=train_kwargs["shape_types"],
            rotate_shapes=train_kwargs["rotate_shapes"],
            device=device,
        )
        train_loader: Any = GpuShapeLoader(
            batch_size=cfg.batch_size,
            num_images=cfg.num_train_images,
            seed=train_seed,
            **gpu_kwargs,
        )
        val_loader: Any = GpuShapeLoader(
            batch_size=cfg.batch_size,
            num_images=cfg.num_val_images,
            seed=cfg.val_seed,
            **gpu_kwargs,
        )
        num_classes = len(ShapeType)
        class_names = tuple(s.name for s in ShapeType)
    else:
        train_dataset = ShapeDataset(num_images=cfg.num_train_images, seed=train_seed, **train_kwargs)
        val_dataset = ShapeDataset(num_images=cfg.num_val_images, seed=cfg.val_seed, **train_kwargs)
        pin_memory = device.type == "cuda"
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.batch_size, shuffle=True,
            collate_fn=ShapeDataset.collate_function,
            num_workers=cfg.num_workers, worker_init_fn=seed_worker,
            persistent_workers=cfg.num_workers > 0, pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=cfg.batch_size, shuffle=False,
            collate_fn=ShapeDataset.collate_function,
            num_workers=0, pin_memory=pin_memory,
        )
        num_classes = len(train_dataset.get_classes())
        class_names = tuple(train_dataset.get_classes())
    if cfg.task == "single":
        model: torch.nn.Module = SimpleCenterNet(
            num_classes=num_classes,
            encoder_type=encoder_type,
            model_type=model_type,
        )
    else:
        model = CenterPredictor(
            num_classes=num_classes,
            model_type=model_type,
            encoder_type=encoder_type,
            max_objects=cfg.max_objects,
            hidden_dims=list(cfg.hidden_dims) if cfg.hidden_dims else None,
        )
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    mse = torch.nn.MSELoss()
    multi_loss = CenterPredictionLoss(model_type=model_type)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run_name = f"{cfg.task}_{cfg.encoder}_{img_size[0]}_bs{cfg.batch_size}"
    print(f"trainable params: {n_params:,}")

    epoch_times: list[float] = []
    final_metrics: dict[str, float] = {}
    with TrainingLogger(root="runs", run_name=run_name) as logger:
        print(f"TB log dir: {logger.run_dir}")
        for epoch in range(cfg.epochs):
            t0 = time.time()
            model.train()
            running_loss = 0.0
            for images, anns in train_loader:
                if images.device != device:
                    images = images.to(device, non_blocking=True)
                if cfg.task == "single":
                    centers, classes = build_targets_single(anns, device)
                    out = model(images)
                    loss = mse(out[:, :2], centers) + F.cross_entropy(out[:, 2:], classes)
                else:
                    centers_list, classes_list = build_targets_multi(anns, device)
                    out = model(images)
                    loss = multi_loss(out, centers_list, classes_list)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

            dt = time.time() - t0
            epoch_times.append(dt)
            n_train = cfg.batch_size * len(train_loader)
            epoch_loss = running_loss / max(n_train, 1)
            ips = n_train / dt

            logger.log_scalar("train/loss", epoch_loss, step=epoch)
            logger.log_scalar("train/images_per_sec", ips, step=epoch)
            logger.log_scalar("train/epoch_seconds", dt, step=epoch)

            if cfg.task == "single":
                vm = evaluate_single(model, val_loader, device, img_size)
            else:
                vm = evaluate_multi(model, val_loader, device, img_size, class_names)
            logger.log_metrics(vm, step=epoch)
            final_metrics = vm

            extras = ", ".join(
                f"{k.split('/')[-1]}={v:.3f}"
                for k, v in vm.items()
                if "_px" in k or "accuracy" in k or "map" in k
            )
            print(
                f"epoch {epoch + 1}/{cfg.epochs} loss={epoch_loss:.4f}"
                f" dt={dt:.2f}s ips={ips:.1f} | {extras}"
            )

        # Pad results.json with run-level summary so downstream analysis
        # doesn't need to re-parse TB events.
        results = {
            "config": asdict(cfg),
            "n_params": n_params,
            "device": str(device),
            "run_dir": str(logger.run_dir),
            "epoch_times_s": epoch_times,
            "mean_epoch_s": sum(epoch_times) / max(len(epoch_times), 1),
            "median_epoch_s": sorted(epoch_times)[len(epoch_times) // 2] if epoch_times else 0.0,
            "mean_images_per_sec": cfg.num_train_images * len(epoch_times) / sum(epoch_times) if epoch_times else 0.0,
            "final_metrics": final_metrics,
        }
        results_path = Path(logger.run_dir) / "results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"results: {results_path}")

        logger.log_hparams(
            {
                "task": cfg.task, "encoder": cfg.encoder, "epochs": cfg.epochs,
                "batch_size": cfg.batch_size, "image_size": cfg.image_size,
                "lr": cfg.lr, "n_params": n_params,
            },
            {k: float(v) for k, v in final_metrics.items() if isinstance(v, (int, float))},
        )


if __name__ == "__main__":
    main()
