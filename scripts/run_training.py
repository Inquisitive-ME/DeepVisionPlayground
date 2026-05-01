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
from models.center_heatmap_net import CenterHeatmapNet
from models.encoders import EncodeType
from models.multiple_center_predictor import CenterPredictor
from models.simple_center_net import SimpleCenterNet
from models.types import ModelType
from utils.heatmap_loss import HeatmapLoss
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
    lambda_class: float
    lambda_conf: float


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=("single", "multi", "heatmap"), default="single")
    p.add_argument(
        "--encoder",
        choices=(
            "simple", "simple_bn", "simple_gap", "simple_bn_gap",
            "resnet18", "resnet18_spatial",
            "resnet34", "resnet34_spatial",
        ),
        default="simple_bn",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)  # working default for the simple+raw-output config
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
    p.add_argument(
        "--lambda-class", type=float, default=1.0,
        help="Multi-object loss weight on the classification cross-entropy.",
    )
    p.add_argument(
        "--lambda-conf", type=float, default=1.0,
        help="Multi-object loss weight on the confidence BCE.",
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
        lambda_class=args.lambda_class,
        lambda_conf=args.lambda_conf,
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


def evaluate_heatmap(model, loader, device, image_size):
    """Decode the heatmap output into (B, 2+num_classes) and reuse evaluate_single_object."""
    model.eval()
    all_centers_norm: list[torch.Tensor] = []
    all_class_logits: list[torch.Tensor] = []
    all_gt_centers: list[torch.Tensor] = []
    all_gt_classes: list[torch.Tensor] = []
    w, h = image_size
    with torch.no_grad():
        for images, anns in loader:
            if images.device != device:
                images = images.to(device, non_blocking=True)
            gt_centers = torch.tensor(
                [a[0]["center"] for a in anns], dtype=torch.float32, device=device,
            )
            gt_classes = torch.tensor(
                [a[0]["shape"] for a in anns], dtype=torch.long, device=device,
            )
            out = model(images)
            centers_px, class_ids = model.decode(out)
            # Normalize to [0, 1] and synthesize a per-class one-hot logit
            # tensor so the existing single-object metrics can score this.
            centers_norm = centers_px / torch.tensor(
                [w, h], dtype=torch.float32, device=device,
            )
            num_classes = out.class_logits.shape[1]
            class_logits = torch.zeros(
                (centers_norm.shape[0], num_classes),
                dtype=torch.float32, device=device,
            )
            class_logits.scatter_(1, class_ids.unsqueeze(1), 5.0)
            all_centers_norm.append(centers_norm.detach().cpu())
            all_class_logits.append(class_logits.detach().cpu())
            all_gt_centers.append(gt_centers.detach().cpu())
            all_gt_classes.append(gt_classes.detach().cpu())
    if not all_centers_norm:
        return {}
    centers_norm = torch.cat(all_centers_norm)
    class_logits = torch.cat(all_class_logits)
    preds = torch.cat([centers_norm, class_logits], dim=-1)
    metrics = evaluate_single_object(
        preds, torch.cat(all_gt_centers), image_size,
        gt_classes=torch.cat(all_gt_classes), has_classes=True,
    )
    return metrics.to_dict()


def evaluate_single(model, loader, device, image_size):
    """Pool predictions over the val set, then call evaluate_single_object once.

    Aggregating per-batch averages would give a correct mean_center_px but
    a wrong global Pearson (Pearson over batch means != Pearson over the
    pooled population). Collecting once is much simpler and exact.
    """
    model.eval()
    all_preds: list[torch.Tensor] = []
    all_centers: list[torch.Tensor] = []
    all_classes: list[torch.Tensor] = []
    with torch.no_grad():
        for images, anns in loader:
            if images.device != device:
                images = images.to(device, non_blocking=True)
            centers, classes = build_targets_single(anns, device)
            out = model(images)
            all_preds.append(out.detach().cpu())
            all_centers.append(centers.detach().cpu())
            all_classes.append(classes.detach().cpu())
    if not all_preds:
        return {}
    preds = torch.cat(all_preds, dim=0)
    centers = torch.cat(all_centers, dim=0)
    classes = torch.cat(all_classes, dim=0)
    metrics = evaluate_single_object(
        preds, centers, image_size, gt_classes=classes, has_classes=True,
    )
    return metrics.to_dict()


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

    if cfg.task in ("single", "heatmap"):
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
    elif cfg.task == "heatmap":
        model = CenterHeatmapNet(num_classes=num_classes)
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
    multi_loss = CenterPredictionLoss(
        model_type=model_type,
        lambda_class=cfg.lambda_class,
        lambda_conf=cfg.lambda_conf,
    )
    heatmap_loss_fn = HeatmapLoss(lambda_class=cfg.lambda_class)

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
            # Track decomposed terms so we can see which one is moving.
            # For multi-center we don't have a clean three-way split (the
            # CenterPredictionLoss bundles them internally), so only
            # single-center reports the breakdown.
            sums = {
                "loss": 0.0, "mse_centers": 0.0, "ce_class": 0.0,
                "hm_heatmap": 0.0, "hm_offset": 0.0, "hm_class": 0.0,
            }
            for images, anns in train_loader:
                if images.device != device:
                    images = images.to(device, non_blocking=True)
                if cfg.task == "single":
                    centers, classes = build_targets_single(anns, device)
                    out = model(images)
                    mse_term = mse(out[:, :2], centers)
                    ce_term = F.cross_entropy(out[:, 2:], classes)
                    loss = mse_term + ce_term
                    sums["mse_centers"] += float(mse_term.detach()) * images.size(0)
                    sums["ce_class"] += float(ce_term.detach()) * images.size(0)
                elif cfg.task == "heatmap":
                    # GT centers in IMAGE PIXEL coords (not normalized).
                    w, h = img_size
                    centers_norm = torch.tensor(
                        [a[0]["center"] for a in anns], dtype=torch.float32, device=device,
                    )
                    centers_px = centers_norm * torch.tensor(
                        [w, h], dtype=torch.float32, device=device,
                    )
                    classes = torch.tensor(
                        [a[0]["shape"] for a in anns], dtype=torch.long, device=device,
                    )
                    out = model(images)
                    terms = heatmap_loss_fn(out, centers_px, classes)
                    loss = terms.total
                    sums["hm_heatmap"] += float(terms.heatmap.detach()) * images.size(0)
                    sums["hm_offset"] += float(terms.offset.detach()) * images.size(0)
                    sums["hm_class"] += float(terms.class_.detach()) * images.size(0)
                else:
                    centers_list, classes_list = build_targets_multi(anns, device)
                    out = model(images)
                    loss = multi_loss(out, centers_list, classes_list)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sums["loss"] += loss.item() * images.size(0)

            dt = time.time() - t0
            epoch_times.append(dt)
            n_train = cfg.batch_size * len(train_loader)
            epoch_loss = sums["loss"] / max(n_train, 1)
            ips = n_train / dt

            logger.log_scalar("train/loss", epoch_loss, step=epoch)
            logger.log_scalar("train/images_per_sec", ips, step=epoch)
            logger.log_scalar("train/epoch_seconds", dt, step=epoch)
            if cfg.task == "single":
                logger.log_scalar("train/mse_centers", sums["mse_centers"] / max(n_train, 1), step=epoch)
                logger.log_scalar("train/ce_class", sums["ce_class"] / max(n_train, 1), step=epoch)
            elif cfg.task == "heatmap":
                logger.log_scalar("train/hm_heatmap", sums["hm_heatmap"] / max(n_train, 1), step=epoch)
                logger.log_scalar("train/hm_offset", sums["hm_offset"] / max(n_train, 1), step=epoch)
                logger.log_scalar("train/hm_class", sums["hm_class"] / max(n_train, 1), step=epoch)

            if cfg.task == "single":
                vm = evaluate_single(model, val_loader, device, img_size)
            elif cfg.task == "heatmap":
                vm = evaluate_heatmap(model, val_loader, device, img_size)
            else:
                vm = evaluate_multi(model, val_loader, device, img_size, class_names)
            logger.log_metrics(vm, step=epoch)
            final_metrics = vm

            interesting_substrings = ("_px", "accuracy", "map", "pearson")
            extras = ", ".join(
                f"{k.split('/')[-1]}={v:.3f}"
                for k, v in vm.items()
                if any(s in k for s in interesting_substrings)
                # Skip the per-class breakdown to keep the line short;
                # those go to TensorBoard via log_metrics() above.
                and k.count("/") <= 1
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
