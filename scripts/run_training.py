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
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from data.annotations import ShapeType
from data.dataset_config import DatasetConfig, build_cpu_dataset, build_gpu_loader
from data.synthetic_shapes_dataset import ShapeDataset, seed_worker
from models.center_heatmap_net import CenterHeatmapNet
from models.encoders import EncodeType
from models.multi_heatmap_net import MultiHeatmapNet
from models.multiple_center_predictor import CenterPredictor
from models.seg_net import ShapeSegNet
from models.simple_center_net import SimpleCenterNet
from models.types import ModelType
from utils.heatmap_loss import HeatmapLoss, MultiHeatmapLoss
from utils.losses import CenterPredictionLoss
from utils.metrics import (
    evaluate_multi_object,
    evaluate_segmentation,
    evaluate_single_object,
    segmentation_confusion,
)
from utils.perf import configure_for_speed, pick_device
from utils.seg_loss import SegLoss
from utils.training_logger import TrainingLogger


@dataclass
class RunConfig:
    """A full, self-contained run/study spec.

    The data distributions are first-class: ``train`` and ``val`` are each a
    ``DatasetConfig``, so "train on one distribution, evaluate on another"
    (e.g. train without rotation, validate with it) is expressed by giving
    ``val`` different knobs — no per-augmentation CLI flag needed. Defaults
    match the historical CLI behavior, so a config file can omit anything it
    doesn't change.
    """
    task: str = "single"
    encoder: str = "simple_gn"
    epochs: int = 30
    batch_size: int = 100
    image_size: int = 256
    lr: float = 1e-4
    num_train_images: int = 1000
    num_val_images: int = 200
    max_objects: int = 5
    hidden_dims: tuple[int, ...] = ()
    seed: int = 0
    val_seed: int = 1234
    num_workers: int = 4
    gpu_data: bool = False
    lambda_class: float = 1.0
    lambda_conf: float = 1.0
    class_match_weight: float = 0.1
    heatmap_stride: int = 4
    seg_stride: int = 1
    val_shape_counts: tuple[int, ...] = ()
    val_shape_sizes: tuple[tuple[int, int], ...] = ()
    val_overlaps: tuple[float, ...] = ()
    # The train and val data distributions. Both default to an independent
    # default DatasetConfig here; parse_args() and load_run_config() always set
    # them explicitly (val inherits train), so val mirrors train unless a study
    # config overrides it. (Construct RunConfig directly => set val yourself.)
    train: DatasetConfig = field(default_factory=DatasetConfig)
    val: DatasetConfig = field(default_factory=DatasetConfig)


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", type=str, default="",
        help=(
            "Path to a YAML study config (see configs/). When given it is the "
            "authoritative, self-contained run spec — including the train/val "
            "data distributions — and the other CLI flags are ignored. This is "
            "how you run a distribution-shift study (e.g. train without rotation, "
            "validate with it) without per-augmentation flags."
        ),
    )
    p.add_argument(
        "--task",
        choices=("single", "multi", "heatmap", "multi_heatmap", "segmentation"),
        default="single",
    )
    p.add_argument(
        "--encoder",
        choices=(
            "simple", "simple_bn", "simple_gn",
            "simple_gap", "simple_bn_gap", "simple_gn_gap",
            "resnet18", "resnet18_spatial",
            "resnet34", "resnet34_spatial",
        ),
        default="simple_gn",
        help=(
            "GroupNorm (simple_gn) is the default: per-sample norm behaves "
            "identically in train/eval. simple_bn and the resnet* encoders use "
            "BatchNorm, whose frozen eval stats bias distribution-shift studies."
        ),
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
    p.add_argument(
        "--class-match-weight", type=float, default=0.1,
        help=(
            "Weight of the class term inside the multi-object Hungarian "
            "matching cost. Kept small (default 0.1) so class only breaks "
            "geometric ties; centers are normalized to [0, 1], so a larger "
            "weight would let class override spatially-correct assignments."
        ),
    )
    p.add_argument(
        "--heatmap-stride", type=int, default=4,
        help=(
            "Output stride for heatmap models. 4 = 64x64 heatmap on 256-px input, "
            "fast and ~2-3 px median error. 2 = 128x128, sub-pixel error. 1 = 256x256."
        ),
    )
    p.add_argument(
        "--seg-stride", type=int, default=1,
        help=(
            "Output stride for the segmentation model (--task segmentation). "
            "1 = full-resolution masks (the target for mIoU->1.0); larger is "
            "cheaper and upsampled for the loss/metric."
        ),
    )
    p.add_argument(
        "--val-shape-counts", type=str, default="",
        help=(
            "Optional. Comma-separated list of additional shape counts to "
            "evaluate the trained model against (e.g. '1,3,5,10,20'). For each "
            "count N we build a val set with num_shapes_range=(N,N) and run "
            "the same metrics, recording the results under 'final_metrics_by_count' "
            "in results.json. Multi-object tasks only."
        ),
    )
    p.add_argument(
        "--val-shape-sizes", type=str, default="",
        help=(
            "Optional. Comma-separated 'min:max' pairs to sweep shape size at "
            "fixed count (e.g. '10:30,30:60,60:120,120:200'). For each pair we "
            "build a val set with shape_size_range=(min,max) and num_shapes_range "
            "matched to the training config. Recorded under 'final_metrics_by_size'."
        ),
    )
    p.add_argument(
        "--val-overlaps", type=str, default="",
        help=(
            "Optional. Comma-separated max_overlap floats (e.g. '0.1,0.3,0.6,0.9'). "
            "Builds a val set per overlap level and tests whether NMS collapses "
            "nearby shapes into one detection. Recorded under "
            "'final_metrics_by_overlap'."
        ),
    )
    args = p.parse_args()

    if args.config:
        return load_run_config(args.config)

    hd = tuple(int(x) for x in args.hidden_dims.split(",") if x) if args.hidden_dims else ()
    val_shape_counts = tuple(int(x) for x in args.val_shape_counts.split(",") if x) if args.val_shape_counts else ()
    val_shape_sizes_raw = [s for s in args.val_shape_sizes.split(",") if s] if args.val_shape_sizes else []
    val_shape_sizes: list[tuple[int, int]] = []
    for s in val_shape_sizes_raw:
        lo, hi = s.split(":")
        val_shape_sizes.append((int(lo), int(hi)))
    val_overlaps = tuple(float(x) for x in args.val_overlaps.split(",") if x) if args.val_overlaps else ()
    # No config file: build the train distribution from the per-task default
    # plus the surviving --num-shapes flags; val mirrors train (the historical
    # behavior). Augmentation/rotation studies go through a --config file.
    train_dc = DatasetConfig.default_for_task(
        args.task, num_shapes_range=(args.num_shapes_min, args.num_shapes_max),
    )
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
        seed=args.seed,
        val_seed=args.val_seed,
        num_workers=args.num_workers,
        gpu_data=args.gpu_data,
        lambda_class=args.lambda_class,
        lambda_conf=args.lambda_conf,
        class_match_weight=args.class_match_weight,
        heatmap_stride=args.heatmap_stride,
        seg_stride=args.seg_stride,
        val_shape_counts=val_shape_counts,
        val_shape_sizes=tuple(val_shape_sizes),
        val_overlaps=val_overlaps,
        train=train_dc,
        val=train_dc.merged(None),
    )


def load_run_config(path: str) -> RunConfig:
    """Load a YAML study config into a RunConfig.

    Top-level keys are run/model scalars (task, encoder, epochs, ...). The
    ``train:`` and ``val:`` sub-maps are DatasetConfig knobs; ``val`` inherits
    every field of ``train`` and overrides only what it lists, so a shift study
    is just the one knob that differs.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    train_section = raw.pop("train", None)
    val_section = raw.pop("val", None)
    # Seed from the per-task default so a config can omit task-specific knobs
    # (e.g. a heatmap study inherits num_shapes_range=(1,1) without restating it).
    base = DatasetConfig.default_for_task(raw.get("task", "single"))
    train_dc = base.merged(train_section)
    val_dc = train_dc.merged(val_section)  # val inherits train, overrides applied

    known = {f.name for f in fields(RunConfig)} - {"train", "val"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown study-config keys: {sorted(unknown)} (valid: {sorted(known)})")
    scalars: dict[str, Any] = dict(raw)
    if "hidden_dims" in scalars and scalars["hidden_dims"] is not None:
        scalars["hidden_dims"] = tuple(scalars["hidden_dims"])
    for k in ("val_shape_counts", "val_overlaps"):
        if scalars.get(k) is not None:
            scalars[k] = tuple(scalars[k])
    if scalars.get("val_shape_sizes") is not None:
        scalars["val_shape_sizes"] = tuple(tuple(p) for p in scalars["val_shape_sizes"])
    return RunConfig(train=train_dc, val=val_dc, **scalars)


def _ann_size_px(o: Any) -> float:
    """Extract a single 'size' scalar (max of width / height in pixels) from
    one annotation dict. ``bbox`` is a 4-tuple ``(x_min, y_min, x_max, y_max)``
    in pixel coordinates, regardless of whether it came from the CPU or GPU
    dataset path."""
    bb = o["bbox"]
    if hasattr(bb, "x_min"):
        return float(max(bb.x_max - bb.x_min, bb.y_max - bb.y_min))
    return float(max(bb[2] - bb[0], bb[3] - bb[1]))


def _build_sweep_loader(
    cfg: "RunConfig",
    img_size: tuple[int, int],
    device: torch.device,
    num_shapes_range: tuple[int, int],
    shape_size_range: tuple[int, int],
    label: str,
    max_overlap: float | None = None,
) -> Any:
    """Build a one-off val loader at a different num_shapes / shape_size /
    max_overlap config.

    The sweep starts from the run's val distribution (``cfg.val``) and overrides
    only the swept dimension, so everything else (rotation, background, outline,
    noise) matches the model's evaluation distribution. With --gpu-data every
    sweep point — including overlap — is built by the GPU loader, which now
    enforces max_overlap, so all points come from ONE renderer and are
    comparable; otherwise it uses the CPU dataset.
    """
    print(f"  building sweep loader for {label}")
    overrides: dict[str, Any] = {
        "num_shapes_range": num_shapes_range,
        "shape_size_range": shape_size_range,
    }
    if max_overlap is not None:  # only the overlap sweep overrides this
        overrides["max_overlap"] = max_overlap
    sweep_dc = cfg.val.merged(overrides)
    if cfg.gpu_data:
        return build_gpu_loader(
            sweep_dc, batch_size=cfg.batch_size, num_images=cfg.num_val_images,
            image_size=img_size, seed=cfg.val_seed, device=device,
            reseed_each_epoch=True,
        )
    sweep_dataset = build_cpu_dataset(
        sweep_dc, num_images=cfg.num_val_images, image_size=img_size,
        seed=cfg.val_seed, transform=transforms.ToTensor(),
    )
    return DataLoader(
        sweep_dataset, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=ShapeDataset.collate_function,
        num_workers=0, pin_memory=device.type == "cuda",
    )


def build_targets_multi(annotations, device):
    centers, classes, sizes = [], [], []
    for ann in annotations:
        if ann:
            centers.append(torch.tensor([o["center"] for o in ann], dtype=torch.float32, device=device))
            classes.append(torch.tensor([o["shape"] for o in ann], dtype=torch.long, device=device))
            sizes.append(torch.tensor([_ann_size_px(o) for o in ann], dtype=torch.float32, device=device))
        else:
            centers.append(torch.zeros((0, 2), dtype=torch.float32, device=device))
            classes.append(torch.zeros((0,), dtype=torch.long, device=device))
            sizes.append(torch.zeros((0,), dtype=torch.float32, device=device))
    return centers, classes, sizes


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
    all_sizes: list[torch.Tensor] = []
    with torch.no_grad():
        for images, anns in loader:
            if images.device != device:
                images = images.to(device, non_blocking=True)
            centers, classes, sizes = build_targets_multi(anns, device)
            out = model(images)
            all_outputs.append(out.detach().cpu())
            all_centers.extend(c.detach().cpu() for c in centers)
            all_classes.extend(c.detach().cpu() for c in classes)
            all_sizes.extend(s.detach().cpu() for s in sizes)
    stacked = torch.cat(all_outputs, dim=0) if all_outputs else torch.empty((0, 0, 0))
    metrics = evaluate_multi_object(
        stacked, all_centers, image_size,
        gt_classes_list=all_classes if all_classes else None,
        has_classes=True,
        class_names=class_names,
        gt_sizes_list=all_sizes if all_sizes else None,
    )
    return metrics.to_dict()


def evaluate_multi_heatmap(model, loader, device, image_size, class_names, max_objects):
    """Decode top-K from heatmap and feed into evaluate_multi_object.

    Constructs a (B, K, 3 + num_classes) tensor where slot 2 is the
    heatmap peak score (already in [0, 1]) and the remaining channels
    are one-hot-ish class logits — same format CenterPredictor produces.
    """
    model.eval()
    num_classes = model.num_classes
    all_outputs: list[torch.Tensor] = []
    all_centers: list[torch.Tensor] = []
    all_classes: list[torch.Tensor] = []
    all_sizes: list[torch.Tensor] = []
    w, h = image_size
    with torch.no_grad():
        for images, anns in loader:
            if images.device != device:
                images = images.to(device, non_blocking=True)
            out = model(images)
            dec = model.decode(out, max_objects=max_objects)
            B = dec.centers_px.shape[0]
            centers_norm = dec.centers_px / torch.tensor(
                [w, h], dtype=torch.float32, device=device,
            )
            class_logits = torch.zeros(
                (B, max_objects, num_classes), dtype=torch.float32, device=device,
            )
            class_logits.scatter_(2, dec.class_ids.unsqueeze(-1), 5.0)
            packed = torch.cat([
                centers_norm,
                dec.scores.unsqueeze(-1),
                class_logits,
            ], dim=-1)
            all_outputs.append(packed.detach().cpu())
            for ann in anns:
                if ann:
                    all_centers.append(torch.tensor(
                        [o["center"] for o in ann], dtype=torch.float32,
                    ))
                    all_classes.append(torch.tensor(
                        [o["shape"] for o in ann], dtype=torch.long,
                    ))
                    all_sizes.append(torch.tensor(
                        [_ann_size_px(o) for o in ann], dtype=torch.float32,
                    ))
                else:
                    all_centers.append(torch.zeros((0, 2), dtype=torch.float32))
                    all_classes.append(torch.zeros((0,), dtype=torch.long))
                    all_sizes.append(torch.zeros((0,), dtype=torch.float32))
    if not all_outputs:
        return {}
    stacked = torch.cat(all_outputs, dim=0)
    metrics = evaluate_multi_object(
        stacked, all_centers, image_size,
        gt_classes_list=all_classes, has_classes=True,
        class_names=class_names,
        gt_sizes_list=all_sizes,
        confidence_threshold=0.1,
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


def evaluate_seg(model, loader, device, num_classes, class_names):
    """Accumulate a confusion matrix over the val set, then mIoU / pixel-acc.

    Predictions are upsampled to the GT mask resolution so the score is at full
    resolution regardless of the model's output stride.
    """
    model.eval()
    k = num_classes + 1  # + background
    conf = torch.zeros((k, k), dtype=torch.long)
    with torch.no_grad():
        for batch in loader:
            images, masks = batch[0], batch[2]
            if images.device != device:
                images = images.to(device, non_blocking=True)
            pred = model.decode(model(images))  # (B, h, w) at the output stride
            if pred.shape[-2:] != masks.shape[-2:]:
                pred = F.interpolate(
                    pred.unsqueeze(1).float(), size=masks.shape[-2:], mode="nearest",
                ).squeeze(1).long()
            conf += segmentation_confusion(pred, masks.to(pred.device), k)
    metrics = evaluate_segmentation(conf, class_names=class_names + ("background",))
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

    train_dc = cfg.train
    val_dc = cfg.val
    train_seed = cfg.seed if cfg.seed != 0 else None
    # The model always has one class slot per shape type; a config that
    # generates only a subset still uses the full class space (so e.g.
    # train-on-rect+circle / test-on-triangle keeps consistent indices).
    num_classes = len(ShapeType)
    class_names = tuple(s.name for s in ShapeType)
    # Segmentation needs per-pixel label maps from the loaders.
    need_masks = cfg.task == "segmentation"

    if cfg.gpu_data:
        if device.type != "cuda":
            raise RuntimeError("gpu_data requires CUDA")
        train_loader: Any = build_gpu_loader(
            train_dc, batch_size=cfg.batch_size, num_images=cfg.num_train_images,
            image_size=img_size, seed=train_seed, device=device,
            reseed_each_epoch=False,  # train draws fresh data every epoch
            with_masks=need_masks,
        )
        # Val must be a FIXED dataset across epochs (reseed_each_epoch=True), or
        # the val metric is computed on different data every epoch.
        val_loader: Any = build_gpu_loader(
            val_dc, batch_size=cfg.batch_size, num_images=cfg.num_val_images,
            image_size=img_size, seed=cfg.val_seed, device=device,
            reseed_each_epoch=True, with_masks=need_masks,
        )
    else:
        pin_memory = device.type == "cuda"
        train_dataset = build_cpu_dataset(
            train_dc, num_images=cfg.num_train_images, image_size=img_size,
            seed=train_seed, transform=transform, with_masks=need_masks,
        )
        val_dataset = build_cpu_dataset(
            val_dc, num_images=cfg.num_val_images, image_size=img_size,
            seed=cfg.val_seed, transform=transform, with_masks=need_masks,
        )
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
    if cfg.task == "single":
        model: torch.nn.Module = SimpleCenterNet(
            num_classes=num_classes,
            encoder_type=encoder_type,
            model_type=model_type,
        )
    elif cfg.task == "heatmap":
        model = CenterHeatmapNet(num_classes=num_classes, stride=cfg.heatmap_stride)
    elif cfg.task == "multi_heatmap":
        model = MultiHeatmapNet(num_classes=num_classes, stride=cfg.heatmap_stride)
    elif cfg.task == "segmentation":
        model = ShapeSegNet(num_classes=num_classes, stride=cfg.seg_stride)
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
        class_match_weight=cfg.class_match_weight,
    )
    heatmap_loss_fn = HeatmapLoss(lambda_class=cfg.lambda_class)
    multi_heatmap_loss_fn = MultiHeatmapLoss(lambda_class=cfg.lambda_class)
    seg_loss_fn = SegLoss()

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
            for batch in train_loader:
                images, anns = batch[0], batch[1]
                masks = batch[2] if len(batch) == 3 else None
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
                elif cfg.task == "multi_heatmap":
                    w, h = img_size
                    scale = torch.tensor([w, h], dtype=torch.float32, device=device)
                    centers_per_image = []
                    classes_per_image = []
                    for ann in anns:
                        if ann:
                            centers_per_image.append(
                                torch.tensor([o["center"] for o in ann],
                                             dtype=torch.float32, device=device) * scale
                            )
                            classes_per_image.append(
                                torch.tensor([o["shape"] for o in ann],
                                             dtype=torch.long, device=device)
                            )
                        else:
                            centers_per_image.append(
                                torch.zeros((0, 2), dtype=torch.float32, device=device)
                            )
                            classes_per_image.append(
                                torch.zeros((0,), dtype=torch.long, device=device)
                            )
                    out = model(images)
                    terms = multi_heatmap_loss_fn(out, centers_per_image, classes_per_image)
                    loss = terms.total
                    sums["hm_heatmap"] += float(terms.heatmap.detach()) * images.size(0)
                    sums["hm_offset"] += float(terms.offset.detach()) * images.size(0)
                    sums["hm_class"] += float(terms.class_.detach()) * images.size(0)
                elif cfg.task == "segmentation":
                    assert masks is not None
                    if masks.device != device:
                        masks = masks.to(device, non_blocking=True)
                    out = model(images)
                    loss = seg_loss_fn(out, masks)
                else:
                    centers_list, classes_list, _ = build_targets_multi(anns, device)
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
            elif cfg.task in ("heatmap", "multi_heatmap"):
                logger.log_scalar("train/hm_heatmap", sums["hm_heatmap"] / max(n_train, 1), step=epoch)
                logger.log_scalar("train/hm_offset", sums["hm_offset"] / max(n_train, 1), step=epoch)
                logger.log_scalar("train/hm_class", sums["hm_class"] / max(n_train, 1), step=epoch)

            if cfg.task == "single":
                vm = evaluate_single(model, val_loader, device, img_size)
            elif cfg.task == "heatmap":
                vm = evaluate_heatmap(model, val_loader, device, img_size)
            elif cfg.task == "multi_heatmap":
                vm = evaluate_multi_heatmap(
                    model, val_loader, device, img_size, class_names, cfg.max_objects,
                )
            elif cfg.task == "segmentation":
                vm = evaluate_seg(model, val_loader, device, num_classes, class_names)
            else:
                vm = evaluate_multi(model, val_loader, device, img_size, class_names)
            logger.log_metrics(vm, step=epoch)
            final_metrics = vm

            interesting_substrings = ("_px", "accuracy", "map", "pearson", "miou", "pixel_acc")
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

        # Optional post-training sweeps over alternate val configurations.
        # We only support these on the multi tasks for now — single-object
        # evaluation has no shape-count dimension to sweep along.
        final_metrics_by_count: dict[int, dict[str, float]] = {}
        final_metrics_by_size: dict[str, dict[str, float]] = {}
        final_metrics_by_overlap: dict[str, dict[str, float]] = {}
        if cfg.task in ("multi", "multi_heatmap") and (cfg.val_shape_counts or cfg.val_shape_sizes or cfg.val_overlaps):
            print("\n=== Post-training sweeps ===")
            for n in cfg.val_shape_counts:
                sweep_loader = _build_sweep_loader(
                    cfg, img_size, device,
                    num_shapes_range=(n, n),
                    shape_size_range=cfg.val.shape_size_range,
                    label=f"count={n}",
                )
                vm_n = (
                    evaluate_multi_heatmap(
                        model, sweep_loader, device, img_size, class_names, cfg.max_objects,
                    )
                    if cfg.task == "multi_heatmap"
                    else evaluate_multi(
                        model, sweep_loader, device, img_size, class_names,
                    )
                )
                final_metrics_by_count[n] = vm_n
                mp = vm_n.get("multi/mean_matched_center_px", 0.0)
                acc = vm_n.get("multi/matched_class_accuracy", 0.0)
                pmap = vm_n.get("multi/map_center", 0.0)
                print(f"  count={n:3d}: mean_px={mp:6.2f}  acc={acc:.3f}  map_center={pmap:.3f}")

            for lo, hi in cfg.val_shape_sizes:
                sweep_loader = _build_sweep_loader(
                    cfg, img_size, device,
                    num_shapes_range=cfg.val.num_shapes_range,
                    shape_size_range=(lo, hi),
                    label=f"size={lo}-{hi}",
                )
                vm_s = (
                    evaluate_multi_heatmap(
                        model, sweep_loader, device, img_size, class_names, cfg.max_objects,
                    )
                    if cfg.task == "multi_heatmap"
                    else evaluate_multi(
                        model, sweep_loader, device, img_size, class_names,
                    )
                )
                key = f"{lo}-{hi}"
                final_metrics_by_size[key] = vm_s
                mp = vm_s.get("multi/mean_matched_center_px", 0.0)
                acc = vm_s.get("multi/matched_class_accuracy", 0.0)
                pmap = vm_s.get("multi/map_center", 0.0)
                print(f"  size={key:>8s}: mean_px={mp:6.2f}  acc={acc:.3f}  map_center={pmap:.3f}")

            for ov in cfg.val_overlaps:
                sweep_loader = _build_sweep_loader(
                    cfg, img_size, device,
                    num_shapes_range=cfg.val.num_shapes_range,
                    shape_size_range=cfg.val.shape_size_range,
                    label=f"overlap={ov:.2f}",
                    max_overlap=ov,
                )
                vm_o = (
                    evaluate_multi_heatmap(
                        model, sweep_loader, device, img_size, class_names, cfg.max_objects,
                    )
                    if cfg.task == "multi_heatmap"
                    else evaluate_multi(
                        model, sweep_loader, device, img_size, class_names,
                    )
                )
                final_metrics_by_overlap[f"{ov:.2f}"] = vm_o
                mp = vm_o.get("multi/mean_matched_center_px", 0.0)
                acc = vm_o.get("multi/matched_class_accuracy", 0.0)
                pmap = vm_o.get("multi/map_center", 0.0)
                print(f"  overlap={ov:.2f}: mean_px={mp:6.2f}  acc={acc:.3f}  map_center={pmap:.3f}")

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
            "final_metrics_by_count": final_metrics_by_count,
            "final_metrics_by_size": final_metrics_by_size,
            "final_metrics_by_overlap": final_metrics_by_overlap,
        }
        results_path = Path(logger.run_dir) / "results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"results: {results_path}")

        # Save the trained model + the config it was trained with, so
        # scripts.eval_sweep can rebuild the model later and run it
        # against arbitrary val configurations without retraining.
        model_path = Path(logger.run_dir) / "model.pth"
        torch.save(model.state_dict(), model_path)
        config_path = Path(logger.run_dir) / "config.json"
        with open(config_path, "w") as f:
            json.dump(asdict(cfg), f, indent=2, default=str)
        print(f"model:   {model_path}")

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
