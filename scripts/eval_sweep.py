"""Load a trained model from a run directory and sweep one perturbation
dimension across val sets, printing a comparison table.

Each sweep flag takes a comma-separated list of values; we build a fresh
val loader for each value (with everything else held at the training
defaults) and run the same evaluation pipeline as run_training. Results
are appended to ``results.json`` under ``eval_sweeps/<dimension>``.

Available sweep dimensions:

  --counts   "1,3,5,10"             num_shapes_range = (n, n)
  --sizes    "10:30,30:60"          shape_size_range = (lo, hi)
  --overlaps "0.1,0.6,0.9"          max_overlap
  --rotate   "true,false"           rotate_shapes
  --backgrounds "solid,texture"     BackgroundType
  --noise    "false,true"           add_noise
  --outlines "fill,thin,thick,random"  ShapeOutline
  --color-thresholds "20,50,80"     min RGB distance from background

You can specify multiple sweep dimensions in one call; they are run
sequentially (no Cartesian product) so the cost stays small.

Usage:

    python -m scripts.eval_sweep --run-dir runs/<dir> --counts 1,3,5,10
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.annotations import BackgroundType, ShapeOutline, ShapeType
from data.synthetic_shapes_dataset import ShapeDataset
from models.center_heatmap_net import CenterHeatmapNet
from models.encoders import EncodeType
from models.multi_heatmap_net import MultiHeatmapNet
from models.multiple_center_predictor import CenterPredictor
from models.simple_center_net import SimpleCenterNet
from models.types import ModelType


# Per-dimension defaults so a sweep that doesn't specify the field uses
# the training value. We read these from the saved config when possible.
@dataclass
class _ValDefaults:
    image_size: tuple[int, int]
    num_shapes_range: tuple[int, int]
    shape_size_range: tuple[int, int]
    rotate_shapes: bool
    background: BackgroundType
    add_noise: bool
    shape_outline: ShapeOutline
    max_overlap: float
    color_threshold: float
    num_val_images: int
    val_seed: int
    batch_size: int
    max_objects: int
    task: str


def _load_run(run_dir: Path) -> tuple[torch.nn.Module, dict[str, Any], _ValDefaults, torch.device]:
    """Reconstruct model + return its training config."""
    cfg_path = run_dir / "config.json"
    model_path = run_dir / "model.pth"
    if not cfg_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"{run_dir} does not have config.json + model.pth — was the run "
            f"completed with a recent run_training that saves them?"
        )
    with open(cfg_path) as f:
        cfg = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = (int(cfg["image_size"]), int(cfg["image_size"]))
    num_classes = len(ShapeType)
    encoder_type = EncodeType[cfg["encoder"]]

    if cfg["task"] == "single":
        model: torch.nn.Module = SimpleCenterNet(
            num_classes=num_classes,
            encoder_type=encoder_type,
            model_type=ModelType.center_localization_and_class_id,
        )
    elif cfg["task"] == "heatmap":
        model = CenterHeatmapNet(
            num_classes=num_classes, stride=int(cfg["heatmap_stride"]),
        )
    elif cfg["task"] == "multi_heatmap":
        model = MultiHeatmapNet(
            num_classes=num_classes, stride=int(cfg["heatmap_stride"]),
        )
    else:  # multi
        model = CenterPredictor(
            num_classes=num_classes,
            model_type=ModelType.center_localization_and_class_id,
            encoder_type=encoder_type,
            max_objects=int(cfg["max_objects"]),
            hidden_dims=cfg.get("hidden_dims") or None,
        )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    is_single = cfg["task"] in ("single", "heatmap")
    defaults = _ValDefaults(
        image_size=img_size,
        num_shapes_range=(1, 1) if is_single else (
            int(cfg["num_shapes_min"]), int(cfg["num_shapes_max"]),
        ),
        shape_size_range=(20, 128) if is_single else (20, 90),
        rotate_shapes=is_single,
        background=BackgroundType.SOLID,
        add_noise=False,
        shape_outline=ShapeOutline.FILL,
        max_overlap=0.6,
        color_threshold=50.0,
        num_val_images=int(cfg["num_val_images"]),
        val_seed=int(cfg["val_seed"]),
        batch_size=int(cfg["batch_size"]),
        max_objects=int(cfg["max_objects"]),
        task=cfg["task"],
    )
    return model, cfg, defaults, device


def _build_loader(d: _ValDefaults, device: torch.device) -> Any:
    """Build a CPU val loader that respects every override.

    We always use the CPU dataset path for sweeps because the GPU loader
    doesn't support backgrounds / noise / outlines / color thresholds /
    overlap; the val pass is small and one-shot, so the speed difference
    isn't material here.
    """
    ds = ShapeDataset(
        num_images=d.num_val_images,
        seed=d.val_seed,
        image_size=d.image_size,
        num_shapes_range=d.num_shapes_range,
        shape_size_range=d.shape_size_range,
        shape_types=tuple(ShapeType),
        background=d.background,
        shape_outline=d.shape_outline,
        rotate_shapes=d.rotate_shapes,
        max_overlap=d.max_overlap,
        add_noise=d.add_noise,
        transform=transforms.ToTensor(),
    )
    return DataLoader(
        ds, batch_size=d.batch_size, shuffle=False,
        collate_fn=ShapeDataset.collate_function,
        num_workers=0, pin_memory=device.type == "cuda",
    )


def _evaluate(model: torch.nn.Module, loader: Any, defaults: _ValDefaults,
              device: torch.device) -> dict[str, float]:
    """Run the right evaluator for the model's task."""
    # Lazy imports keep the module tree decoupled in the docstring.
    from scripts.run_training import (
        evaluate_heatmap,
        evaluate_multi,
        evaluate_multi_heatmap,
        evaluate_single,
    )
    class_names = tuple(s.name for s in ShapeType)
    if defaults.task == "single":
        return evaluate_single(model, loader, device, defaults.image_size)
    if defaults.task == "heatmap":
        return evaluate_heatmap(model, loader, device, defaults.image_size)
    if defaults.task == "multi_heatmap":
        return evaluate_multi_heatmap(
            model, loader, device, defaults.image_size, class_names, defaults.max_objects,
        )
    return evaluate_multi(model, loader, device, defaults.image_size, class_names)


def _sweep(model: torch.nn.Module, defaults: _ValDefaults, device: torch.device,
           dim_name: str, perturbations: list[tuple[str, dict[str, Any]]]) -> dict[str, dict[str, float]]:
    """Run one eval per perturbation, return a dict keyed by label."""
    results: dict[str, dict[str, float]] = {}
    print(f"\n=== sweep: {dim_name} ===")
    for label, overrides in perturbations:
        # Build a per-perturbation defaults instance.
        from dataclasses import replace
        d = replace(defaults, **overrides)
        loader = _build_loader(d, device)
        vm = _evaluate(model, loader, d, device)
        # Pick the right metric subset for the task.
        is_multi = defaults.task in ("multi", "multi_heatmap")
        med = vm.get("multi/median_matched_center_px" if is_multi else "single/median_center_px", 0.0)
        mean = vm.get("multi/mean_matched_center_px" if is_multi else "single/mean_center_px", 0.0)
        acc = vm.get("multi/matched_class_accuracy" if is_multi else "single/accuracy", 0.0)
        pcx = vm.get("multi/pearson_cx" if is_multi else "single/pearson_cx", 0.0)
        print(f"  {label:>20s}: median_px={med:6.2f}  mean_px={mean:6.2f}  acc={acc:.3f}  pearson_cx={pcx:.3f}")
        results[label] = vm
    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None,
                   help="Write the sweep results here (default: <run-dir>/eval_sweeps.json)")
    p.add_argument("--counts", type=str, default="")
    p.add_argument("--sizes", type=str, default="")
    p.add_argument("--overlaps", type=str, default="")
    p.add_argument("--rotate", type=str, default="")
    p.add_argument("--backgrounds", type=str, default="")
    p.add_argument("--noise", type=str, default="")
    p.add_argument("--outlines", type=str, default="")
    p.add_argument("--color-thresholds", type=str, default="")
    args = p.parse_args()

    model, cfg, defaults, device = _load_run(args.run_dir)
    print(f"loaded model from {args.run_dir} (task={cfg['task']}, encoder={cfg['encoder']})")
    print(f"defaults: {defaults}")

    sweeps: dict[str, dict[str, dict[str, float]]] = {}

    Pert = tuple[str, dict[str, Any]]

    if args.counts:
        count_perts: list[Pert] = [
            (f"n={n}", {"num_shapes_range": (int(n), int(n))})
            for n in args.counts.split(",") if n
        ]
        sweeps["counts"] = _sweep(model, defaults, device, "shape count", count_perts)

    if args.sizes:
        size_perts: list[Pert] = []
        for s in args.sizes.split(","):
            if not s:
                continue
            lo, hi = s.split(":")
            size_perts.append((f"{lo}-{hi}px", {"shape_size_range": (int(lo), int(hi))}))
        sweeps["sizes"] = _sweep(model, defaults, device, "shape size", size_perts)

    if args.overlaps:
        overlap_perts: list[Pert] = [
            (f"o={o}", {"max_overlap": float(o)})
            for o in args.overlaps.split(",") if o
        ]
        sweeps["overlaps"] = _sweep(model, defaults, device, "max overlap", overlap_perts)

    if args.rotate:
        rotate_perts: list[Pert] = []
        for v in args.rotate.split(","):
            v = v.strip().lower()
            if v in ("true", "1", "yes"):
                rotate_perts.append(("rotate=true", {"rotate_shapes": True}))
            elif v in ("false", "0", "no"):
                rotate_perts.append(("rotate=false", {"rotate_shapes": False}))
        sweeps["rotate"] = _sweep(model, defaults, device, "rotate_shapes", rotate_perts)

    if args.backgrounds:
        bg_perts: list[Pert] = []
        for v in args.backgrounds.split(","):
            v = v.strip().lower()
            try:
                bg = BackgroundType[v.upper()]
            except KeyError:
                raise SystemExit(f"unknown background {v!r}") from None
            bg_perts.append((v, {"background": bg}))
        sweeps["backgrounds"] = _sweep(model, defaults, device, "background", bg_perts)

    if args.noise:
        noise_perts: list[Pert] = []
        for v in args.noise.split(","):
            v = v.strip().lower()
            if v in ("true", "1", "yes"):
                noise_perts.append(("noise=true", {"add_noise": True}))
            elif v in ("false", "0", "no"):
                noise_perts.append(("noise=false", {"add_noise": False}))
        sweeps["noise"] = _sweep(model, defaults, device, "add_noise", noise_perts)

    if args.outlines:
        outline_perts: list[Pert] = []
        for v in args.outlines.split(","):
            v = v.strip().lower()
            try:
                ol = ShapeOutline[v.upper()]
            except KeyError:
                raise SystemExit(f"unknown outline {v!r}") from None
            outline_perts.append((v, {"shape_outline": ol}))
        sweeps["outlines"] = _sweep(model, defaults, device, "shape_outline", outline_perts)

    if args.color_thresholds:
        # color threshold isn't a ShapeDataset constructor arg — it's
        # baked into select_shape_color. Plumbing it through would be a
        # separate change; for now we just refuse with a clear message.
        print("(color-thresholds sweep not yet wired through ShapeDataset; skipped)")

    if not sweeps:
        print("no sweeps requested. pass at least one of --counts/--sizes/--overlaps/--rotate/"
              "--backgrounds/--noise/--outlines.")
        return 0

    out_path = args.out or (args.run_dir / "eval_sweeps.json")
    with open(out_path, "w") as f:
        json.dump(sweeps, f, indent=2, default=str)
    print(f"\nresults: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
