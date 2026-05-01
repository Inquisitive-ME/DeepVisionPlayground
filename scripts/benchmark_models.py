"""Run a fixed set of (model, task) configurations and collect results.

Drives ``scripts.run_training`` as a subprocess so each config gets its
own clean process state, then reads the per-run ``results.json`` files
into a single pretty comparison table.

Usage:
    python -m scripts.benchmark_models                # default sweep
    python -m scripts.benchmark_models --epochs 200   # all configs at 200 epochs

Each config is a single CLI command; tweak ``CONFIGS`` below to change
the sweep.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    name: str
    args: list[str]


# Curated sweep — single-object regression baseline, single-object heatmap
# at multiple strides, and multi-object FC vs heatmap.
CONFIGS: list[Config] = [
    Config(
        name="single-fc-simple",
        args=["--task", "single", "--encoder", "simple"],
    ),
    Config(
        name="single-heatmap-stride4",
        args=["--task", "heatmap", "--heatmap-stride", "4"],
    ),
    Config(
        name="single-heatmap-stride2",
        args=["--task", "heatmap", "--heatmap-stride", "2"],
    ),
    Config(
        name="multi-fc-simple",
        args=[
            "--task", "multi", "--encoder", "simple",
            "--num-shapes-min", "0", "--num-shapes-max", "3",
            "--max-objects", "5", "--lambda-class", "3.0",
        ],
    ),
    Config(
        name="multi-heatmap-stride4",
        args=[
            "--task", "multi_heatmap",
            "--num-shapes-min", "0", "--num-shapes-max", "3",
            "--max-objects", "5",
        ],
    ),
]


def run_one(cfg: Config, common: list[str]) -> dict | None:
    cmd = [
        sys.executable, "-m", "scripts.run_training",
        *cfg.args, *common,
    ]
    print(f"\n=== {cfg.name} ===")
    print(" ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    last_lines = proc.stdout.splitlines()[-3:]
    for line in last_lines:
        print(line)
    if proc.returncode != 0:
        print(f"!! failed with exit {proc.returncode}")
        sys.stderr.write(proc.stderr[-2000:])
        return None
    # Find the results.json mentioned in the last lines.
    for line in proc.stdout.splitlines()[::-1]:
        if line.startswith("results: "):
            results_path = Path(line.split(":", 1)[1].strip())
            with open(results_path) as f:
                return json.load(f)
    return None


def fmt_metric(d: dict | None, key: str) -> str:
    if d is None:
        return "—"
    val = d.get("final_metrics", {}).get(key)
    if val is None:
        return "—"
    return f"{val:.3f}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--num-train-images", type=int, default=1000)
    p.add_argument("--num-val-images", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=100)
    args = p.parse_args()
    common = [
        "--gpu-data",
        "--epochs", str(args.epochs),
        "--num-train-images", str(args.num_train_images),
        "--num-val-images", str(args.num_val_images),
        "--lr", str(args.lr),
        "--batch-size", str(args.batch_size),
    ]
    results: list[tuple[Config, dict | None]] = []
    for cfg in CONFIGS:
        results.append((cfg, run_one(cfg, common)))

    # Pretty-print a Markdown-ish table.
    print()
    print(f"=== Benchmark summary ({args.epochs} epochs each) ===")
    print()
    rows = [
        ("config", "n_params", "mean_px", "median_px",
         "accuracy", "pearson_cx", "pearson_cy", "map_center"),
    ]
    for cfg, r in results:
        if r is None:
            rows.append((cfg.name, "—", "—", "—", "—", "—", "—", "—"))
            continue
        is_multi = cfg.args[1].startswith("multi")
        if is_multi:
            mean_key = "multi/mean_matched_center_px"
            med_key = "multi/median_matched_center_px"
            acc_key = "multi/matched_class_accuracy"
            cx_key = "multi/pearson_cx"
            cy_key = "multi/pearson_cy"
            map_key = "multi/map_center"
        else:
            mean_key = "single/mean_center_px"
            med_key = "single/median_center_px"
            acc_key = "single/accuracy"
            cx_key = "single/pearson_cx"
            cy_key = "single/pearson_cy"
            map_key = ""  # n/a
        rows.append((
            cfg.name,
            f"{r.get('n_params', 0):,}",
            fmt_metric(r, mean_key),
            fmt_metric(r, med_key),
            fmt_metric(r, acc_key),
            fmt_metric(r, cx_key),
            fmt_metric(r, cy_key),
            fmt_metric(r, map_key) if map_key else "—",
        ))
    widths = [max(len(str(c)) for c in col) for col in zip(*rows)]
    for row in rows:
        print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
