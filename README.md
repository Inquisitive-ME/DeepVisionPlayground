# DeepVision Playground

An open-source sandbox for studying what it actually takes to **fully solve**
a computer-vision task — not just incrementally improve a metric. Built around
a synthetic shape-detection benchmark so the data is controllable, the task is
gameable, and the architectural choices that matter are visible.

## What this is

The premise (see `claude_project_notes/project_notes.md`) is that real
understanding of vision models comes from picking a clean task, **fully
solving it**, and quantifying which architectural choices were necessary.
This repo gives you:

- A **synthetic shape generator** with knobs for shape type, size, count,
  rotation, color, background, and overlap. Generates fresh data per epoch
  on the GPU so you never overfit and never wait on the dataloader.
- **Four model families** for center-based detection, ranging from a
  ~400 K-parameter heatmap predictor to a ResNet18-backbone variant —
  with a clean `--task` switch between them.
- **Real metrics**: per-pixel-threshold precision/recall, per-class breakdowns,
  per-shape-size buckets, per-image-density buckets, Pearson correlation
  between predicted and true positions, and a `map_center` summary.
- **Post-training sweeps**: take one trained model, eval it against a grid
  of (shape count, shape size) val sets, and read the failure landscape.
- **Real benchmarking infrastructure**: a CLI driver, a pytest harness, a
  GPU monitor, TensorBoard logging, and a `results.json` per run.

## Current state

The single-object task is **fully solved**: ~400 K-param CenterNet-style
heatmap model hits **median 1.8 px localization error** and **99.7%
classification accuracy** in 5 minutes of training on a 3090.

The multi-object task converges to **median 1.55 px** and **97.5% matched
class accuracy** at 0–3 shapes per image; sweeps to higher counts are an
ongoing experiment (results in `claude_project_notes/`).

The most recent learning is documented in
`claude_project_notes/2026-05-01_final_status.md`.

## Quick start

```bash
pip install -r requirements.txt

# Single-object, exact-pixel localization (5 min on a 3090):
python -m scripts.run_training \
    --task heatmap --gpu-data \
    --epochs 1000 --batch-size 100 --image-size 256 \
    --num-train-images 1000 --num-val-images 1000 \
    --lr 1e-4 --heatmap-stride 1

# Multi-object with post-training sweeps:
python -m scripts.run_training \
    --task multi_heatmap --gpu-data \
    --epochs 1000 --batch-size 100 --image-size 256 \
    --num-train-images 1000 --num-val-images 500 \
    --num-shapes-min 0 --num-shapes-max 5 --max-objects 10 \
    --lr 1e-4 \
    --val-shape-counts '1,2,3,5,7,10' \
    --val-shape-sizes '10:30,30:60,60:120,120:180'

# Pretty-print the bucket sweeps from a finished run:
python -m scripts.sweep_buckets runs/<your-run-dir>

# Compare model families against each other:
python -m scripts.benchmark_models --epochs 300
```

TensorBoard: `tensorboard --logdir runs`.

## How the task is set up

Each image is an RGB canvas (default 256×256) drawn from
`data/synthetic_shapes_dataset.ShapeDataset` (CPU+PIL) or, much faster,
`data/gpu_shapes.GpuShapeLoader` (vectorized torch ops on GPU).

Every shape has random:
- type (rectangle / circle / triangle)
- color (any RGB, sampled away from the background)
- size (default 20–90 pixels for multi, 20–128 for single)
- position (uniform over the canvas, with overlap controls)
- rotation (rectangles + triangles, optional)

Backgrounds are random solid colors.

## Model families

| `--task` | Model | What it predicts |
|---|---|---|
| `single` | `SimpleCenterNet` (FC regression) | One `(cx, cy, class_logits)` per image. Fast but capped by encoder stride. |
| `multi` | `CenterPredictor` (FC + slots) | `max_objects` `(cx, cy, conf, class_logits)` slots, Hungarian-matched to GT. |
| `heatmap` | `CenterHeatmapNet` | Per-pixel heatmap + offset + class. Sub-pixel localization via heatmap argmax + offset. |
| `multi_heatmap` | `MultiHeatmapNet` | Same as heatmap but with top-K NMS decode for multi-object. |

For heatmaps: `--heatmap-stride 4` is fastest, `1` is exact-pixel.

## Encoders

- `simple` / `simple_bn`: 4-conv stack, stride 16, 128 final channels.
- `simple_gap` / `simple_bn_gap`: same but global-average-pooled. Position-
  invariant, only useful for classification, not localization.
- `resnet18` / `resnet34`: torchvision backbones, GAP'd. Same caveat.
- `resnet18_spatial` / `resnet34_spatial`: torchvision backbones with the
  AvgPool stripped, ~32 K spatial features.

ResNet runs need `lr=1e-5` (~10× smaller than the simple-encoder default)
to avoid the initial-update overshoot.

## Metrics

Reported per training run via `--task heatmap` and `multi_heatmap`:

- **`mean_center_px`, `median_center_px`**: pixel-distance between predicted
  and true centers.
- **`pearson_cx`, `pearson_cy`**: across all val images, do the predictions
  track the true positions? A model that always predicts (0.5, 0.5)
  scores 0; a perfect model scores 1.
- **`accuracy`** (single) / **`matched_class_accuracy`** (multi): classification
  on the matched / per-image prediction.
- **`recall@T px`** at thresholds {2, 4, 8, 16}: detection rate.
- **`map_center`**: mean of `precision*recall` across pixel thresholds.
- **`by_size/{xs,sm,md,lg}/...`**: same metrics restricted to GT shapes
  in each size bucket.
- **`by_count/{n1,n2,n3-5,...}/...`**: restricted to images with n GTs.

`scripts/sweep_buckets.py runs/<run>` pretty-prints them.

## Layout

```
data/                 ShapeDataset, GpuShapeLoader, annotations
models/               SimpleCenterNet, CenterPredictor, *HeatmapNet, encoders
utils/                losses, metrics, training_logger, perf
scripts/              run_training, benchmark_models, sweep_buckets, gpu_monitor
training/             original imperative training scripts (legacy; prefer scripts/run_training)
tests/                pytest suite (~50 tests, ~2 s on CPU)
claude_project_notes/ design notes and per-session status writeups
```

## Status notes

The `claude_project_notes/` directory contains dated writeups of what was
tried, what worked, and what didn't. Particularly useful:

- `2026-05-01_gpu_and_training_analysis.md` — early diagnosis of why the
  baseline FC model wasn't converging (turned out to be a regression I
  introduced — `sigmoid` on a regression target whose mean is 0.5).
- `2026-05-01_solved.md` — first time the single-object task hit pixel
  accuracy.
- `2026-05-01_final_status.md` — final numbers across all four model
  families, with the architectural choices that mattered.

## Contributing

This is a research sandbox; pick a task you care about, add a new model
family or a new dataset configuration, and PR with a `claude_project_notes/`
writeup of what you tried.

Pre-commit hooks (`.pre-commit-config.yaml`) run ruff, flake8, and mypy.
Tests via `pytest`.
