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
  on the GPU so you never overfit and never wait on the dataloader — and emits
  pixel-perfect segmentation masks for free (the generator is the labeller).
- **Seven tasks** behind one `--task` switch: center detection (`single`,
  `multi`, `heatmap`, `multi_heatmap`), semantic `segmentation`,
  `instance_seg`, and whole-image `classification` — covering localize /
  classify / segment from the vision.
- **Real metrics** per task: detection P/R + `map_center` AP (with per-class,
  per-size, per-density breakdowns and Pearson), segmentation mIoU / per-class
  IoU / pixel accuracy, instance matched-IoU / recall@IoU, and classification
  accuracy.
- **Post-training sweeps**: take one trained model, eval it against a grid
  of (shape count, shape size) val sets, and read the failure landscape.
- **Real benchmarking infrastructure**: a CLI driver, a pytest harness, a
  GPU monitor, TensorBoard logging, and a `results.json` per run.

## Current state

The single-object task is **fully solved**: ~400 K-param CenterNet-style
heatmap model hits **median 1.8 px localization error** and **99.7%
classification accuracy** in 5 minutes of training on a 3090.

The multi-object task localizes **matched** objects to **median ~1.5 px** with
**~97% matched-class accuracy** at 0–3 shapes per image. Read those as what they
are: *matched-subset* metrics that, by construction, ignore false positives and
missed objects. The honest "is it actually solved" number is **`map_center`** —
a confidence-integrated average precision over pixel thresholds — together with
`cardinality_error`; both are markedly less flattering than the matched numbers,
and closing that gap (and sweeping to higher counts) is an ongoing experiment
(results in `claude_project_notes/`).

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

## Distribution-shift studies

The point of the sandbox is to train on one distribution and measure on
another. A study is **declared in a config file**: run/model scalars at the
top level, and `train:` / `val:` blocks that each set any dataset knob
(rotation, noise, background, outline, sizes, counts, shape types, overlap).
`val` inherits `train` and overrides only what shifts — there are no
per-augmentation CLI flags to memorize, and the study is reproducible from one
file. See `configs/` for the schema and worked examples.

```bash
# Train WITHOUT rotation, validate WITH it (the val set is the shift):
python -m scripts.run_training --config configs/train_clean_eval_rotated.yaml
```

```yaml
# configs/train_clean_eval_rotated.yaml (excerpt)
task: heatmap
train:
  rotate_shapes: false   # trained on axis-aligned shapes
val:
  rotate_shapes: true    # measured on rotated shapes; everything else inherits train
```

Because `val` is evaluated every epoch, the train-vs-shifted-val curve *is* the
study. To then close the gap, flip `train.rotate_shapes` to `true` and rerun.

Two other ways to run a shift study:

- **Post-hoc sweeps** — `scripts.eval_sweep` loads a saved model and sweeps one
  val-time perturbation across several values, holding everything else at the
  model's training distribution:
  `python -m scripts.eval_sweep --run-dir runs/<run> --rotate false,true --noise false,true`.
- **By hand in Python** — `examples/clean_train_rotated_val.py` composes the
  train/val datasets and loop directly, for full control without a config.

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

## Tasks (`--task`)

| `--task` | Model | What it predicts | Metric |
|---|---|---|---|
| `single` | `SimpleCenterNet` (FC regression) | One `(cx, cy, class)` per image. Fast but capped by encoder stride. | center px, accuracy |
| `multi` | `CenterPredictor` (FC + slots) | `max_objects` `(cx, cy, conf, class)` slots, Hungarian-matched to GT. | `map_center` AP, matched px |
| `heatmap` | `CenterHeatmapNet` | Per-pixel heatmap + offset + class. Sub-pixel localization via heatmap argmax + offset. | center px, accuracy |
| `multi_heatmap` | `MultiHeatmapNet` | Same as heatmap but with top-K NMS decode for multi-object. | `map_center` AP, matched px |
| `segmentation` | `ShapeSegNet` | Per-pixel class (background + each shape type). Encoder-decoder, argmax decode. | mIoU, per-class IoU, pixel acc |
| `classification` | `ShapeClassifier` | The class of the single shape (no localization). Encoder + linear head. | accuracy |
| `instance_seg` | `InstanceSegNet` | Per-pixel class + a center heatmap; foreground pixels grouped to the nearest detected center → per-shape instances. | matched IoU, recall@IoU |

For heatmaps: `--heatmap-stride 4` is fastest, `1` is exact-pixel. For
segmentation/instance_seg: `--seg-stride 1` is full-resolution (target mIoU→1.0).

```bash
# Semantic segmentation (free pixel-perfect masks; ~400 K params):
python -m scripts.run_training --task segmentation --gpu-data \
    --epochs 500 --batch-size 100 --image-size 256 \
    --num-shapes-min 1 --num-shapes-max 5 --seg-stride 1 --lr 1e-3

# Instance segmentation (separates individual shapes via center grouping):
python -m scripts.run_training --task instance_seg --gpu-data \
    --epochs 500 --batch-size 64 --image-size 128 \
    --num-shapes-min 1 --num-shapes-max 5 --max-objects 8 --seg-stride 2 --lr 2e-3

# Whole-image shape classification (GAP encoder is ideal here):
python -m scripts.run_training --task classification --gpu-data \
    --encoder simple_gn_gap --epochs 200 --image-size 128
```

## Encoders

- `simple_gn` (**default**) / `simple` / `simple_bn`: 4-conv stack, stride 16,
  128 final channels. `simple_gn` uses GroupNorm, which is per-sample and so
  behaves identically in train and eval — the right choice for honest
  distribution-shift studies. `simple_bn` (BatchNorm) and the `resnet*`
  encoders carry running statistics that are frozen at eval, which biases
  predictions on shifted val data; **don't use them for robustness or
  "which component is necessary" comparisons.**
- `simple_gn_gap` / `simple_gap` / `simple_bn_gap`: same but global-average-
  pooled. Position-invariant, only useful for classification, not localization.
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
- **`map_center`**: confidence-integrated average precision (area under the
  precision–recall curve, averaged over pixel thresholds). Independent of the
  confidence threshold, so it is comparable across model families (e.g. FC
  multi vs heatmap, whose raw confidence scores live on different scales).
- **`by_size/{xs,sm,md,lg}/...`**: same metrics restricted to GT shapes
  in each size bucket.
- **`by_count/{n1,n2,n3-5,...}/...`**: restricted to images with n GTs.

`scripts/sweep_buckets.py runs/<run>` pretty-prints them.

Segmentation (`--task segmentation`) instead reports:

- **`seg/miou`**: mean IoU over the classes present (the "is it solved" number).
- **`seg/iou/<class>`**: per-class IoU, including `background`.
- **`seg/pixel_acc`**: fraction of correctly-classified pixels.

Classification (`--task classification`) reports **`classification/accuracy`**.

Instance segmentation (`--task instance_seg`) reports **`instance/mean_iou`**
(matched-IoU averaged over GT instances) and **`instance/recall@{0.5,0.75}`**
(fraction of GT instances matched at that IoU).

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
