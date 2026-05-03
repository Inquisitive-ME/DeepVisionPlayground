# Compaction note — read this first

Date: 2026-05-03
Purpose: durable record of this session for continuation after context
compaction. If you (or a future Claude) are reading this, **read this
file first** — it's the canonical map of what was done, what works,
what was learned, and what's still open.

## Where to look in the repo

| Path | What's there |
|---|---|
| `README.md` | Repo overview, quick-start. Already rewritten this session. |
| `claude_project_notes/HOW_TO_RUN.md` | Verbose practical guide. CLI flags, debugging cheats. |
| `claude_project_notes/2026-05-01_*.md` | Per-experiment writeups. Most recent first below. |
| `claude_project_notes/2026-05-02_more_data_per_epoch.md` | The most important late-session result. |
| `models/` | All four model families. |
| `utils/` | `losses`, `heatmap_loss`, `metrics`, `training_logger`, `perf`. |
| `scripts/` | `run_training`, `eval_sweep`, `sweep_buckets`, `benchmark_models`, `gpu_monitor`. |
| `tests/` | 51 pytest cases, ~2 s on CPU. |
| `runs/` | Per-run TB logs + `model.pth` + `config.json` + `results.json` + `eval_sweeps.json`. |

## Where the project started (before this work)

The user is studying "what does it actually take to fully solve a CV
task" via a synthetic shape-detection benchmark. Goals:
- Generate unlimited fresh data with controllable knobs.
- Find the smallest model that fully solves shape detection.
- Build a benchmarking platform for architecture ablations.

Initial repo: a CPU PIL dataset (`ShapeDataset`), three imperative
training scripts (`train_single_center`, `train_multi_center`,
`train_faster_rcnn`), an FC-regression model
(`SimpleCenterNet`/`CenterPredictor`), and a slot-based multi-object
loss (`CenterPredictionLoss`). The user reported "the small custom
model isn't doing as well as expected."

## What the session actually delivered

47 commits across two days. Three logical phases.

### Phase 1: bug-hunt and solve the single-object task (commits 1–18)

Diagnosed and fixed several bugs the previous "code-quality" commits
had introduced or papered over:

1. **Sigmoid on regression outputs** in `SimpleCenterNet` and
   `CenterPredictor`. Targets in `[0, 1]` are roughly symmetric around
   0.5, sigmoid output starts at 0.5, per-batch gradient signal
   cancels — the model crawled to Pearson 0.4 over hundreds of epochs.
   Reverted both. (`bc548e1`, `a5c709a`)
2. **`BCE(zeros, ones)` term** in the multi-object loss for "missed
   detections." Both inputs constants → no gradient, but inflated the
   loss by ~100 per missed object and broke `ReduceLROnPlateau`.
   Removed. (`5474f49`)
3. **Hungarian cost subtracted `0.1 * conf`** which rewarded the
   model for confidently asserting every slot. Removed. (`5474f49`)
4. **Two distinct `ModelType` enums** in different modules with the
   same names. Comparisons across modules silently failed. Unified.
   (`5b366f1`)
5. **`fixed_dataset` length mismatch** silently returned wrong
   `__len__`. Now raises. (`5474f49`)
6. **GPU was severely underutilized** at 9.8% mean. Built a vectorized
   GPU shape rasterizer (`data/gpu_shapes.py`) — 3× throughput.
   (`ac08cca`)
7. **Pearson correlation, decomposed loss logging, per-index
   deterministic val** added to make progress actually visible.
   (`51e2b6d`)
8. **The architectural finding**: FC-regression is bounded by encoder
   stride. The 16×16 feature grid means one cell is 16 pixels — no
   gradient signal finer than that. Built `CenterHeatmapNet` (single)
   and `MultiHeatmapNet` (multi), CenterNet-style with focal loss +
   sub-pixel offset + GroupNorm. (`7d23621`, `6d2659e`)

After phase 1: single-object solved (median 1.8 px / 99.7% accuracy),
multi-object solved (median 1.55 px / 97.5% accuracy).

### Phase 2: build the benchmarking platform (commits 19–34)

After the task was solvable, the focus shifted to "how do we study
which choices matter?":

- **Bucketed metrics**: `evaluate_multi_object` gained per-class,
  per-shape-size, and per-image-density breakdowns. (`ac1b922`)
- **`scripts/benchmark_models.py`**: runs 5 reference configs and
  prints a comparison table. (`ca3ac3a`)
- **`scripts/sweep_buckets.py`**: pretty-prints the bucket metrics
  from any results.json. (`52f71e6`)
- **In-training sweep flags**: `--val-shape-counts`, `--val-shape-sizes`,
  `--val-overlaps`. Convenient but conflated training and evaluation.
  (`d2db8b2`, `9901db6`)
- **HOW_TO_RUN.md**: verbose user guide. (`7fa72d3`)
- **README rewrite**: real description of the four model families,
  quick-start commands, pointers to the per-session notes. (`52f71e6`)

### Phase 3: augmentation studies and the eval_sweep tool (commits 35–47)

User pushed back: post-hoc evaluation should be a separate tool, not
bolted onto training. That refactor was the big late-session change:

- **`run_training` saves `model.pth` + `config.json`** alongside
  `results.json`. (`4479c10`)
- **`scripts/eval_sweep.py`**: loads any trained run, sweeps one
  perturbation dimension at a time across rotation, backgrounds, noise,
  outlines, counts, sizes, overlaps. Writes `eval_sweeps.json`.
  (`4479c10`)
- **Training-time augmentation flags**: `--train-background`,
  `--train-outline`, `--train-add-noise`, `--train-shape-size`. (`c224ba0`)
- **First augmentation study**: trained 300 ep heatmap model on clean
  config, swept 5 dimensions. Found two catastrophic failure modes
  (texture backgrounds, hollow outlines). (`35e4acb`)
- **Followup study**: trained 300 ep with augmentations on, re-swept;
  showed the two failure modes vanished but with a 16 pp accuracy
  cost on solid+fill. (`c7622a4`)
- **Converged 1000-epoch comparison**: trained both at 1000 ep and
  reswept. Confirmed texture/outline failures are *structural* in the
  baseline (don't fix with more training) but the augmented model has
  a 20 pp tax on the easy case. (`7c19fc4`)
- **The "20 pp tax was undertraining" result**: bumping
  `--num-train-images` from 1000 to 5000 (2.5× total samples) closed
  the gap from 20 pp to 8 pp. Augmented model now strictly beats the
  baseline on every cross-distribution perturbation by +47 to +62 pp.
  (`06392b5`)

## The numbers that matter

### Single-object, 1000 epochs, lr=1e-4

| model | params | median_px | accuracy |
|---|---:|---:|---:|
| FC `SimpleCenterNet` (`simple` encoder) | 8.4M | 14.7 | 0.81 |
| `CenterHeatmapNet` stride=4 | 405k | 2.3 | 0.997 |
| `CenterHeatmapNet` stride=2 | 422k | 2.1 | 0.972 |
| `CenterHeatmapNet` stride=1 | 438k | **1.8** | 0.96 |

### Multi-object, 1000 epochs, 0–5 shapes, lr=1e-4

| model | matched median_px | matched class_acc | map_center |
|---|---:|---:|---:|
| `CenterPredictor` (FC slots, lambda_class=3) | 38 | 0.44 | 0.005 |
| `MultiHeatmapNet` stride=4 | **1.55** | 0.975 | 0.44 |

### Distribution-shift study (1000-epoch heatmap models, eval_sweep)

| val perturbation | Baseline | Aug-1M (1k×1k) | Aug-2.5M (5k×500) |
|---|---:|---:|---:|
| solid + fill (in-dist for baseline) | **99.9%** | 79.9% | 91.6% |
| texture background | 44.3% | 82.1% | **90.9%** |
| thin outline | 35.5% | 89.4% | **95.8%** |
| thick outline | 36.6% | 93.2% | **99.0%** |
| 10–30 px shape | 66.8% | 71.5% | **91.1%** |
| 120–200 px shape | 97.6% | 67.4% | 72.2% |

**Headlines**:

- The texture and hollow-outline failure modes are *structural* in the
  baseline. They don't fix with more training (1000 ep doesn't help).
- They DO fix with training-time augmentation, given enough total
  samples. At 2.5M samples the augmented model essentially solves
  every cross-distribution perturbation tested.
- The remaining 8 pp gap on solid+fill is almost certainly more
  undertraining (the augmented training-val accuracy was still
  climbing at 500 epochs of 5000 imgs).

### One failure mode neither fixes

**Large-shape localization** (120–200 px): both 1000-epoch models
still fail. ~10–13 px median error, ~67–98% accuracy. Wider training
range did NOT help. **This is architectural, not a data-distribution
issue**: a heatmap can't naturally place a peak in the empty interior
of a large filled rectangle. Fixing it would need a different output
target (mask, bbox, or extent regression).

## Settled conclusions

1. **Output formulation > parameter count for localization**. A 405k
   heatmap beats an 8.4M FC regressor by 6× on median error. FC
   regression is bounded by encoder stride.

2. **Most "convergence" issues are loss-shape bugs**. Sigmoid on
   regression targets, BCE on dense Gaussians, BN running stats on
   fresh-data — every single one had to be fixed before the model
   could solve the task. Track Pearson and decomposed loss; the
   headline scalar lies.

3. **`num_train_images` is the actual training-budget knob**, not
   `epochs`. Fresh-per-call data generation makes "epoch" a vestigial
   unit. Bumping `--num-train-images` from 1000 to 5000 closed a
   12 pp accuracy gap on a wider augmented distribution.

4. **Distribution shift maps to architectural cues**. The model
   trained on filled shapes uses interior color as a primary class
   signal — that's why hollow shapes catastrophically fail. Augment
   the right things at training time and the failures vanish.

5. **Some failure modes are architectural, not data**. Large-shape
   localization with a centerness-based heatmap is the example here —
   no amount of wider training fixes it because the geometric center
   of a uniform region carries no distinctive feature.

## Open questions to attack next

In rough priority order:

1. **Run augmented at 5M total samples** (10000 imgs/ep × 500 ep, or
   5000 imgs/ep × 1000 ep). Does the remaining 8 pp gap on solid+fill
   close? My bet: yes, with maybe 2–3 pp residual.

2. **Fix the large-shape failure mode**. Two paths:
   - Add a bbox / extent regression head that predicts shape size
     alongside center; the loss can then focus on locating extent
     rather than pure peak.
   - Predict a binary mask of the shape and take its center of mass.
     Cleaner but a real architecture change.

3. **Encoder ablation** with the existing eval_sweep harness. Does
   ResNet18-spatial beat the simple encoder on these tasks now that
   the lr is right? At what train budget does each saturate?

4. **Make the GPU loader support textures + outlines + noise**. Right
   now augmentation experiments fall back to the 3× slower CPU PIL
   path. Wiring up texture / outline rendering on GPU would speed up
   augmented training 3× — the experiments above would all run in
   ~10 min instead of 30–60.

5. **Multi-object with augmentations**. We did the augmentation study
   on single-object. Does the multi-object model show the same
   pattern? Plausibly yes but worth verifying.

6. **Color-isoluminance attack**. Lower the shape/background contrast
   threshold and see if the heatmap collapses. The current model has
   only seen contrast > 50 pixels in RGB distance.

7. **Scale invariance**. Train at 256, eval at 128 / 384 / 512.
   Reveals whether the model is using absolute-pixel features.

## Concrete next-action

If picking up cold: run the 5M-sample experiment.

```bash
python -m scripts.run_training --task heatmap \
  --epochs 500 --batch-size 100 --image-size 256 \
  --num-train-images 10000 --num-val-images 1000 --lr 1e-4 \
  --train-background random --train-outline random \
  --train-shape-size 10:200

python -m scripts.eval_sweep --run-dir runs/<latest> \
  --rotate true,false --backgrounds solid,texture,random \
  --noise false,true --outlines fill,thin,thick,random \
  --sizes 10:30,30:60,60:120,120:200
```

Expected wall: ~1 hour on CPU PIL path. Compare to the numbers in
`2026-05-02_more_data_per_epoch.md` to see whether the in-distribution
gap closed further.

## Things to NOT do (lessons learned the hard way)

- Don't add sigmoid to regression outputs whose target distribution is
  centered at 0.5. Gradient signal cancels at init.
- Don't use plain BCE/MSE on a sparse Gaussian heatmap target — use
  CenterNet's penalty-reduced focal loss (already in
  `utils/heatmap_loss.py`).
- Don't use BatchNorm with fresh-data-every-epoch training; the
  running stats drift. GroupNorm everywhere on the heatmap models.
- Don't trust the headline loss number. Track Pearson per axis and
  the decomposed loss terms.
- Don't add post-training sweep flags to `run_training` itself —
  that's what `scripts/eval_sweep.py` is for.
- Don't train with `--num-train-images 1000` on a wide distribution
  and conclude the model is "fundamentally less accurate" — it's
  almost certainly undertraining. Bump to 5000+.

## Stable invariants of the codebase

- 51 pytest tests pass, ~2s on CPU. Run them after any change.
- Lint (`ruff check`) and typecheck (`mypy --explicit-package-bases ...`)
  must stay clean on `models/`, `utils/`, `data/`, `scripts/run_training.py`,
  `scripts/eval_sweep.py`, `scripts/sweep_buckets.py`.
- Default `--lr 1e-4` works for the simple encoder. ResNets need
  `lr=1e-5` or they collapse to constant predictions.
- `--gpu-data` is 3× faster but only supports solid backgrounds + filled
  shapes. Augmentation experiments must drop it.
