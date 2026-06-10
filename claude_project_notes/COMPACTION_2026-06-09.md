# Compaction note — read this first

Date: 2026-06-09
Purpose: durable record of this session for continuation after context
compaction. **Read this file first.** It supersedes `COMPACTION_2026-05-03.md`
as the latest map (that one is still accurate for the detection-era work).

Everything below is already **committed and merged to `main`** (PRs #1–#6).
Nothing is uncommitted/at risk. `main` == `origin/main`.

## Environment (non-obvious, needed to run anything)

- Python venv with torch 2.8 + CUDA (RTX 3090):
  `/home/richard/.virtualenvs/project_object_detection/bin/python`
  The repo's own `python`/`.venv` do NOT have torch. Always use that interpreter.
- Run tests: `<venv>/bin/python -m pytest` → **129 pass, ~2.5 s**.
- Lint as the project does: `ruff check`, `flake8`, and
  `mypy --namespace-packages --explicit-package-bases --check-untyped-defs --disable-error-code=import-untyped`.
- `runs/`, `*.pth`, `shape_dataset/`, and tool caches are now git-ignored.

## What this session delivered (six merged PRs)

This session took the repo from "detection-only, with latent measurement bugs"
to a **complete, validated CV study sandbox**.

1. **Measurement-correctness review + fixes** (the big one). A multi-agent
   adversarial review found and we fixed: GPU val set drifting each epoch
   (`reseed_each_epoch`); GPU vs CPU data parity (overlap rejection + in-bounds
   rotation now on the GPU path); Hungarian matching dominated by class cost
   (`class_match_weight`, decoupled from `lambda_class`); BatchNorm eval-stats
   confound → **GroupNorm is the default** (encoder `simple_gn`, GroupNorm head);
   `map_center` is now a **confidence-integrated AP** (threshold-independent,
   tie-stable); `eval_sweep` evaluates against the model's actual training
   distribution. Plus medium-tier: MultiHeatmap plateau dedup, shape-size
   validation, honest multi-object claims, CPU/GPU parity docs.

2. **Config-driven studies.** `DatasetConfig` (all shape knobs) + YAML study
   configs with independent `train:`/`val:` distributions (`val` inherits
   `train`). Replaced the per-knob `--train-*` flags. A distribution-shift study
   is now one file (`configs/`), evaluated online every epoch.

3. **Segmentation + classification** (`--task segmentation`, `--task classification`).

4. **Instance segmentation** (`--task instance_seg`) — Panoptic-DeepLab-style:
   semantic + center-heatmap + **offset-to-center** heads; foreground pixels vote
   `pixel + offset` → nearest detected center. Metrics: mean IoU, recall@IoU,
   **mask AP**, **Panoptic Quality**.

5. **Size-agnostic encoders.** Spatial-flatten encoders had a hardcoded 256px
   head (single/multi crashed at other sizes); an `AdaptiveAvgPool` to the
   canonical grid (no-op at 256px) fixes it.

6. **Full augmentation palette.** rotation, noise, texture bg, **outlines (incl.
   in masks)**, **blur**, **color-threshold**, overlap — all settable per
   distribution and sweepable via `eval_sweep`.

## Current capabilities — seven tasks behind `--task`

| task | model | metric |
|---|---|---|
| single / heatmap | SimpleCenterNet / CenterHeatmapNet | center px, accuracy |
| multi / multi_heatmap | CenterPredictor / MultiHeatmapNet | `map_center` AP, matched px |
| segmentation | ShapeSegNet | mIoU, per-class IoU, pixel acc |
| instance_seg | InstanceSegNet | mean IoU, recall@IoU, mask AP, PQ |
| classification | ShapeClassifier | accuracy |

Validated convergence (GPU smokes): single heatmap → 0.59 px / 99%; multi_heatmap
→ matched 2.1 px, AP rising; segmentation → mIoU 0.94; instance_seg → mean_iou
0.77 / recall@0.75 0.71 / AP50 0.85 / PQ 0.64; classification → 0.82.

## Where to look in the repo (new/changed since last note)

| Path | What |
|---|---|
| `data/dataset_config.py` | `DatasetConfig` + `build_cpu_dataset` / `build_gpu_loader`. The knob source of truth. |
| `models/seg_net.py`, `models/instance_seg_net.py`, `models/shape_classifier.py` | the new task models |
| `utils/seg_loss.py` | `SegLoss`, `InstanceSegLoss` |
| `utils/metrics.py` | now also `evaluate_segmentation`, `evaluate_instance_segmentation`, the AP/PQ helpers, the confidence-integrated `map_center` |
| `scripts/run_training.py` | `--config`, all seven tasks, `evaluate_seg/classification/instance_seg`, `check_single_shape_tasks` |
| `scripts/eval_sweep.py` | sweeps for every task incl. `--blur` / `--color-thresholds` |
| `configs/`, `examples/` | YAML study configs + a worked Python example |
| `docs/instance_segmentation_design.md` | instance-seg design + status |

## What's open (future ideas, nothing blocking)

- Instance seg: DETR-style fixed-slot mask head (option 3 in the design doc) for
  a direct comparison; multi-scale decoders for very dense scenes.
- `instance_seg` per-epoch eval is a bit slow (computes AP/PQ); fine for small
  val sets, could be sped up.
- A fresh dated `claude_project_notes/` writeup once real long runs are done on
  the new tasks (the convergence numbers above are short smokes, not the
  "fully solved" long-run numbers).

## Working conventions established this session

- **Workflow:** branch off `main` → commit per phase (each phase tested + linted)
  → push → open PR → user merges → delete branch (local + remote). Commit
  messages end with the `Co-Authored-By: Claude Opus 4.8 (1M context)` trailer.
- **Validation discipline:** every feature gets unit tests + a GPU convergence
  smoke; larger work gets a multi-agent adversarial review of the diff before
  the PR; fix only what's confirmed.
- Only commit MY files (explicit `git add` paths) — the repo has several
  pre-existing untracked files (notebooks, `scripts/matching_algorithm.py`,
  `training/*.pth`) that are the user's and must NOT be committed.
