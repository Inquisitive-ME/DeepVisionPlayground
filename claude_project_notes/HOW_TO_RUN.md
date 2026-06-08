# How to actually use this thing

A practical guide. If you just want to reproduce the headline result, read
"Quick start"; if you want to add a model or run a study, read on.

## Setup

```bash
pip install -r requirements.txt
# Optional but recommended for benchmarking:
pip install tensorboard scipy
```

GPU is strongly recommended; CPU works for tests and tiny smoke runs.

```bash
# Sanity-check that everything is wired:
pytest -q
# 50+ tests, ~2 seconds.
```

## Quick start: solve the single-object task

```bash
python -m scripts.run_training \
    --task heatmap --gpu-data \
    --epochs 1000 --batch-size 100 --image-size 256 \
    --num-train-images 1000 --num-val-images 1000 \
    --lr 1e-4 --heatmap-stride 1
```

5–15 minutes on a 3090. Final numbers: median ~1.8 px localization,
~99.7% classification accuracy. Look at `runs/<your-run>/results.json`
for the full metric set.

## Quick start: multi-object with sweeps

```bash
python -m scripts.run_training \
    --task multi_heatmap --gpu-data \
    --epochs 1000 --batch-size 100 --image-size 256 \
    --num-train-images 1000 --num-val-images 500 \
    --num-shapes-min 0 --num-shapes-max 5 --max-objects 10 \
    --lr 1e-4 \
    --val-shape-counts '1,2,3,5,7,10' \
    --val-shape-sizes '10:30,30:60,60:120,120:180'

python -m scripts.sweep_buckets runs/multi_heatmap_simple_bn_256_bs100_<TS>
```

The first command trains and writes `results.json`; the second
pretty-prints the bucket sweeps as a Markdown table.

## The CLI flags worth knowing

```
--config path/to/study.yaml
    A self-contained study: run/model scalars plus train: / val: data
    distributions (see configs/). When given it is authoritative — the other
    flags are ignored. This is how you set up a distribution-shift study
    (train on one distribution, validate on another); val inherits train and
    overrides only what shifts, so there are no per-augmentation flags.

--task {single, multi, heatmap, multi_heatmap}
    single         FC regression on one shape
    multi          FC + slot-based detection on N shapes (Hungarian-matched)
    heatmap        CenterNet heatmap on one shape (sub-pixel)
    multi_heatmap  CenterNet heatmap with NMS for N shapes (sub-pixel + multi)

--encoder {simple, simple_bn, simple_gn, simple_gap, simple_bn_gap,
           simple_gn_gap, resnet18, resnet18_spatial, resnet34, resnet34_spatial}
    Affects: single, multi (heatmap variants use a built-in encoder).
    Default simple_gn (GroupNorm) behaves the same in train/eval; simple_bn
    and resnet* carry BatchNorm and bias distribution-shift measurements.
    *_gap variants are for classification — they have NO position info.
    *_spatial variants strip the AvgPool from torchvision ResNet.

--gpu-data
    Generate batches directly on the GPU. ~3x faster than the PIL
    workers. Solid backgrounds and filled shapes only.

--heatmap-stride {1, 2, 4, 8}
    Output stride for heatmap models. 1 = exact-pixel, 4 = fast.

--num-shapes-min / --num-shapes-max
    Range of shapes per image (multi tasks).

--max-objects N
    Number of detection slots in multi / multi_heatmap. Must be >=
    --num-shapes-max so the model has somewhere to put each shape.

--val-shape-counts '1,3,5,10'
--val-shape-sizes '10:30,30:60,60:120'
    Optional. After training, eval against alternate val distributions
    and write metrics under final_metrics_by_count / final_metrics_by_size.

--lr
    1e-4 is the working default for the simple encoder. ResNet variants
    need ~1e-5 or they collapse to constant predictions early in
    training.

--lambda-class / --lambda-conf
    Loss weights for the multi_loss. Default 1.0 each. Bumping
    lambda_class to 3.0 unblocks classification on the FC multi path.

--class-match-weight
    Weight of the class term inside the multi-object Hungarian matching cost
    (default 0.1, separate from --lambda-class). Kept small so class only
    breaks ties between near-coincident objects; a larger value lets class
    override spatially-correct assignments (centers are normalized to [0,1]).
```

Augmentations (rotation, noise, texture/outline, sizes, counts, shape types)
are NOT CLI flags — they're per-distribution knobs in a `--config` study file,
set independently for `train:` and `val:`. See `configs/README.md`.

## Useful scripts

| Script | What it does |
|---|---|
| `python -m scripts.run_training ...` | Main training entry point |
| `python -m scripts.eval_sweep --run-dir runs/<dir> ...` | Load a trained model and sweep a val-time perturbation (`--rotate`, `--noise`, `--backgrounds`, `--outlines`, `--counts`, `--sizes`, `--overlaps`) against its training distribution |
| `python -m scripts.benchmark_models` | Runs the 5 reference configs and prints a comparison table |
| `python -m scripts.sweep_buckets runs/<dir>` | Reads a results.json and pretty-prints the bucket metrics |
| `python -m scripts.gpu_monitor` | Polls nvidia-smi and writes a CSV; run alongside training |

## Reading the metrics

The metrics are intentionally a bit verbose so the failure mode is
visible at a glance. Mental model for what each one tells you:

- **mean_center_px / median_center_px** — pixel distance between
  prediction and truth. Median is robust to a few outliers; mean
  reflects the worst cases.
- **pearson_cx / pearson_cy** — across all val images, do the
  predicted x's and y's track the true ones? A constant predictor
  gets 0; a perfect predictor gets 1. Lights up when localization
  works qualitatively even before pixel error is good.
- **accuracy** (single) / **matched_class_accuracy** (multi) — class
  correctness; multi is restricted to Hungarian-matched pairs. NOTE: all
  `matched_*` multi metrics (matched center px, matched class accuracy) are
  computed only over matched prediction/GT pairs — they ignore false positives
  and missed objects, so judge detection quality by `map_center` +
  `cardinality_error`, not the matched numbers alone.
- **recall@T px** — fraction of GTs whose nearest prediction is within
  T pixels. Different thresholds reveal different failure modes:
  recall@2 is "are the centers exact?"; recall@16 is "did we find
  every object roughly?".
- **map_center** — confidence-integrated average precision (area under the
  PR curve, averaged over pixel thresholds). Single number for ranking runs;
  independent of the confidence threshold, so FC-multi and heatmap models are
  comparable on it.
- **by_size/{xs,sm,md,lg}/...** — same metrics restricted to GT shapes
  in each size bucket. Tells you whether the model is failing on
  small shapes specifically.
- **by_count/{n0..n21+}/...** — restricted to images with that many
  GTs. Tells you if the model breaks at high density.

## Adding a new model

1. Drop a new `nn.Module` into `models/`.
2. Add it to the `--task` argparse choices and the model-construction
   `if/elif` in `scripts/run_training.py`.
3. If it has a different output shape than the existing `(B, 2 + C)`
   or `(B, max_objects, 3 + C)`, write a tiny adapter so the existing
   `evaluate_*` metrics can still score it.
4. Add tests in `tests/test_models.py` for shape correctness and
   gradient flow.
5. Run a 30-epoch smoke and confirm the metrics behave as expected;
   open a results note in `claude_project_notes/`.

## Adding a new dataset

The path is via `data/synthetic_shapes_dataset.py` (CPU+PIL) or
`data/gpu_shapes.py` (GPU rasterizer). The CPU path is more flexible
(textures, noise, outlines); the GPU path is faster but limited to
solid backgrounds and filled shapes.

For an entirely new domain (e.g. lines, letters), look at
`data/synthetic_lines_dataset.py` for the shape of the API, and
write per-domain metrics in `utils/metrics.py`.

## Performance tips

- Use `--gpu-data` whenever you can; 3× speedup.
- `--num-workers 4 --persistent-workers` is already the default for
  the CPU dataloader.
- Heatmap stride 4 is much cheaper than stride 1; use it for early
  experimentation and switch to stride 1 only when you want the
  exact-pixel localization.
- `--num-train-images` doesn't need to be large because we generate
  fresh data every epoch. 1000 is the right ballpark.
- `--num-val-images` should be larger (1000+) to keep the metrics
  noise-free. Each epoch's val pass is fast (~0.3 s).

## Where to look when something doesn't work

- **Loss looks stuck**: check the per-axis Pearson — if it's 0, the
  model collapsed to a constant; if it's 0.5+ it's actually learning
  but slowly.
- **Loss exploded on epoch 1**: lr is too high for this encoder.
  ResNets typically need lr=1e-5; the `simple` encoder works at 1e-4.
- **Pearson stays at 0 for many epochs with ResNet**: see above.
- **Multi-object recall@T very low but classes correct**: the
  matching radius in the metric is the limit. Try a longer training
  or a smaller `--heatmap-stride`.
- **All metrics zero on first epoch**: the confidence threshold in
  `evaluate_multi_object` is filtering everything out. Heatmap
  models use threshold 0.1 by default; FC multi uses 0.5.
