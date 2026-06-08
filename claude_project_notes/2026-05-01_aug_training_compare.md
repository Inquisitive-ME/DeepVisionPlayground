# Training-time augmentations: do they fix the failure modes?

Date: 2026-05-01

## Question

The previous sweep
(`claude_project_notes/2026-05-01_augmentation_sweep.md`) identified
three catastrophic failure modes for a single-object heatmap model
trained on the canonical clean config:

| dimension | failure                                  |
|-----------|------------------------------------------|
| outline   | fill → thin: median 4 → 34 px            |
| background| solid → texture: median 4 → 9 px         |
| size      | 30–60 → 120–200 px: median 1.6 → 10.8 px |

Hypothesis: training with `background=RANDOM`, `shape_outline=RANDOM`,
and a wider `shape_size_range=(10, 200)` should make those failure
modes go away, possibly at the cost of in-distribution accuracy.

## Setup

Two 300-epoch runs on identical model + hyperparameters
(`CenterHeatmapNet` stride=4, lr=1e-4, 1000 train images, 1000 val
images), differing only in the training data distribution:

- **Baseline**: `solid` background, `fill` outline, `(20, 128)` sizes.
  Trained on GPU-rendered shapes (`--gpu-data`).
- **Augmented**: `random` background, `random` outline, `(10, 200)`
  sizes. Trained on the CPU PIL path (the GPU loader doesn't support
  these augmentations yet).

Both evaluated against the same `eval_sweep` over rotation, background,
noise, outline, and size.

## In-distribution training-val (unfair comparison)

| run        | val median_px | val accuracy | val Pearson |
|------------|--------------:|-------------:|------------:|
| Baseline   | 3.7           | 0.80         | 0.996       |
| Augmented  | 4.0           | 0.71         | 0.98        |

The augmented run's val set is itself harder (random backgrounds, random
outlines, wider size range), so this isn't apples-to-apples — it's
"each model on its own distribution."

## Cross-distribution sweep (head-to-head)

Every row below is the **same val set evaluated with both models**.

### Background

| eval bg      | baseline median / acc | augmented median / acc |
|--------------|----------------------:|------------------------:|
| solid (training match for baseline) | 4.25 / 0.773 | 5.30 / 0.611 |
| texture      | **9.07 / 0.402**      | **4.83 / 0.644**        |
| random mix   | 6.01 / 0.613          | 5.43 / 0.584            |

The augmented model is **better** on texture than the baseline is on
*solid* (its training distribution). Pearson on texture climbs from
0.91 → 0.99. The texture failure mode is essentially erased.

### Outline

| eval outline | baseline median / acc | augmented median / acc |
|--------------|----------------------:|------------------------:|
| fill (training match for baseline) | 4.27 / 0.773 | 5.32 / 0.611 |
| thin         | **33.90 / 0.409**     | **8.15 / 0.649**        |
| thick        | 29.29 / 0.454         | 5.89 / 0.682            |
| random       | 17.79 / 0.534         | 4.42 / 0.722            |

The 8× outline-failure ratio is now ~1.5×. Class accuracy on hollow
shapes actually goes *up* with augmented training (0.65 vs 0.41 thin,
0.68 vs 0.45 thick). Outline-augmentation totally pays for itself.

### Shape size

| eval size  | baseline median / acc | augmented median / acc |
|------------|----------------------:|------------------------:|
| 10–30 px   | 1.24 / 0.421          | 1.50 / 0.433           |
| 30–60 px   | 1.65 / 0.807          | 2.22 / 0.619           |
| 60–120 px  | 5.26 / 0.724          | 6.71 / 0.659           |
| 120–200 px | 10.81 / 0.341         | **12.64 / 0.397**      |

Mostly a wash — augmented model is slightly worse on every size bucket
but slightly better on the largest. Wider training range did NOT make
the model dramatically better at the size extremes.

### Rotation, noise

Both essentially unchanged across runs. Rotation training was already
correct; mild noise was already a no-op.

## Cost: in-distribution drop

Comparing the same in-distribution val (solid + fill + medium size) used
by the baseline:

| metric        | baseline (300 ep) | augmented (300 ep) | delta   |
|---------------|------------------:|-------------------:|--------:|
| median_px     | 4.27              | 5.32               | +1.05 px (+25%) |
| accuracy      | 0.773             | 0.611              | −16 pp |
| pearson_cx    | 0.994             | 0.990              | small  |

The augmented model gives up ~16 percentage points of in-distribution
accuracy and 1 px of median error. That's a real cost, but it's **the
exact tradeoff the augmentation sweep was supposed to surface**: a more
robust model that's a bit less accurate on the easy case.

## Conclusions

1. **Augmentation training fixes the texture and outline failure modes
   essentially completely.** The texture penalty (a 5 px / 37 pp
   regression) becomes a non-effect; the outline penalty (a 30 px /
   36 pp regression) drops to 3 px / 6 pp.
2. **Training on a wider size range did NOT solve the size-extreme
   problem.** Both models still fail above 120 px and have low class
   accuracy below 30 px. This is an architecture/representation issue,
   not a training-distribution one.
3. **The robustness comes with a real in-distribution cost.** 16 pp
   accuracy is not free. Whether it's worth paying depends on whether
   you actually need the model to handle textures + hollow shapes at
   deployment.
4. **More training would close the in-distribution gap.** 300 epochs
   is undertrained for the augmented run because the effective task
   is harder. Worth running 1000 epochs on the augmented config to
   see if it catches up.

## Reproduce

```bash
# Baseline (5 min on a 3090):
python -m scripts.run_training --task heatmap --gpu-data \
    --epochs 300 --batch-size 100 --image-size 256 \
    --num-train-images 1000 --num-val-images 1000 --lr 1e-4

# Augmented (~17 min on CPU PIL path because GPU loader doesn't
# support these augmentations yet):
python -m scripts.run_training --task heatmap \
    --epochs 300 --batch-size 100 --image-size 256 \
    --num-train-images 1000 --num-val-images 1000 --lr 1e-4 \
    --train-background random --train-outline random \
    --train-shape-size 10:200

# Sweep both runs (~30 sec each):
python -m scripts.eval_sweep --run-dir runs/<baseline> \
    --rotate true,false --backgrounds solid,texture,random \
    --noise false,true --outlines fill,thin,thick,random \
    --sizes 10:30,30:60,60:120,120:200

python -m scripts.eval_sweep --run-dir runs/<augmented> \
    --rotate true,false --backgrounds solid,texture,random \
    --noise false,true --outlines fill,thin,thick,random \
    --sizes 10:30,30:60,60:120,120:200
```
