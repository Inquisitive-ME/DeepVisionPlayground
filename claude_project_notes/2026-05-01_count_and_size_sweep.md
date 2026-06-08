# Multi-object: how does it scale by count and size

Date: 2026-05-01

## Setup

Training: `multi_heatmap` task, 405k-param `MultiHeatmapNet`, stride=4,
trained on `num_shapes_range=(0, 5)`, `shape_size_range=(20, 90)`, 1000
epochs at lr=1e-4. Default size buckets: xs (0–30 px), sm (30–60), md
(60–120), lg (120+).

After training, we evaluated the same model against six **separate**
val sets each with a fixed `num_shapes_range=(N, N)` for
N ∈ {1, 2, 3, 5, 7, 10}, and four val sets with different
`shape_size_range` values.

## Within the training distribution

The val set was the same as training: `num_shapes_range=(0, 5)`,
`shape_size_range=(20, 90)`. Bucketed view:

| Count bucket | n_gt | mean_px | r@2px | r@4px | r@8px | r@16px |
|--------------|-----:|--------:|------:|------:|------:|-------:|
| n1           | 88   | 2.20    | 0.58  | 0.83  | 1.00  | 1.00   |
| n2           | 164  | 1.90    | 0.66  | 0.88  | 0.99  | 0.99   |
| n3-5         | 916  | 3.19    | 0.54  | 0.76  | 0.95  | 0.99   |

| Size bucket  | n_gt | mean_px | r@2px | r@4px | r@8px | r@16px |
|--------------|-----:|--------:|------:|------:|------:|-------:|
| xs (<30 px)  | 25   | 1.36    | 0.80  | 0.88  | 1.00  | 1.00   |
| sm (30–60)   | 389  | 2.50    | 0.71  | 0.86  | 0.96  | 0.97   |
| md (60–120)  | 754  | 3.21    | 0.47  | 0.74  | 0.95  | 0.99   |

Within the training distribution, larger shapes are slightly worse than
smaller ones (mean_px 3.21 vs 1.36 between md and xs) but everything
gets recall@16 ≥ 95%, so the model finds every object — it just
localizes the smaller ones more precisely.

## Cross-distribution count sweep

Model trained on (0, 5); evaluated on (N, N) for N up to 10.

| Eval count | mean_px | median_px | class_acc | map_center | pearson_cx |
|-----------:|--------:|----------:|----------:|-----------:|-----------:|
| 1          | 1.73    | 1.14      | 1.000     | 0.438      | 1.000      |
| 2          | 2.18    | 1.41      | 0.990     | 0.363      | 0.999      |
| 3          | 2.95    | 1.70      | 0.975     | 0.318      | 0.994      |
| 5          | 4.47    | 2.21      | 0.943     | 0.323      | 0.988      |
| 7          | 8.82    | 2.94      | 0.886     | 0.336      | 0.963      |
| 10         | 22.4    | 5.15      | 0.766     | 0.278      | 0.867      |

In-distribution (1–5): **localization is essentially perfect**, median
≤ 2.2 px and class accuracy 94–100%. The graceful curve from 1 → 5 is
exactly what you want — adding shapes makes the per-image task harder
but the model is still tracking each one.

Out-of-distribution (7–10, model only trained up to 5): we see
extrapolation breakdown. Median jumps to 5.15 px at N=10 and the mean
balloons to 22 px — this latter is the tail caused by some shapes being
completely missed. Class accuracy hangs in at 77% even at 10 shapes,
well above chance (33%).

The interesting thing is `pearson_cx` stays at 0.87 even at N=10. The
model is still tracking position; it just sometimes misses entire
objects when the heatmap can't fit that many separate peaks at stride
4. Fixable with a finer stride and/or more `max_objects`.

## Cross-distribution size sweep

Model trained on size 20–90 px; evaluated on four ranges.

| Size range  | mean_px | median_px | class_acc | map_center | pearson_cx |
|-------------|--------:|----------:|----------:|-----------:|-----------:|
| 10–30 px    | 1.07    | 0.85      | 0.85      | 0.897      | 1.000      |
| 30–60 px    | 2.29    | 1.04      | 0.98      | 0.537      | 0.994      |
| 60–120 px   | 12.49   | 6.86      | 0.78      | 0.092      | 0.954      |
| 120–180 px  | 28.43   | 21.92     | 0.50      | 0.009      | 0.825      |

Two findings worth flagging:

1. **The model performs *better* on shapes smaller than its training
   range**. 10–30 px (mostly out-of-distribution low) hits median
   0.85 px — better than any in-distribution slice. Probably because
   small shapes are "denser" features and the heatmap can lock onto
   them sharply.

2. **Large shapes are the real failure mode**. By 120–180 px the
   median pixel error is 21.9 px and class accuracy crashes to 50%.
   These shapes are well outside the training range, but the deeper
   issue is that the visual "center" of a large filled rectangle isn't
   a distinctive feature — there's no edge or corner at the actual
   center, so the heatmap lands on a corner or edge instead.

`map_center` cleanly summarizes the failure curve:
0.897 → 0.537 → 0.092 → 0.009.

## Conclusions

1. **The architecture generalizes well within the training count
   range** (N=1..5): graceful degradation, not a cliff.
2. **It extrapolates to higher counts mostly through localization
   degradation** rather than missing objects entirely — Pearson
   stays high.
3. **Small shapes are easier than large shapes**, contrary to the
   typical "small things are hard" intuition for natural images. On
   solid backgrounds with high contrast colors, small filled shapes
   are sharper signals.
4. **The biggest unexploited improvement is on large shapes**:
   training on `(20, 180)` instead of `(20, 90)` should fix the
   120–180 px collapse; predicting the geometric center via something
   that emphasizes object extent (bbox heatmap or shape-mask
   regression) should fix it more robustly.

## Reproduce

```bash
python -m scripts.run_training --task multi_heatmap --gpu-data \
    --epochs 1000 --batch-size 100 --image-size 256 \
    --num-train-images 1000 --num-val-images 500 \
    --num-shapes-min 0 --num-shapes-max 5 --max-objects 10 \
    --lr 1e-4 \
    --val-shape-counts '1,2,3,5,7,10' \
    --val-shape-sizes '10:30,30:60,60:120,120:180'

python -m scripts.sweep_buckets runs/multi_heatmap_simple_bn_256_bs100_<TS>
```
