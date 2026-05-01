# Multi-object density / overlap sweep

Date: 2026-05-01

## Hypothesis

The 3×3 max-pool NMS in `MultiHeatmapNet.decode` suppresses peaks
within ~12 image pixels of each other (stride 4 × kernel 3). When
shapes overlap heavily, two true centers might be < 12 px apart and
the NMS would collapse them into one detection — a deterministic
failure mode that more training cannot fix.

Test: train at standard density (max_overlap=0.6) and evaluate against
val sets at 0.10, 0.30, 0.60, and 0.90 max_overlap.

## Setup

- Task: `multi_heatmap`, 405k-param `MultiHeatmapNet`, stride=4
- Train: `num_shapes_range=(2, 5)`, `shape_size_range=(20, 90)`, default
  `max_overlap=0.6`, 800 epochs, lr=1e-4, `--gpu-data`.
- Sweep: `--val-overlaps "0.1,0.3,0.6,0.9"`. The overlap sweep falls
  back to CPU because the GPU loader doesn't support overlap controls.

## Result

| max_overlap | mean_px | median_px | class_acc | map_center | pearson_cx |
|------------:|--------:|----------:|----------:|-----------:|-----------:|
| 0.10        | 2.52    | 1.48      | **0.999** | 0.363      | 0.997      |
| 0.30        | 2.75    | 1.60      | 0.995     | 0.350      | 0.996      |
| 0.60        | 3.05    | 1.99      | 0.965     | 0.320      | 0.998      |
| 0.90        | 3.46    | 1.87      | 0.968     | 0.320      | 0.994      |

## Conclusion

The hypothesized NMS-collapse failure mode does not show up at any
tested overlap level. The model degrades smoothly from `max_overlap=0.1`
to `0.9`:

- Median pixel error climbs from 1.48 → 1.87 (+25%)
- Class accuracy drops from 99.9% → 96.8% (−3 pp)
- Pearson stays > 0.99 throughout

That's gentle. Why?

1. **NMS only suppresses pixels strictly less than the local max.** Two
   close shapes that each fire their own peak above their neighbors
   are both kept by NMS. Overlap by area doesn't translate directly
   to "peaks within 12 px" — the GT centers can still be far apart
   even when the bboxes overlap heavily, because `compute_overlap_ratio`
   uses intersection over the smaller area, not center distance.
2. **The Gaussian target with sigma=1.5 sharpens fast.** Once the model
   learns it, peaks 4–8 px apart are clearly separated despite a 3×3
   suppression window.

To actually trip the NMS we'd need to either:
- Force shapes to be co-located by center (not just overlapping bboxes), or
- Increase the NMS kernel size, or
- Use stride 1 with the same kernel (the stride-4 collapse window is
  small relative to typical shape sizes).

This is a pleasant surprise. Density isn't a bottleneck for this
configuration.

## Reproduce

```bash
python -m scripts.run_training --task multi_heatmap --gpu-data \
    --epochs 800 --batch-size 100 --image-size 256 \
    --num-train-images 1000 --num-val-images 500 \
    --num-shapes-min 2 --num-shapes-max 5 --max-objects 10 \
    --lr 1e-4 \
    --val-overlaps '0.1,0.3,0.6,0.9'

python -m scripts.sweep_buckets runs/multi_heatmap_simple_bn_256_bs100_<TS>
```
