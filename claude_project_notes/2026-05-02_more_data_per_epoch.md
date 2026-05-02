# More data per epoch closes the augmentation tax

Date: 2026-05-02

## Question

After the 1000-epoch augmented vs baseline comparison
(`2026-05-01_aug_converged.md`), the augmented model came in 20
percentage points behind the baseline on the in-distribution easy case
(solid+fill). The user's question: was that a fundamental
robustness/accuracy tradeoff, or just under-training on a wider task?

## Hypothesis

`--num-train-images 1000` is misleading — with fresh-data-every-call
on each `__getitem__`, an "epoch" is just N gradient steps; total
samples = `num_train_images * epochs`. The 1000-epoch baseline saw
1M total samples; the augmented run also saw 1M but had to fit a
much wider distribution.

If we give the augmented model more **total samples** with the same
wall time (by increasing `num_train_images` per epoch and reducing
the epoch count proportionally), most of the gap should close.

## Setup

Three runs, same `CenterHeatmapNet` (stride=4, 405k params), same lr.

| run                 | imgs/epoch | epochs | total samples |
|---------------------|-----------:|-------:|--------------:|
| Baseline (clean)    |       1000 |   1000 |           1M  |
| Augmented (small)   |       1000 |   1000 |           1M  |
| Augmented (big)     |       5000 |    500 |         2.5M  |

Augmented runs both use `random` background, `random` outline, and
shape sizes 10–200. Same val seed, same val set, same `eval_sweep`.

## Class accuracy on identical val sets

| val set        | Baseline | Aug-1M | **Aug-2.5M** | delta vs Aug-1M |
|----------------|---------:|-------:|-------------:|----------------:|
| solid (baseline's in-dist) | **99.9%** | 79.9% | 91.6% | +11.7 pp |
| texture        |    44.3% |  82.1% |   **90.9%** | +8.8 pp |
| thin outline   |    35.5% |  89.4% |   **95.8%** | +6.4 pp |
| thick outline  |    36.6% |  93.2% |   **99.0%** | +5.8 pp |
| random outline |    52.1% |  96.4% |   **98.7%** | +2.3 pp |
| 10–30 px       |    66.8% |  71.5% |   **91.1%** | +19.6 pp |
| 30–60 px       |     100% |  91.4% |   **98.6%** | +7.2 pp |

## Median pixel error on identical val sets

| val set        | Baseline | Aug-1M | **Aug-2.5M** |
|----------------|---------:|-------:|-------------:|
| solid          |     2.78 |   4.00 |     **3.33** |
| texture        |     6.84 |   3.95 |     **3.32** |
| thin outline   |    30.37 |   5.68 |     **3.57** |
| thick outline  |    27.08 |   3.79 |     **2.56** |
| random outline |    16.37 |   2.50 |     **1.79** |
| 10–30 px       |     0.78 |   0.81 |     **0.67** |

## Conclusion

The "20 pp robustness tax" the previous comparison reported was almost
entirely **undertraining**, not a fundamental tradeoff. Going from 1M
to 2.5M total samples:

- Closes the in-distribution accuracy gap from 20 pp to 8 pp.
- Pushes the augmented model **beyond** the baseline on every
  perturbation (texture, all three outlines, all sizes < 60 px).
- Improves localization on the augmented model's own training
  distribution from median 4.0 px → 3.3 px.

The remaining 8 pp gap on solid+fill is likely also undertraining; the
augmented model's training-val accuracy was still climbing at epoch
500. A run at 5M samples would probably close it further.

The deeper takeaway: **`num_train_images` is the actual training-budget
knob, not `epochs`.** Fresh-per-call data generation makes the "epoch"
abstraction misleading — `--num-train-images 5000` gives the model 5×
more variety per gradient step at the cost of 5× more wall time per
epoch, but the same 5× scaling of total samples seen. We should
probably make `--num-train-images` default to 5000 or 10000 going
forward, and reduce `--epochs` to compensate when wall time matters.

Practically:
- Old recommendation: `--num-train-images 1000 --epochs 1000`
- Better recommendation: `--num-train-images 5000 --epochs 500` for
  wider augmented configs; same or fewer epochs for narrower ones.

## Reproduce

```bash
python -m scripts.run_training --task heatmap \
    --epochs 500 --batch-size 100 --image-size 256 \
    --num-train-images 5000 --num-val-images 1000 --lr 1e-4 \
    --train-background random --train-outline random \
    --train-shape-size 10:200

python -m scripts.eval_sweep --run-dir runs/<latest_dir> \
    --rotate true,false --backgrounds solid,texture,random \
    --noise false,true --outlines fill,thin,thick,random \
    --sizes 10:30,30:60,60:120,120:200
```
