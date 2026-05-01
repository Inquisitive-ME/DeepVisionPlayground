# Augmentation distribution-shift study

Date: 2026-05-01

## Setup

Trained one heatmap model (`CenterHeatmapNet` stride=4, 405 K params)
for 300 epochs on the canonical single-object config:

- `num_shapes_range=(1, 1)`, `shape_size_range=(20, 128)`, rotation **on**
- `background=SOLID`, `add_noise=False`, `shape_outline=FILL`
- 256×256 canvas, lr=1e-4, `--gpu-data`

In-distribution training-val: median 3.7 px, accuracy 0.80, Pearson 0.996.

Then ran `scripts/eval_sweep.py` against the trained model along five
augmentation dimensions, holding everything else at training defaults.
**The model never sees these val sets during training**; this is purely
a distribution-shift study.

## Results

### Shape size (training range was 20–128 px)

| size range | median_px | mean_px | accuracy | pearson_cx |
|------------|----------:|--------:|---------:|-----------:|
| 10–30 px   | **1.24**  | 1.51    | 0.421    | 1.000      |
| 30–60 px   | 1.65      | 1.86    | 0.807    | 1.000      |
| 60–120 px  | 5.26      | 6.48    | 0.724    | 0.993      |
| 120–200 px | 10.81     | 12.23   | 0.341    | 0.976      |

The "small shapes are easier than big ones" pattern from the count/size
sweep replicates: localization is best at the smallest sizes (1.2 px
median!) and degrades on big shapes that exceed the training range.
The 10–30 px accuracy is suspiciously low (0.42) — those tiny shapes
are localized cleanly but the class head can't disambiguate them
because at 10 px a square and a triangle become two-pixel blobs.

### Rotation in val (training had rotation=on)

| eval rotation | median_px | accuracy | pearson_cx |
|---------------|----------:|---------:|-----------:|
| true (match)  | 4.26      | 0.773    | 0.994      |
| false         | 4.92      | **0.664**| 0.995      |

Slightly worse on non-rotated val. Counterintuitive — you'd expect
non-rotated to be the easier case. The mechanism is probably that the
val seed produces *different* shapes when you turn rotation off, so
"rotation off" is itself a small distribution shift.

### Background

| background | median_px | accuracy | pearson_cx |
|------------|----------:|---------:|-----------:|
| solid (match) | 4.25   | 0.773    | 0.994      |
| texture    | **9.07**  | **0.402**| 0.910      |
| random mix | 6.01      | 0.613    | 0.948      |

**Sharpest single failure mode** in the study. Median pixel error more
than doubles on textured backgrounds and class accuracy drops 37
percentage points. Pearson 0.91 is still high (the model is roughly
tracking position) but the heatmap is noticeably less peaked because
the textured pixels look like distractor activations.

### add_noise (Gaussian std=10 added post-render)

| noise        | median_px | accuracy | pearson_cx |
|--------------|----------:|---------:|-----------:|
| false (match)| 4.27      | 0.773    | 0.994      |
| true         | 4.46      | 0.777    | 0.992      |

Essentially no effect. The model is robust to mild image noise — a
nice signal that the heatmap features are not overly sensitive to
per-pixel values.

### Shape outline (training was FILL)

| outline     | median_px | accuracy | pearson_cx |
|-------------|----------:|---------:|-----------:|
| fill (match)| 4.27      | 0.773    | 0.994      |
| **thin**    | **33.90** | 0.409    | 0.902      |
| **thick**   | 29.29     | 0.454    | 0.904      |
| random      | 17.79     | 0.534    | 0.947      |

**The most catastrophic finding.** Median pixel error goes from 4 px
to 34 px on hollow shapes — an 8× regression. This means the model
trained on filled shapes is using **interior color information** as a
primary cue, and a hollow shape simply doesn't have the feature it
learned to look for. Pearson holds at 0.90 (it still finds the shape
in some sense), but accuracy crashes to 41%.

## Conclusions

### What the model has learned

1. **Position information from peak features at the shape boundary** —
   this is what gives Pearson > 0.9 even under heavy distribution
   shift. The heatmap's peak structure transfers.
2. **Class identity from the shape's interior color/extent** — this
   is what catastrophically fails when you switch to outlines or
   textured backgrounds.
3. **Approximate scale invariance within the trained range**, but no
   extrapolation to either much smaller or much larger shapes.

### What the model has NOT learned

1. **Topological shape class.** A hollow triangle is "not a triangle"
   to this model. Real generalization to outlines would need either
   training-time augmentation (mix in `shape_outline=RANDOM`) or a
   shape-mask training target.
2. **Background invariance.** Solid-trained models can't see through
   textured noise. Trivially fixable by training on
   `BackgroundType.RANDOM`.
3. **Robust class identity.** Class accuracy is the headline weakness:
   77% → 41% with outline change, 77% → 40% with texture, 80% → 34%
   with large shapes. The localization head is robust; the class head
   is brittle.

### What this implies for the data-generation surface

We now have evidence for **what augmentations are worth turning on at
training time** (in priority order, by expected payoff):

1. **`background=RANDOM`** — eliminates the texture failure mode.
   Cheap.
2. **`shape_outline=RANDOM`** — eliminates the outline failure mode.
   Equally cheap.
3. **Wider `shape_size_range=(10, 200)`** — eliminates the
   large-shape collapse and the small-shape class-confusion.
4. **`add_noise=True`** — model is already robust, so this is a
   no-op rather than a fix. Keep it off in default training.

The next experiment (covered in a separate writeup) trains with these
three augmentations on, then re-runs the same sweep to see whether
the failure modes disappear.

## Reproduce

```bash
# Train (5 minutes on a 3090):
python -m scripts.run_training --task heatmap --gpu-data \
    --epochs 300 --batch-size 100 --image-size 256 \
    --num-train-images 1000 --num-val-images 1000 \
    --lr 1e-4 --heatmap-stride 4

# Sweep all five dimensions (~30 seconds total on CPU):
python -m scripts.eval_sweep --run-dir runs/heatmap_simple_bn_256_bs100_<TS> \
    --rotate true,false \
    --backgrounds solid,texture,random \
    --noise false,true \
    --outlines fill,thin,thick,random \
    --sizes 10:30,30:60,60:120,120:200
```
