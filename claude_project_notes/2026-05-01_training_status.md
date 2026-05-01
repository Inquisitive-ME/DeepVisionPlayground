# Training status — single-object task

Date: 2026-05-01
Question: are we where we should be? Long runs and architecture comparison.

## tl;dr

**Localization is doing fine** — `simple` model converges to ~16 px median
error (6% of canvas), Pearson 0.97. That's solid for a tiny model.

**Classification is the real laggard** — 81% accuracy after 1000 epochs on
a 3-shape task. Climbing throughout, not a permanent plateau, but well
short of the 99%+ a "fully solved" answer would look like.

**Architecture comparison is broken in an instructive way**: `simple` (no
BN) actually outperforms `simple_bn` here, and ResNet was silently
collapsing at the lr we'd been using. Once you set lr per-encoder, ResNet
trains, but I haven't run a long enough comparison to claim it's better.

## Numbers from this session

All single-object task, 256x256 images, batch 100, 1000 train images
fresh-generated each epoch on the GPU, lambda_class=1.0.

### `simple` encoder, lr=1e-4, 1000 epochs

| epoch | mean_px | median_px | accuracy | pearson_cx | pearson_cy |
|------:|--------:|----------:|---------:|-----------:|-----------:|
|     1 |    81.6 |      78.7 |    0.318 |       0.07 |       0.00 |
|    50 |    48.0 |      44.5 |    0.343 |       0.74 |       0.77 |
|   100 |    30.4 |      24.4 |    0.372 |       0.86 |       0.88 |
|   200 |    19.2 |      14.5 |    0.377 |       0.95 |       0.94 |
|   300 |    16.5 |      13.4 |    0.462 |       0.97 |       0.97 |
|   500 |    16.2 |      14.3 |    0.578 |       0.97 |       0.97 |
|   700 |    17.6 |      15.5 |    0.651 |       0.97 |       0.97 |
|  1000 |    16.6 |      14.7 |    0.810 |       0.97 |       0.97 |

Localization plateaus by epoch ~200; classification continues to climb
throughout. Train loss drops smoothly from 1.23 → 0.48.

### `simple_bn` encoder, lr=1e-4, 300 epochs

|  epoch | mean_px | median_px | accuracy | pearson_cx | pearson_cy |
|-------:|--------:|----------:|---------:|-----------:|-----------:|
|    100 |    33.9 |      28.7 |    0.445 |       0.87 |       0.88 |
|    200 |    28.0 |      22.9 |    0.405 |       0.90 |       0.90 |
|    300 |    29.0 |      25.5 |    0.420 |       0.90 |       0.90 |

Strictly worse than `simple` at 300 epochs (mean_px 29 vs 16, Pearson 0.90
vs 0.97). BN is hurting here, probably because:

- The val pass uses `model.eval()` → BN running statistics, not per-batch.
- With ten batches of 100 fresh-random images per epoch and shapes that
  vary wildly in color/position, the running stats are themselves noisy,
  and the eval-time activations don't match the train-time ones.

Worth noting, but not urgent.

### `resnet18_spatial`, lr=1e-4 vs lr=1e-5

At lr=1e-4 (the default in `run_training`):

|  epoch | mean_px | accuracy | pearson_cx | pearson_cy |
|-------:|--------:|---------:|-----------:|-----------:|
|     10 |   185.9 |    0.310 |       0.00 |       0.00 |
|    100 |   157.2 |    0.290 |       0.00 |       0.00 |
|    200 |   117.7 |    0.325 |       0.00 |       0.00 |

Pearson exactly zero means predictions all have std=0 — the model is
emitting a constant for every input. The first epoch sees loss=7+ (the
initial output is well outside [0,1]) and Adam yanks the network into a
basin where the FC head outputs a fixed value to keep the gradient calm.
Once stuck there, BN and the residual stream keep it stuck.

At lr=1e-5 (50 epochs):

|  epoch | mean_px | accuracy | pearson_cx | pearson_cy |
|-------:|--------:|---------:|-----------:|-----------:|
|      1 |    93.2 |    0.365 |       0.13 |      −0.07 |
|     10 |    50.2 |    0.415 |       0.60 |       0.67 |
|     25 |    40.8 |    0.380 |       0.80 |       0.78 |
|     50 |    34.3 |    0.355 |       0.86 |       0.84 |

Now it learns. The lesson: lr=1e-4 is fine for the ~8M-parameter `simple`
encoder, but ResNet18-spatial is ~20M parameters and needs an order of
magnitude smaller lr or initial gradient norm clipping.

## Are these the numbers we should expect?

**Localization on `simple`**: yes, basically. 16 px median = 6% of the
canvas; Pearson 0.97 means the model is tracking the shape well, not
regressing to the mean. The residual error is consistent with the
prediction noise floor of an 8M-parameter raw-regression FC head on a
relatively low-resolution feature map (16x16 spatial grid → 16 px is
essentially "one feature cell").

**Classification on `simple`**: not yet. 81% on 3 classes after 1000
epochs is well above chance (33%) but a long way from solved. Random
rotation is the genuine difficulty: a small filled triangle rotated 90°
on a low-contrast background really does look like a rectangle to a
small CNN. Random colors remove the easy color cue. With a bigger model
or a better feature representation this would land at >95%.

**Architecture ordering**: I expected `resnet > simple_bn > simple`.
What I got was `simple > simple_bn` (BN hurts) and ResNet was either
broken at the wrong lr or about even with `simple` at the right lr. Two
caveats:

1. ResNet hasn't been run long enough at lr=1e-5 to see the steady-state
   accuracy. 50 epochs is "warmup" for it; 500+ would be a fair test.
2. Both BN issues are diagnosable but I haven't fixed them.

## Concrete things that should happen next

In rough order of payoff:

1. **Auto-pick a reasonable lr per encoder** — at minimum, halve the lr
   when the encoder name starts with `resnet`. The current single
   `--lr` default papers over a real difference.
2. **Long ResNet run** — `resnet18_spatial` at lr=1e-5 for 1000+ epochs.
   This is the experiment that actually answers "does a bigger model
   beat the small one on this task?". Probably 20-30 minutes of GPU.
3. **Fix BN behavior on this task** — likely either (a) increase batch
   size so running stats are stable, (b) freeze BN running stats after
   warmup, or (c) replace BN with GroupNorm. The fact that
   `simple > simple_bn` in our setup is a real bug in BN configuration,
   not a fundamental limitation of BN.
4. **Sweep `lambda_class`** — single-object accuracy is climbing slowly
   in part because the localization MSE is much smaller than the class
   CE, so the optimizer mostly tunes the localization head. Try 3.0, 5.0
   on single-object too.
5. **Bigger val set** — the 200-image val set has ~6% std on accuracy
   per epoch, which is why the accuracy curve looks noisy in earlier
   runs. Bumping val to 1000 (which the long run already used) is
   cheap and gives clean curves.

## What should "fully solved" look like

For 256x256 single-shape, 3-class, random rotation/color, fresh data:

- Localization: median error <= 2 px (one feature cell of a 128x128
  feature map) — needs a moderate-resolution feature head, not 16x16.
- Classification: >= 99% — a ResNet-style backbone with global pool
  for the class head and spatial features for the localization head.

We're at 16 px / 81% with the small model. To reach <=2 px / 99%, we
likely need: bigger encoder (ResNet18 with the lr fixed), separate
localization and classification heads, and probably a heatmap-style
localization head.
