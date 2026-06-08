# GPU utilization + training-convergence analysis

Date: 2026-05-01
Hardware: RTX 3090 (24 GB, compute capability 8.6), single-CPU image generation
What we ran: short training experiments using the new `scripts.run_training`
+ `scripts.gpu_monitor` tools, all on the single-object center-prediction task.

## Headline findings

1. **GPU is severely underutilized** — averaged 9.8% utilization with 1.7 GB of
   24 GB memory in use. The bottleneck is CPU image generation, not GPU compute.
2. **The "small custom model not doing as well as expected" puzzle is real but
   not what it looks like.** Across encoder architectures (simple, simple_gap,
   simple_bn, simple_bn_gap, resnet18) the *training loss* plateaus at the
   same ~1.14 floor and `val/mean_center_px` stays around 70 px. This *looked
   like* "model can't learn", but it's actually:
   - The model **is** learning. Pearson correlation between prediction and
     target after 30 epochs of `simple_bn` is 0.68 for cx and 0.60 for cy.
   - It's converging towards a **shrunk** estimate: prediction std is half the
     target std. So predictions land on the correct side of center but
     underestimate how far the shape is from the middle.
   - The reported `mean_center_px ≈ 70` is between "predict the constant
     mean" (~80 px) and "predict perfectly" (0 px); given the pearson values,
     it's exactly where you'd expect a half-converged regressor.
   - The loss number stays at 1.14 because cross-entropy at 3-way uniform is
     ln(3)=1.099 and MSE on the shrunk regressor is ~0.04 — the bulk of the
     loss is the class term, not the localization term. Tracking
     `mean_center_px` and pearson correlation directly is much more
     informative.

## Throughput and GPU stats

| Run | Encoder | Params | epoch_s | images/s | GPU util mean | GPU util max |
|-----|---------|--------|---------|----------|---------------|--------------|
| short | simple_gap   |    132K | 0.47 | 2,150 | 9.8% (167 samples) | 77% |
| short | simple       |    8.4M | 0.47 | 2,140 | similar | similar |
| short | simple_bn    |    8.4M | 0.48 | 2,090 | similar | similar |
| short | resnet18     |     11M | 1.13 |   880 | not collected | — |
| long  | simple_bn (200ep) | 8.4M | 0.46 | 2,170 | similar | similar |

The fact that a 132K-parameter model and an 8.4M-parameter model both run at
~2,150 images/s on a 3090 confirms the GPU is sitting idle waiting for the
DataLoader. ResNet18 is twice as slow because the conv pipeline finally has
enough work to overlap with data loading.

## Why the GPU is idle

`scripts/run_training.py` already turned on the easy GPU wins (`pin_memory`,
`non_blocking=True`, `cudnn.benchmark`, TF32 matmul) in commit `d71ca51`. The
remaining bottleneck is **CPU image generation in the workers**. Each batch
of 100 images requires ~100 PIL `ImageDraw` calls (background, ellipse /
polygon, optional rotation) plus tensor conversion. With 4 workers, peak
throughput is ~2,150 images/s; that means each step waits while the workers
catch up before the GPU has anything to consume.

Effective fixes, in order of expected impact:

1. **Pre-generate fixed datasets and cache to disk** for short experiments.
   The dataset already supports `fixed_dataset=True`. A 10K-image cache loads
   from disk much faster than generating from scratch on every iteration.
2. **Vectorize image generation in NumPy / PIL.** The current code does one
   primitive per shape via PIL; batched NumPy color-fill primitives would be
   much faster.
3. **Move generation to the GPU** (e.g. with `kornia` or a custom torch op).
   This is the highest-throughput option but the largest engineering lift.
4. **`prefetch_factor=4-8`** on the DataLoader. Default is 2; raising this
   gives the workers more lookahead.
5. **Larger batch sizes** to amortize per-step Python overhead. Memory headroom
   is huge (1.7 GB / 24 GB used).

## Why the training looks broken

Three things compound to produce the impression that nothing is learning:

1. **Train loss is dominated by classification.** With 3 classes and
   uniform-random predictions, CE = ln(3) ≈ 1.099. With a half-converged
   regressor on normalized centers, MSE is ~0.04. Total = ~1.14, which is
   what we see. The class loss has to be driven down before the total
   number moves much, but the localization signal is improving meanwhile —
   it just isn't visible in the headline scalar.
2. **Class accuracy is near chance because shape classification under
   random rotation, random color, and random size is genuinely hard for a
   small CNN at 50–200 epochs of 1000 samples/epoch.** The model can
   discriminate triangles from circles given enough capacity and data, but
   not in a few hundred gradient steps.
3. **The val set drifts across epochs.** `ShapeDataset` uses a single
   per-instance `random.Random(seed)` whose state advances on every
   `__getitem__` call. After one full val pass the state has moved, so
   epoch N+1 sees different images. The val numbers fluctuate by ~5 px
   epoch to epoch as a result, which compounds the "stuck" appearance.

### Evidence that learning is happening

After 30 epochs of `simple_bn` at lr=1e-3 (3-class, rotation-on, single
shape, 256x256, batch 100):

```
predictions: cx mean=0.494 std=0.114 (range 0.239..0.756)
             cy mean=0.561 std=0.104
targets:     cx mean=0.483 std=0.218
             cy mean=0.545 std=0.221
pearson corr (pred, target): cx=0.6804  cy=0.6042
```

A constant-prediction baseline would give pearson ≈ 0. Random predictions
would give pearson ≈ 0 too. We see 0.6+, with prediction std at roughly
half the target std — classic regression-to-the-mean of a partially
converged model.

## Concrete things to fix next

**For visibility (cheap):**

- Log `pearson_cx` / `pearson_cy` and a plain `mse_centers` scalar in
  `evaluate_single_object`. The current single number hides progress.
- Reset `_rng` / `_np_rng` at the start of every `__getitem__` when `seed`
  is set, derived from `seed + idx`. That makes the val set deterministic
  per-index and stable across epochs.
- Print loss decomposition (mse vs ce) per epoch so it's obvious which
  term is driving the total.

**For convergence (more involved):**

- Cosine LR schedule with a short warmup, peaking at lr=3e-3 or higher.
  The current ReduceLROnPlateau only kicks in after the loss stalls — but
  the loss already stalled at the floor we can't escape.
- Either remove the output sigmoid on centers OR use a Smooth-L1 (Huber)
  loss instead of MSE. Sigmoid + MSE has dramatically smaller gradients
  near the saturation tails than the unconstrained version.
- Add positional encoding (CoordConv-style: append coord channels to the
  input). For localization specifically, this is a known massive win.
- Try a heatmap head (predict a 2D Gaussian heatmap, take its expectation
  as the center). Removes the regression-to-the-mean failure mode entirely.

**For throughput (only if scaling experiments):**

- Pre-cache the dataset to disk for repeated short experiments.
- Vectorize PIL image generation, or use `kornia` for GPU-side rendering.

## Files added in this session

- `utils/perf.py` — `configure_for_speed()` + `pick_device()`.
- `models/encoders.py` — added `simple_bn` and `simple_bn_gap` variants.
- `scripts/run_training.py` — CLI training driver with `results.json` output.
- `scripts/gpu_monitor.py` — nvidia-smi sampler.
- `runs/_logs/gpu_*.csv` — sampled GPU stats from each experiment.
- `runs/<run_name>/results.json` — per-run config, throughput, final metrics.

## Next experiment to run (not done in this session)

1. Bump `--num-train-images 10000`, train 50 epochs → check if pearson > 0.9.
2. Add CoordConv channels in front of `simple_bn` → measure whether `mean_center_px` drops below 20.
3. Replace MSE-on-sigmoid with Huber-on-raw → measure convergence speed.
