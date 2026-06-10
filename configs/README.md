# Study configs

A study config is a self-contained, declarative spec for one run:
run/model scalars at the top level, and the **train** and **val** data
distributions as `DatasetConfig` blocks. `val` inherits every field of
`train` and overrides only what differs — so a distribution-shift study is
just the one knob that changes.

```bash
python -m scripts.run_training --config configs/train_clean_eval_rotated.yaml
```

When `--config` is given it is authoritative (the other CLI flags are
ignored). Any field you omit falls back to the per-task default, so configs
stay short. `task:` can be any of the seven (`single`, `multi`, `heatmap`,
`multi_heatmap`, `segmentation`, `instance_seg`, `classification`).

## Examples here

- `train_clean_eval_rotated.yaml` — train without rotation, validate with it.
- `train_clean_eval_noisy.yaml` — train clean, validate on noisy images (CPU path).
- `segment_zero_shot.yaml` — semantic segmentation trained on rectangles +
  circles only, then evaluated on a set that also contains triangles (does it
  segment a shape class it never saw? watch `seg/iou/TRIANGLE`).
- `instance_seg.yaml` — instance segmentation (separates individual shapes).

Every `DatasetConfig` knob is settable per distribution:

| key | values | meaning |
|---|---|---|
| `num_shapes_range` | `[min, max]` | shapes per image |
| `shape_size_range` | `[min, max]` | shape size in px (capped at image/2) |
| `shape_types` | `[RECTANGLE, CIRCLE, TRIANGLE]` | which shapes to draw |
| `rotate_shapes` | `true` / `false` | rotate rectangles & triangles |
| `background` | `solid` / `texture` / `random` | background type (non-solid is CPU-only) |
| `shape_outline` | `fill` / `thin` / `thick` / `random` | filled vs outlined (non-fill is CPU-only) |
| `add_noise` | `true` / `false` | additive Gaussian noise (CPU-only) |
| `blur` | `0.0`+ | Gaussian blur radius in px (0 = off; CPU-only) |
| `color_threshold` | `0`–`441` | min RGB distance of a shape's colour from the background |
| `max_overlap` | `0.0`–`1.0` | max allowed inter-shape overlap |

`texture` / non-`fill` outlines / `add_noise` / `blur` require the CPU path
(`gpu_data: false`); the GPU rasterizer supports solid + filled only.
`color_threshold` works on both paths. Segmentation masks support outlined
shapes (CPU); the mask draws the same outline so it matches the image.
