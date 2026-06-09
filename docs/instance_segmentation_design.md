# Instance segmentation — design (scoped, not yet implemented)

Semantic segmentation (`--task segmentation`) labels each pixel with a *class*.
Instance segmentation additionally separates *individual shapes* — two
overlapping rectangles are one semantic region but two instances. This is the
natural next step; the pieces are mostly already here. `InstanceSegNet`
(`models/instance_seg_net.py`) is a stub that points back to this doc.

## Ground truth is (still) free

P1 already composites a per-pixel **class** map in the rasterizer's shape loop.
An **instance-id** map is the same one-liner with the slot index instead of the
class:

```python
# render_batch / generate_image, in the existing per-shape composite loop:
labels   = torch.where(chosen, class_of_slot,    labels)    # already exists (semantic)
instances = torch.where(chosen, slot_index + 1,  instances) # add this (0 = background)
```

So instance GT is `(class_map, instance_map)` — pixel-perfect, draw-order
correct, free. Plumb it through `with_masks` exactly like the class map (a third
tensor in the batch, or stack the two into a (B, 2, H, W) target).

## Model — three options

1. **Detect-then-segment (recommended).** Reuse the already-solved
   `MultiHeatmapNet` detector to find shape centers, add a small mask head that,
   for each detected center, predicts a binary mask (e.g. sample the decoder
   feature at the peak and decode a local mask, CenterNet/CondInst-style). The
   detector is done, so this is the smallest delta and degrades gracefully to
   the semantic result.
2. **Associative embedding / panoptic.** Predict a per-pixel embedding alongside
   the semantic logits; pixels of one instance pull together, different
   instances push apart (push/pull loss). Group at inference. No fixed instance
   cap, but the loss is fiddlier.
3. **Fixed-slot masks (DETR-style).** Predict `K` instance mask channels + class
   + objectness, Hungarian-match to GT instances (exactly the `CenterPredictor`
   matching, but the cost includes mask IoU). Clean but `K`-capped and heavier.

Recommendation: **(1)**, because the detector half already converges and the
project's premise is "smallest model that fully solves it."

## Loss
- Semantic CE/Dice (reuse `SegLoss`) for the class map.
- Per-instance mask loss (BCE/Dice) on matched instances. For option (1) the
  match is the detector's center→GT assignment; for (3) it's Hungarian on
  (class + center + mask-IoU) cost.

## Metrics
- **Mask AP** (per-instance IoU thresholds), reusing the detection AP machinery
  (`_center_average_precision`) with IoU in place of center distance.
- **Panoptic Quality (PQ)** = segmentation quality x recognition quality, the
  standard single number combining semantic + instance.
- Add a `seg/` → `inst/` metrics block in `utils/metrics.py` mirroring
  `SegmentationMetrics`.

## Why it's deferred
Semantic seg + the existing `multi_heatmap` detector are the building blocks and
both converge. Instance seg adds: the instance-id GT (trivial), a mask head + an
instance loss, and a mask-AP/PQ metric — a self-contained follow-up that doesn't
change any existing task. Track it as its own phase (model → loss → metric →
`--task instance_seg` → smoke), the same shape as the semantic-seg track.
