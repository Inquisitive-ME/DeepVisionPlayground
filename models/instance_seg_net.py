"""Instance segmentation — stub.

Semantic segmentation (``ShapeSegNet`` / ``--task segmentation``) labels each
pixel with a class; instance segmentation additionally separates individual
shapes. It is the natural next step and the pieces are mostly in place, but it
is not implemented yet.

See ``docs/instance_segmentation_design.md`` for the full design: the
instance-id ground-truth map is a one-line extension of the existing rasterizer
composite loop, and the recommended model reuses the already-converging
``MultiHeatmapNet`` detector plus a small per-center mask head.
"""
from __future__ import annotations

import torch.nn as nn

_DESIGN_DOC = "docs/instance_segmentation_design.md"


class InstanceSegNet(nn.Module):
    """Placeholder for the instance-segmentation model (see ``_DESIGN_DOC``)."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "Instance segmentation is scoped but not implemented yet; "
            f"see {_DESIGN_DOC} for the design and the recommended next steps."
        )
