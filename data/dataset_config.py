"""Declarative configuration for one synthetic-shape data distribution.

A study is "train on one distribution, measure on another." ``DatasetConfig``
captures every knob of a shape distribution (counts, sizes, rotation, colors,
background, outline, noise, overlap) so the train and val distributions can be
specified independently — in a YAML study config or in Python — instead of via
a growing pile of per-knob CLI flags. The same object builds either the CPU
``ShapeDataset`` or the GPU ``GpuShapeLoader``.

Inheritance: a val distribution usually differs from train in just one knob, so
``DatasetConfig.merged(overrides)`` returns a copy with only those fields
changed — that's how a study config's ``val:`` section inherits ``train:``.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

from data.annotations import BackgroundType, ShapeOutline, ShapeType

_ALL_SHAPES: tuple[str, ...] = tuple(s.name for s in ShapeType)


def _coerce(d: dict[str, Any]) -> dict[str, Any]:
    """Validate keys and normalize incoming values (lists -> tuples, shape
    names -> upper-case) so a config loaded from YAML/JSON matches the
    dataclass field types."""
    valid = {f.name for f in fields(DatasetConfig)}
    unknown = set(d) - valid
    if unknown:
        raise ValueError(
            f"unknown DatasetConfig keys: {sorted(unknown)} (valid: {sorted(valid)})"
        )
    # A null/None value means "not provided" — keep the inherited/default value
    # rather than overriding the field with None.
    out = {k: v for k, v in d.items() if v is not None}
    for k in ("num_shapes_range", "shape_size_range"):
        if k in out:
            out[k] = tuple(out[k])
    if "shape_types" in out:
        names = tuple(str(s).upper() for s in out["shape_types"])
        bad = [n for n in names if n not in ShapeType.__members__]
        if bad:
            raise ValueError(f"unknown shape_types {bad}; valid: {list(ShapeType.__members__)}")
        out["shape_types"] = names
    if "background" in out and str(out["background"]).upper() not in BackgroundType.__members__:
        raise ValueError(
            f"unknown background {out['background']!r}; "
            f"valid: {[m.lower() for m in BackgroundType.__members__]}"
        )
    if "shape_outline" in out and str(out["shape_outline"]).upper() not in ShapeOutline.__members__:
        raise ValueError(
            f"unknown shape_outline {out['shape_outline']!r}; "
            f"valid: {[m.lower() for m in ShapeOutline.__members__]}"
        )
    return out


@dataclass
class DatasetConfig:
    """All knobs of one shape distribution. Strings (background/outline/shape
    names) keep the config human-readable and JSON/YAML-round-trippable; the
    ``*_enum`` views convert to the enums the datasets expect."""
    num_shapes_range: tuple[int, int] = (0, 3)
    shape_size_range: tuple[int, int] = (20, 90)
    shape_types: tuple[str, ...] = _ALL_SHAPES
    rotate_shapes: bool = False
    background: str = "solid"      # solid | texture | random
    shape_outline: str = "fill"    # fill | thin | thick | random
    add_noise: bool = False
    max_overlap: float = 0.6

    @classmethod
    def default_for_task(
        cls, task: str, num_shapes_range: tuple[int, int] | None = None
    ) -> "DatasetConfig":
        """The historical per-task training distribution, used when no study
        config is supplied. Single-object tasks see one rotated shape spanning
        a wide size range; multi-object tasks see a variable count, no rotation.
        """
        if task in ("single", "heatmap"):
            return cls(num_shapes_range=(1, 1), shape_size_range=(20, 128), rotate_shapes=True)
        return cls(
            num_shapes_range=num_shapes_range or (0, 3),
            shape_size_range=(20, 90),
            rotate_shapes=False,
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "DatasetConfig":
        return cls(**_coerce(d or {}))

    def merged(self, overrides: dict[str, Any] | None) -> "DatasetConfig":
        """Copy of self with ``overrides`` applied — how ``val`` inherits ``train``."""
        if not overrides:
            return replace(self)
        return replace(self, **_coerce(overrides))

    def shape_type_enums(self) -> tuple[ShapeType, ...]:
        return tuple(ShapeType[s] for s in self.shape_types)

    def background_enum(self) -> BackgroundType:
        return BackgroundType[self.background.upper()]

    def outline_enum(self) -> ShapeOutline:
        return ShapeOutline[self.shape_outline.upper()]


def build_cpu_dataset(
    cfg: DatasetConfig,
    *,
    num_images: int,
    image_size: tuple[int, int],
    seed: int | None,
    transform: Any,
    with_masks: bool = False,
) -> Any:
    """Build a CPU ``ShapeDataset`` from a DatasetConfig."""
    from data.synthetic_shapes_dataset import ShapeDataset

    return ShapeDataset(
        num_images=num_images,
        seed=seed,
        image_size=image_size,
        num_shapes_range=cfg.num_shapes_range,
        shape_size_range=cfg.shape_size_range,
        shape_types=cfg.shape_type_enums(),
        background=cfg.background_enum(),
        shape_outline=cfg.outline_enum(),
        rotate_shapes=cfg.rotate_shapes,
        max_overlap=cfg.max_overlap,
        add_noise=cfg.add_noise,
        transform=transform,
        with_masks=with_masks,
    )


def build_gpu_loader(
    cfg: DatasetConfig,
    *,
    batch_size: int,
    num_images: int,
    image_size: tuple[int, int],
    seed: int | None,
    device: Any,
    reseed_each_epoch: bool = False,
    with_masks: bool = False,
) -> Any:
    """Build a GPU ``GpuShapeLoader`` from a DatasetConfig.

    The GPU rasterizer only supports solid backgrounds, filled shapes, and no
    additive noise; configs that ask for more must use the CPU path.
    """
    from data.gpu_shapes import GpuShapeLoader

    if cfg.background != "solid":
        raise NotImplementedError(
            f"gpu_data only supports solid backgrounds (got {cfg.background!r}); "
            "use the CPU path (gpu_data: false / drop --gpu-data) for backgrounds."
        )
    if cfg.shape_outline != "fill":
        raise NotImplementedError(
            f"gpu_data only supports filled shapes (got outline {cfg.shape_outline!r}); "
            "use the CPU path for outline studies."
        )
    if cfg.add_noise:
        raise NotImplementedError(
            "gpu_data does not support add_noise; use the CPU path for noise studies."
        )
    return GpuShapeLoader(
        batch_size=batch_size,
        num_images=num_images,
        image_size=image_size,
        seed=seed,
        num_shapes_range=cfg.num_shapes_range,
        shape_size_range=cfg.shape_size_range,
        shape_types=cfg.shape_type_enums(),
        rotate_shapes=cfg.rotate_shapes,
        max_overlap=cfg.max_overlap,
        device=device,
        reseed_each_epoch=reseed_each_epoch,
        with_masks=with_masks,
    )
