import glob
import json
import math
import os
import random
from dataclasses import asdict
from typing import Any, Callable, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from data.annotations import (
    Annotation,
    AnnotationEncoder,
    BackgroundType,
    BoundingBox,
    ShapeOutline,
    ShapeType,
    validate_shape_size_range,
)

rgb_color_type = tuple[int, int, int]


def add_gaussian_noise(image: Image.Image,
                       mean: float = 0,
                       std: float = 10,
                       np_rng: Optional[np.random.Generator] = None,
                       ) -> Image.Image:
    rng = np_rng if np_rng is not None else np.random.default_rng()
    np_img = np.array(image).astype(np.float32)
    noise = rng.normal(mean, std, np_img.shape)
    np_img += noise
    np_img = np.clip(np_img, 0, 255).astype(np.uint8)
    return Image.fromarray(np_img)


def compute_overlap_ratio(box1: BoundingBox, box2: BoundingBox) -> float:
    """Intersection area divided by the smaller of the two box areas.

    This is *not* IoU. It measures "how much of the smaller box is contained
    in the larger one" — 1.0 means the smaller box is fully inside the larger.
    Used to keep generated shapes from sitting on top of each other.
    """
    x_left = max(box1.x_min, box2.x_min)
    y_top = max(box1.y_min, box2.y_min)
    x_right = min(box1.x_max, box2.x_max)
    y_bottom = min(box1.y_max, box2.y_max)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1.x_max - box1.x_min) * (box1.y_max - box1.y_min)
    area2 = (box2.x_max - box2.x_min) * (box2.y_max - box2.y_min)

    min_area = min(area1, area2)
    return intersection_area / min_area if min_area > 0 else 0.0


def is_overlapping(new_box: BoundingBox,
                   existing_boxes: list[BoundingBox],
                   max_overlap_ratio: float = 0.3) -> bool:
    """True if `new_box` overlaps any existing box by more than the threshold.

    The threshold is on the per-pair `compute_overlap_ratio` (intersection
    over min area), not on a sum across pairs — summing ratios across
    unrelated boxes was meaningless and caused spurious rejections when the
    canvas got crowded.
    """
    return any(
        compute_overlap_ratio(new_box, box) > max_overlap_ratio
        for box in existing_boxes
    )


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` that reseeds ShapeDataset RNGs per worker.

    PyTorch already seeds ``random`` and its own RNG per worker, but it does
    *not* seed ``numpy.random``, and our dataset keeps its own
    ``random.Random`` and ``numpy.random.Generator`` instances so a single
    ``seed=`` argument produces reproducible images. Without this hook, each
    worker holds an identical copy of those RNGs after fork/spawn and ends
    up emitting duplicate images.

    Pass as ``DataLoader(..., worker_init_fn=seed_worker)`` whenever
    ``num_workers > 0``. Determinism across runs is the user's
    responsibility (set ``torch.manual_seed`` and/or pass a ``generator=``
    to the DataLoader before workers spawn).
    """
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    base_seed = int(info.seed) % (2 ** 32)
    dataset = info.dataset
    if hasattr(dataset, "_rng"):
        dataset._rng = random.Random(base_seed)
    if hasattr(dataset, "_np_rng"):
        dataset._np_rng = np.random.default_rng(base_seed)


def color_distance(c1: rgb_color_type, c2: rgb_color_type) -> float:
    # Euclidean distance in RGB space.
    return cast("float", sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5)


def select_shape_color(bg_color: rgb_color_type,
                       threshold: float = 50,
                       rng: Optional[random.Random] = None) -> rgb_color_type:
    """Pick a color that is sufficiently different from the background."""
    r = rng if rng is not None else random
    candidate = cast("rgb_color_type", tuple(r.randint(0, 255) for _ in range(3)))
    for _ in range(10):
        candidate = cast(
            "rgb_color_type",
            tuple(r.randint(0, 255) for _ in range(3))
        )
        if color_distance(candidate, bg_color) > threshold:
            return candidate
    return candidate


def rotate_point(x: float,
                 y: float,
                 cx: float,
                 cy: float,
                 angle_rad: float) -> tuple[int, int]:
    dx: float = x - cx
    dy: float = y - cy
    x_new: int = int(round(
        cx + dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
    ))
    y_new: int = int(round(
        cy + dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
    ))
    return x_new, y_new


class ShapeDataset(Dataset[tuple[Any, list[Annotation]]]):
    def __init__(self,
                 num_images: int = 1000,
                 image_size: tuple[int, int] = (256, 256),
                 num_shapes_range: tuple[int, int] = (0, 3),
                 shape_size_range: tuple[int, int] = (20, 90),
                 background: BackgroundType = BackgroundType.SOLID,
                 add_noise: bool = False,
                 fixed_dataset: bool = False,
                 shape_types: tuple[ShapeType, ...] = tuple(ShapeType),
                 shape_outline: ShapeOutline = ShapeOutline.RANDOM,
                 rotate_shapes: bool = True,
                 max_overlap: float = 0.6,
                 transform: Optional[Callable[[Any], Any]] = None,
                 save_location: str = "shape_dataset",
                 seed: Optional[int] = None,
                 with_masks: bool = False,
                 with_instances: bool = False,
                 blur: float = 0.0,
                 color_threshold: float = 50.0,):
        validate_shape_size_range(image_size, shape_size_range)
        need_label = with_masks or with_instances
        if need_label and fixed_dataset:
            # The fixed-dataset cache only stores (img, annotations); masks
            # aren't persisted, so a cached item would return a None mask.
            raise NotImplementedError(
                "with_masks/with_instances are not supported with fixed_dataset=True"
            )
        self.with_masks = with_masks
        self.with_instances = with_instances
        self.blur = blur
        self.color_threshold = color_threshold
        self.num_images = num_images
        self.image_size = image_size
        self.num_shapes_range = num_shapes_range
        self.shape_size_range = shape_size_range
        self.background = background
        self.add_noise = add_noise
        self.fixed_dataset = fixed_dataset
        self.shape_types = shape_types
        self.shape_outline = shape_outline
        self.rotate_shapes = rotate_shapes
        self.max_overlap = max_overlap
        self.transform = transform
        self.save_location = save_location
        # Per-instance RNGs so a fixed seed gives a reproducible dataset even
        # when other code (or other workers) touch the global random state.
        self._seed = seed
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        if self.fixed_dataset:
            images_path = os.path.join(self.save_location, "images")
            annotations_path = os.path.join(self.save_location, "annotations")
            self.data = []
            if os.path.exists(images_path) and os.path.exists(annotations_path):
                image_paths = sorted(glob.glob(os.path.join(images_path, "*.png")))
                if len(image_paths) != num_images:
                    raise ValueError(
                        f"Requested num_images={num_images} but found "
                        f"{len(image_paths)} png files in {images_path}. "
                        f"Either delete {self.save_location} to regenerate, "
                        f"or pass num_images={len(image_paths)}."
                    )
                print(f"loading {len(image_paths)} images and annotations from {self.save_location}")
                for image_file in image_paths:
                    annotation_file = os.path.join(
                        annotations_path,
                        os.path.basename(image_file).replace(".png", ".json"),
                    )
                    with open(annotation_file, 'r') as f:
                        annotations = json.load(f)
                    converted_annotations = [
                        Annotation.from_dict(annotation)
                        for annotation in annotations
                    ]
                    with Image.open(image_file) as raw:
                        image = raw.convert("RGB")
                    self.data.append((image, converted_annotations))
            else:
                os.makedirs(images_path, exist_ok=True)
                os.makedirs(annotations_path, exist_ok=True)
                for i in range(num_images):
                    img, annotations, _ = self.generate_image()
                    self.data.append((img, annotations))
                    img.save(os.path.join(images_path, f"{i}.png"))
                    with open(os.path.join(annotations_path, f'{i}.json'), 'w') as f:
                        json.dump(annotations, f, cls=AnnotationEncoder, indent=4)

    def __len__(self) -> int:
        return self.num_images

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
        mask: Optional[torch.Tensor] = None
        if self.fixed_dataset:
            img, ann = self.data[idx]
        else:
            # When `seed` is set, re-derive the RNGs from `seed + idx` so the
            # mapping idx -> image is deterministic across epochs and across
            # processes. Without this, `_rng` advances on every call, so the
            # same idx returns different images on the second epoch and the
            # val set silently drifts (this was the cause of the noisy val
            # numbers we saw — see claude_project_notes/2026-05-01_*.md).
            if self._seed is not None:
                per_idx_seed = (self._seed + int(idx)) & 0xFFFFFFFF
                self._rng = random.Random(per_idx_seed)
                self._np_rng = np.random.default_rng(per_idx_seed)
            img, ann, mask = self.generate_image()
        if self.transform:
            img = self.transform(img)
        if self.with_masks or self.with_instances:
            return img, ann, mask
        return img, ann

    @staticmethod
    def collate_function(batch: list[tuple[Any, ...]]) -> tuple[Any, ...]:
        # Each batch item is (image, annotations) or, with masks,
        # (image, annotations, mask).
        has_masks = len(batch[0]) == 3
        if has_masks:
            images, annotations, masks = zip(*batch)
        else:
            images, annotations = zip(*batch)

        # Convert images using the default collate function (they're tensors)
        images = default_collate(images)

        def annotation_to_dict(ann: Annotation) -> dict[Any, Any]:
            # Convert the dataclass to a dict.
            d = asdict(ann)
            # Convert the ShapeType enum to its name (or use .value if preferred).
            if 'shape' in d and isinstance(d['shape'], ShapeType):
                d['shape'] = d['shape'].value
            return d

        new_annotations = []
        for ann in annotations:
            new_annotations.append([annotation_to_dict(a) for a in ann])

        if has_masks:
            return images, new_annotations, torch.stack(list(masks))
        return images, new_annotations

    @staticmethod
    def compute_bbox_from_shape_points(
        points: list[tuple[int, int]]
    ) -> tuple[int, int, int, int]:
        """
        Compute the axis-aligned bounding box for a list of points.

        Args:
            points: A list of (x, y) tuples representing
                the vertices of a shape.

        Returns:
            A tuple (min_x, min_y, max_x, max_y) representing
                the bounding box.
        """
        xs: list[int] = [pt[0] for pt in points]
        ys: list[int] = [pt[1] for pt in points]
        return (min(xs), min(ys), max(xs), max(ys))

    def rotate_shape_points(self,
                            shape_points: list[tuple[int, int]],
                            center: tuple[int | float, int | float],
                            max_attempts: int = 10
                            ) -> tuple[list[tuple[int, int]], BoundingBox]:
        """
        Attempts to find a rotation angle (in degrees) such that the rotated shape
        is entirely within the image boundaries.
        """
        width, height = self.image_size
        max_angle = 360.0
        for _ in range(max_attempts):
            angle = self._rng.uniform(0, max_angle)
            angle_rad = math.radians(angle)
            rotated_corners = [rotate_point(x, y, center[0], center[1], angle_rad) for (x, y) in shape_points]
            bbox = BoundingBox(*self.compute_bbox_from_shape_points(rotated_corners))
            if bbox.x_min >= 0 and bbox.y_min >= 0 and bbox.x_max <= width and bbox.y_max <= height:
                return rotated_corners, bbox
            max_angle *= 0.25

        bbox = BoundingBox(*self.compute_bbox_from_shape_points(shape_points))
        return shape_points, bbox

    def convert_center_to_image_coordinates(self, center: tuple[float, float]) -> tuple[int, int]:
        return (int(round(center[0] * self.image_size[0])),
                int(round(center[1] * self.image_size[1])))

    def generate_image(
        self,
    ) -> tuple[Image.Image, list[Annotation], Optional[torch.Tensor]]:
        rng = self._rng
        np_rng = self._np_rng

        background = self.background
        if self.background == BackgroundType.RANDOM:
            # Pick from non-RANDOM members using our seeded rng so the
            # background choice is reproducible.
            choices = [m for m in BackgroundType if m != BackgroundType.RANDOM]
            background = rng.choice(choices)

        if background == BackgroundType.SOLID:
            bg_color = tuple(rng.randint(0, 255) for _ in range(3))
            img = Image.new('RGB', self.image_size, bg_color)
        elif background == BackgroundType.TEXTURE:
            noise_array = np.uint8(
                np_rng.random((self.image_size[1], self.image_size[0], 3)) * 255
            )
            img = Image.fromarray(noise_array, 'RGB')
            bg_color = tuple(int(x) for x in np.mean(noise_array, axis=(0, 1)).astype(np.uint8))
        else:
            raise ValueError(f"unknown background type: {background}")
        bg_color = cast("tuple[int, int, int]", bg_color)

        annotations = []

        num_shapes = rng.randint(*self.num_shapes_range)
        existing_boxes: list[BoundingBox] = []
        need_label = self.with_masks or self.with_instances
        # (draw-primitive, coords, class, use_fill, outline_width) per shape, for
        # the label map. Recorded in draw order and rendered onto a label canvas
        # below, mirroring the image's fill/outline so the mask matches it.
        mask_specs: list[tuple[str, Any, int, bool, int]] = []
        for _ in range(num_shapes):
            shape_type = rng.choice(self.shape_types)
            shape_color = select_shape_color(bg_color, threshold=self.color_threshold, rng=rng)

            min_size, max_size = self.shape_size_range
            max_width = min(max_size, self.image_size[0] // 2)
            max_height = min(max_size, self.image_size[1] // 2)
            shape_width = rng.randint(min_size, max_width)
            shape_height = rng.randint(min_size, max_height)

            num_trys = 10
            bbox = None
            for _ in range(num_trys):
                x0 = rng.randint(0, self.image_size[0] - shape_width)
                y0 = rng.randint(0, self.image_size[1] - shape_height)
                x1 = x0 + shape_width
                y1 = y0 + shape_height

                if not is_overlapping(
                    BoundingBox(x0, y0, x1, y1),
                    existing_boxes,
                    max_overlap_ratio=self.max_overlap,
                ):
                    bbox = BoundingBox(x0, y0, x1, y1)
                    existing_boxes.append(bbox)
                    break
            if bbox is None:
                print(f"Could not find non-overlapping {shape_type} in {num_trys} iterations")
                continue

            draw = ImageDraw.Draw(img)
            max_outline = min(shape_width, shape_height)
            max_outline = int(round(max_outline / 2))
            if self.shape_outline == ShapeOutline.RANDOM:
                outline_width = rng.randint(ShapeOutline.THIN.value, max_outline)
            else:
                outline_width = self.shape_outline.value
            use_fill = self.shape_outline == ShapeOutline.FILL or outline_width == max_outline

            if shape_type == ShapeType.RECTANGLE:
                if self.rotate_shapes:
                    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                    center = ((x0 + x1) / 2, (y0 + y1) / 2)
                    rotated_corners, bbox = self.rotate_shape_points(corners, center)
                    if need_label:
                        mask_specs.append(("polygon", rotated_corners, shape_type.value, use_fill, outline_width))
                    if self.shape_outline == ShapeOutline.FILL or outline_width == max_outline:
                        draw.polygon(
                            rotated_corners,
                            fill=shape_color
                        )
                    else:
                        draw.polygon(
                            rotated_corners,
                            outline=shape_color,
                            width=outline_width
                        )
                else:
                    if need_label:
                        mask_specs.append(("rectangle", list(bbox), shape_type.value, use_fill, outline_width))
                    if self.shape_outline == ShapeOutline.FILL or outline_width == max_outline:
                        draw.rectangle(
                            bbox,
                            fill=shape_color
                        )
                    else:
                        draw.rectangle(
                            bbox,
                            outline=shape_color,
                            width=outline_width
                        )
            elif shape_type == ShapeType.CIRCLE:
                if need_label:
                    mask_specs.append(("ellipse", list(bbox), shape_type.value, use_fill, outline_width))
                if self.shape_outline == ShapeOutline.FILL or outline_width == max_outline:
                    draw.ellipse(
                        bbox,
                        fill=shape_color
                    )
                else:
                    draw.ellipse(
                        bbox,
                        outline=shape_color,
                        width=outline_width
                    )
            elif shape_type == ShapeType.TRIANGLE:
                point1 = (x0 + rng.randint(0, shape_width), y0)
                point2 = (x0, y1)
                point3 = (x1, y1)
                points = [point1, point2, point3]
                center = (
                    (point1[0] + point2[0] + point3[0]) / 3,
                    (point1[1] + point2[1] + point3[1]) / 3
                )
                if self.rotate_shapes:
                    points, bbox = self.rotate_shape_points(points, center)
                if need_label:
                    mask_specs.append(("polygon", points, shape_type.value, use_fill, outline_width))
                if self.shape_outline == ShapeOutline.FILL or outline_width == max_outline:
                    draw.polygon(
                        points,
                        fill=shape_color
                    )
                else:
                    draw.polygon(
                        points,
                        outline=shape_color,
                        width=outline_width
                    )
            else:
                assert False, "Wrong shape type"
            if shape_type != ShapeType.TRIANGLE:
                center = ((x0 + x1) / 2, (y0 + y1) / 2)

            center = (center[0] / self.image_size[0], center[1] / self.image_size[1])
            ann = Annotation(shape=shape_type, bbox=bbox, center=center, color=shape_color)
            annotations.append(ann)

        if self.add_noise:
            img = add_gaussian_noise(img, np_rng=np_rng)
        if self.blur > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=self.blur))

        label = None
        if self.with_masks or self.with_instances:
            # Render the recorded shapes onto a label canvas in the same draw
            # order (and the same fill/outline), so the mask matches the image
            # pixel-for-pixel including overlaps and outlined shapes. Semantic:
            # fill with the class, background=len(ShapeType). Instance: fill with
            # the 1-based shape index, background=0. (Blur is NOT applied to the
            # label — labels stay crisp.)
            bg = 0 if self.with_instances else len(ShapeType)
            label_img = Image.new("L", self.image_size, bg)
            label_draw = ImageDraw.Draw(label_img)
            for inst_idx, (kind, coords, cls, use_fill, width) in enumerate(mask_specs):
                value = (inst_idx + 1) if self.with_instances else cls
                if use_fill:
                    getattr(label_draw, kind)(coords, fill=value)
                else:
                    getattr(label_draw, kind)(coords, outline=value, width=width)
            label = torch.from_numpy(np.array(label_img, dtype=np.int64))  # (H, W)

        return img, annotations, label

    def get_classes(self) -> list[str]:
        """
        Returns a list of shape class names that the dataset can generate.
        """
        return [shape.name for shape in self.shape_types]


# __main__ block to quickly test the dataset
if __name__ == '__main__':
    # Instantiate the dataset with a small number of images for quick testing.
    dataset = ShapeDataset(num_images=10,
                           image_size=(256, 256),
                           num_shapes_range=(1, 10),
                           shape_size_range=(20, 100),
                           background=BackgroundType.RANDOM,
                           shape_outline=ShapeOutline.RANDOM,
                           add_noise=True,
                           fixed_dataset=False)

    # Retrieve one sample from the dataset
    img, annotations = dataset[0]

    # Convert image to a NumPy array for display if it's a PIL image.
    np_img = np.array(img)

    # Use Matplotlib to display the image along with its annotations.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np_img)

    # Loop over annotations and draw the bounding boxes and center points.
    for ann in annotations:
        bbox = ann.bbox
        center = ann.center
        # Draw the bounding box.
        rect = patches.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1],
                                 linewidth=2,
                                 edgecolor='red',
                                 facecolor='none')
        ax.add_patch(rect)
        # Draw the center point.
        ax.plot(*dataset.convert_center_to_image_coordinates(center), 'bo')
        # Optionally, add the shape type as a label.
        ax.text(
            bbox[0], bbox[1] - 5,
            ann.shape.name,
            color='yellow',
            fontsize=8,
            backgroundcolor='black'
        )

    ax.set_title("Sample from ShapeDataset")
    ax.axis('off')
    plt.show()
