import math
import random
from dataclasses import asdict
from typing import Any, Callable, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from data.annotations import Annotation, BackgroundType, BoundingBox, ShapeOutline, ShapeType

rgb_color_type = tuple[int, int, int]


def add_gaussian_noise(image: Image.Image,
                       mean: float = 0,
                       std: float = 10
                       ) -> Image.Image:
    np_img = np.array(image).astype(np.float32)
    noise = np.random.normal(mean, std, np_img.shape)
    np_img += noise
    np_img = np.clip(np_img, 0, 255).astype(np.uint8)
    return Image.fromarray(np_img)


def compute_overlap_ratio(box1: BoundingBox, box2: BoundingBox) -> float:
    """
    Compute the overlap ratio between two bounding boxes.
    The ratio is defined as the area of intersection divided by the area of the smaller box.
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
                   iou_threshold: float = 0.3) ->bool:
    total_overlap = 0.0
    for box in existing_boxes:
        overlap_ratio = compute_overlap_ratio(new_box, box)
        if overlap_ratio > iou_threshold:
            return True
        total_overlap += overlap_ratio
    if total_overlap > iou_threshold:
        return True
    return False


def color_distance(c1: rgb_color_type, c2: rgb_color_type) -> float:
    # Euclidean distance in RGB space.
    return cast(float, sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5)


def select_shape_color(bg_color: rgb_color_type,
                       threshold: float=50) -> rgb_color_type:
    # Try a few times to pick a color that is sufficiently different from the background.
    for _ in range(10):
        candidate = cast(rgb_color_type, tuple(random.randint(0, 255) for _ in range(3)))
        if color_distance(candidate, bg_color) > threshold:
            return candidate
    # If a sufficiently different color isn't found, just return a random color.
    return candidate


def rotate_point(x: float, y: float, cx: float, cy: float, angle_rad: float) -> tuple[int, int]:
    dx: float = x - cx
    dy: float = y - cy
    x_new: int = int(round(cx + dx * math.cos(angle_rad) - dy * math.sin(angle_rad)))
    y_new: int = int(round(cy + dx * math.sin(angle_rad) + dy * math.cos(angle_rad)))
    return x_new, y_new


class ShapeDataset(Dataset[tuple[Image.Image | torch.Tensor, list[Annotation]]]):
    def __init__(self,
                 num_images: int = 1000,
                 image_size: tuple[int, int] = (256, 256),
                 num_shapes_range: tuple[int, int] = (1, 3),
                 shape_size_range: tuple[int, int] = (10, 128),
                 background: BackgroundType = BackgroundType.SOLID,
                 add_noise: bool = False,
                 fixed_dataset: bool = False,
                 shape_types: tuple[ShapeType, ...] = tuple(ShapeType),
                 shape_outline: ShapeOutline = ShapeOutline.RANDOM,
                 rotate_shapes: bool = True,
                 max_overlap: float = 0.6,
                 transform: Optional[Callable[[Any], Any]] = None):
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

        if self.fixed_dataset:
            self.data = [self.generate_image() for _ in range(num_images)]

    def __len__(self) -> int:
        return self.num_images

    def __getitem__(self, idx: int) -> tuple[Image.Image | torch.Tensor, list[Annotation]]:
        if self.fixed_dataset:
            img, ann = self.data[idx]
        else:
            img, ann = self.generate_image()
        if self.transform:
            img = self.transform(img)
        return img, ann

    @staticmethod
    def collate_function(batch: list[tuple[torch.Tensor, dict[Any, Any]]]
                         ) -> tuple[torch.Tensor, list[list[dict[Any, Any]]]]:
        # Each batch item is a tuple: (image, annotations)
        images, annotations = zip(*batch)

        # Convert images using the default collate function (they're tensors)
        images = default_collate(images)  # type: ignore

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

        return images, new_annotations

    @staticmethod
    def compute_bbox_from_shape_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
        """
        Compute the axis-aligned bounding box for a list of points.

        Args:
            points: A list of (x, y) tuples representing the vertices of a shape.

        Returns:
            A tuple (min_x, min_y, max_x, max_y) representing the bounding box.
        """
        xs: list[int] = [pt[0] for pt in points]
        ys: list[int] = [pt[1] for pt in points]
        return (min(xs), min(ys), max(xs), max(ys))

    def convert_center_to_image_coordinates(self, center: tuple[float, float]) -> tuple[int, int]:
        return (int(round(center[0] * self.image_size[0])),
                int(round(center[1] * self.image_size[1])))

    def generate_image(self) -> tuple[Image.Image, list[Annotation]]:
        background = self.background
        if self.background == BackgroundType.RANDOM:
            background = BackgroundType.get_random_background()

        if background == BackgroundType.SOLID:
            bg_color = tuple(random.randint(0, 255) for _ in range(3))
            img = Image.new('RGB', self.image_size, bg_color)
        elif background == BackgroundType.TEXTURE:
            noise_array = np.uint8(np.random.rand(self.image_size[1], self.image_size[0], 3) * 255)
            img = Image.fromarray(noise_array, 'RGB')
            bg_color = tuple(int(x) for x in np.mean(noise_array, axis=(0, 1)).astype(np.uint8))
        else:
            assert False, "Wrong background type"
        bg_color = cast(tuple[int, int, int], bg_color)

        annotations = []

        num_shapes = random.randint(*self.num_shapes_range)
        existing_boxes: list[BoundingBox] = []
        for _ in range(num_shapes):
            shape_type = random.choice(self.shape_types)
            # Select a shape color that contrasts with the background.
            shape_color = select_shape_color(bg_color)

            min_size, max_size = self.shape_size_range
            max_width = min(max_size, self.image_size[0] // 2)
            max_height = min(max_size, self.image_size[1] // 2)
            shape_width = random.randint(min_size, max_width)
            shape_height = random.randint(min_size, max_height)

            num_trys = 10
            for _ in range(num_trys):
                x0 = random.randint(0, self.image_size[0] - shape_width)
                y0 = random.randint(0, self.image_size[1] - shape_height)
                x1 = x0 + shape_width
                y1 = y0 + shape_height
                bbox = None

                if not is_overlapping(BoundingBox(x0, y0, x1, y1), existing_boxes, iou_threshold=self.max_overlap):
                    bbox = BoundingBox(x0, y0, x1, y1)
                    existing_boxes.append(bbox)
                    break
            if bbox is None:
                print("Could not find non overlapping {} in {} iterations".format(shape_type, num_trys))
                continue

            draw = ImageDraw.Draw(img)
            max_outline = min(shape_width, shape_height)
            max_outline = int(round(max_outline / 2))
            if self.shape_outline == ShapeOutline.RANDOM:
                outline_width = random.randint(ShapeOutline.THIN.value, max_outline)
            else:
                outline_width = self.shape_outline.value

            if shape_type == ShapeType.RECTANGLE:
                if self.rotate_shapes:
                    angle = random.uniform(0, 360)
                    angle_rad = math.radians(angle)
                    # Define the corners of the rectangle.
                    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                    center = ((x0 + x1) / 2, (y0 + y1) / 2)
                    rotated_corners = [rotate_point(x, y, center[0], center[1], angle_rad) for (x, y) in corners]
                    bbox = BoundingBox(*self.compute_bbox_from_shape_points(rotated_corners))
                    if self.shape_outline == ShapeOutline.FILL or outline_width == max_outline:
                        draw.polygon(rotated_corners, fill=shape_color)
                    else:
                        draw.polygon(rotated_corners, outline=shape_color, width=outline_width)
                else:
                    if self.shape_outline == ShapeOutline.FILL or outline_width == max_outline:
                        draw.rectangle(bbox, fill=shape_color)
                    else:
                        draw.rectangle(bbox, outline=shape_color, width=outline_width)
            elif shape_type == ShapeType.CIRCLE:
                if self.shape_outline == ShapeOutline.FILL or outline_width == max_outline:
                    draw.ellipse(bbox, fill=shape_color)
                else:
                    draw.ellipse(bbox, outline=shape_color, width=outline_width)
            elif shape_type == ShapeType.TRIANGLE:
                point1 = (x0 + random.randint(0, shape_width), y0)
                point2 = (x0, y1)
                point3 = (x1, y1)
                points = [point1, point2, point3]
                center = ((point1[0] + point2[0] + point3[0]) / 3, (point1[1] + point2[1] + point3[1]) / 3)
                if self.rotate_shapes:
                    angle = random.uniform(0, 360)
                    angle_rad = math.radians(angle)
                    points = [rotate_point(x, y, center[0], center[1], angle_rad) for (x, y) in points]
                    bbox = BoundingBox(*self.compute_bbox_from_shape_points(points))
                if self.shape_outline == ShapeOutline.FILL or outline_width == max_outline:
                    draw.polygon(points, fill=shape_color)
                else:
                    draw.polygon(points, outline=shape_color, width=outline_width)
            else:
                assert False, "Wrong shape type"
            if shape_type != ShapeType.TRIANGLE:
                center = ((x0 + x1) / 2, (y0 + y1) / 2)

            center = (center[0] / self.image_size[0], center[1] / self.image_size[1])
            ann = Annotation(shape=shape_type, bbox=bbox, center=center, color=shape_color)
            annotations.append(ann)

        if self.add_noise:
            img = add_gaussian_noise(img)

        return img, annotations


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
                           num_shapes_range=(10, 10),
                           shape_size_range=(20, 100),
                           background=BackgroundType.TEXTURE,
                           shape_outline=ShapeOutline.FILL,
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
        ax.text(bbox[0], bbox[1] - 5, ann.shape.name, color='yellow', fontsize=8, backgroundcolor='black')

    ax.set_title("Sample from ShapeDataset")
    ax.axis('off')
    plt.show()