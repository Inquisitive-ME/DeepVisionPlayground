import json
import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import NamedTuple, Tuple

from typing_extensions import Self


class BoundingBox(NamedTuple):
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class ShapeType(Enum):
    RECTANGLE = 0
    CIRCLE = 1
    TRIANGLE = 2


class ShapeOutline(Enum):
    THIN = 2
    THICK = 6
    FILL = auto()
    RANDOM = auto()


class AnnotationEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            # For enums, save both name and value for better readability
            return {"__enum__": True, "name": obj.name, "value": obj.value}
        elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
            # Check for namedtuple by looking for _fields attribute
            return {"__namedtuple__": True, "type": type(obj).__name__,
                    "values": {field: getattr(obj, field)
                               for field in obj._fields}}
        elif isinstance(obj, tuple):
            # Regular tuples
            return list(obj)
        elif hasattr(obj, "__dict__"):
            # Regular classes
            return obj.__dict__
        return super().default(obj)


@dataclass
class Annotation:
    shape: ShapeType
    bbox: BoundingBox
    center: Tuple[float, float]      # (x, y)
    color: Tuple[int, int, int]      # (R, G, B)

    @classmethod
    def from_dict(cls, data_dict):
        """
        Create an Annotation object from a dictionary

        Args:
            data_dict: Dictionary containing annotation data

        Returns:
            Annotation: The deserialized Annotation object
        """
        # Handle ShapeType enum from the special format
        shape_data = data_dict.get('shape', {})
        shape = None
        if isinstance(shape_data, dict) and shape_data.get('__enum__') is True:
            enum_name = shape_data.get('name')
            if isinstance(enum_name, str) and enum_name in ShapeType.__members__:
                shape = ShapeType[enum_name]

        # Handle BoundingBox, which appears to be a list
        # [x_min, y_min, x_max, y_max]
        bbox_data = data_dict.get('bbox', [])
        if isinstance(bbox_data, list) and len(bbox_data) == 4:
            bbox = BoundingBox(
                x_min=bbox_data[0],
                y_min=bbox_data[1],
                x_max=bbox_data[2],
                y_max=bbox_data[3]
            )
        else:
            bbox = None

        # Handle center and color
        center = tuple(data_dict.get('center', (0, 0)))
        color = tuple(data_dict.get('color', (0, 0, 0)))

        assert shape is not None
        assert bbox is not None
        return cls(shape=shape, bbox=bbox, center=center, color=color)


class BackgroundType(Enum):
    SOLID = auto()
    TEXTURE = auto()
    RANDOM = auto()

    @classmethod
    def get_random_background(cls) -> Self:
        # Return a random background type from all except RANDOM.
        valid_backgrounds = [member for member in cls if member != cls.RANDOM]
        return random.choice(valid_backgrounds)
