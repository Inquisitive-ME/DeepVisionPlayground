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
    RECTANGLE = auto()
    CIRCLE = auto()
    TRIANGLE = auto()


class ShapeOutline(Enum):
    THIN = 2
    THICK = 6
    FILL = auto()
    RANDOM = auto()


@dataclass
class Annotation:
    shape: ShapeType
    bbox: BoundingBox
    center: Tuple[float, float]      # (x, y)
    color: Tuple[int, int, int]      # (R, G, B)


class BackgroundType(Enum):
    SOLID = auto()
    TEXTURE = auto()
    RANDOM = auto()

    @classmethod
    def get_random_background(cls) -> Self:
        # Return a random background type from all except RANDOM.
        valid_backgrounds = [member for member in cls if member != cls.RANDOM]
        return random.choice(valid_backgrounds)
