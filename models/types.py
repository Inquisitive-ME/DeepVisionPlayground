from enum import Enum, auto


class ModelType(Enum):
    center_localization = auto()
    center_localization_and_class_id = auto()
