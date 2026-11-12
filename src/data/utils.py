from src.utils import StrEnum


class BatchKeys(StrEnum):
    IMAGE = "image"
    GT = "gt"
    SAFETENSOR_PTH = "safetensor_file"
    LABEL = "label"
    MRI = "mri"
    PET = "pet"
    ROI = "roi"
    X = "x"
    EDGE_INDEX = "edge_index"
    EDGE_WEIGHT = "edge_weight"