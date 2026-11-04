from src.utils.strenum import StrEnum


class BatchKeys(StrEnum):
    X = "x"
    Y = "y"
    ADJ = "adj"
    EDGE_INDEX = "edge_index"
    EDGE_WEIGHT = "edge_weight"
    EDGE_TYPE = "edge_type"
    TABULAR = "tabular"
    BATCH = "batch"
    NUM_NODES = "num_nodes"
    SUBJECT_ID = "subject_id"

