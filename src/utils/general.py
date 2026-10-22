import json
import os
import random
from pathlib import Path
from typing import Union

import numpy as np
import torch
from ruamel.yaml import YAML

DEFAULT_DATA_ROOT = "/media/BarraCuda/dcirilli2"


def resolve_data_path(path: Union[str, Path, None]) -> Union[str, None]:
    """Resolve data paths, automatically normalizing `/media/barractuda/dcirilli2` to `/media/BarraCuda/dcirilli2`."""
    if path is None:
        return None
    path_str = str(path)
    if "/media/barractuda/dcirilli2" in path_str and os.path.exists(DEFAULT_DATA_ROOT):
        path_str = path_str.replace("/media/barractuda/dcirilli2", DEFAULT_DATA_ROOT)
    return path_str


def load_ruamel(path: Union[str, Path], typ: str = "safe") -> dict:
    yaml = YAML(typ=typ)
    return yaml.load(Path(path))


def save_ruamel(data: dict, path: Union[str, Path]) -> None:
    yaml = YAML()
    yaml.default_flow_style = False
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(path_obj, "w+", encoding="utf-8") as f:
        yaml.dump(data, f)


def load_json(path: Union[str, Path]) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Union[str, Path], content: dict, mode: str = "w+", indent: int = 2) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(path_obj, mode, encoding="utf-8") as f:
        json.dump(content, f, indent=indent)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

