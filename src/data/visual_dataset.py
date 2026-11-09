from torch.utils.data import Dataset
import pandas as pd
from safetensors.torch import load_file
from .utils import BatchKeys
import torch


class VisualDataset(Dataset):
    def __init__(
        self,
        multimodal_root: str,
        fold_root: str,
        split: str = "train",
        fold_id: int = None,
    ):
        super().__init__()
        self.multimodal_root = multimodal_root
        fold = f"fold/{fold_id}" if fold_id is not None else ""
        self.dataset = pd.read_csv(f"{fold_root}/{fold}/{split}.csv")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        file_pth = f"{self.multimodal_root}/{row[BatchKeys.SAFETENSOR_PTH]}"
        label = row[BatchKeys.LABEL]
        data = load_file(file_pth)
        x = torch.cat([data[BatchKeys.MRI], data[BatchKeys.PET]], dim=0)
        return {BatchKeys.IMAGE: x, BatchKeys.GT: label}

