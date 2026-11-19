from torch.utils.data import Dataset
import torch
from src.data.utils import BatchKeys
from safetensors.torch import load_file
import pandas as pd


class BrainFeatureExtractorDataset(Dataset):
    def __init__(
        self,
        root: str,
        dataset: str,
        mri_col: int,
        pet_col: int,
        sort_by_node: bool = False,
        filter_label_value: str = None,
    ):
        super().__init__()
        self.root = root
        self.dataset = pd.read_csv(dataset)
        self.mri_col = mri_col
        self.pet_col = pet_col
        if sort_by_node:
            self.dataset = self.dataset.sort_values(by="node_id").reset_index()
        if filter_label_value is not None:
            self.dataset = self.dataset[self.dataset.label == filter_label_value]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        raw_item = self.dataset.iloc[index][BatchKeys.SAFETENSOR_PTH]
        data = load_file(f"{self.root}/{raw_item}")[BatchKeys.ROI]
        return {BatchKeys.MRI: data[:, self.mri_col], BatchKeys.PET: data[:, self.pet_col]}
