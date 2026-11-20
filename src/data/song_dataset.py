from .visual_dataset import VisualDataset
from .utils import BatchKeys
from safetensors.torch import load_file
import torch


class LassoDataset(VisualDataset):
    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        file_pth = f"{self.multimodal_root}/{row[BatchKeys.SAFETENSOR_PTH]}"
        label = row[BatchKeys.LABEL]
        data = load_file(file_pth)
        mri = data[BatchKeys.SMRI]
        pet = data.get(BatchKeys.PET, mri * 0.0)
        features = torch.cat([mri, pet])
        return {BatchKeys.X: features, BatchKeys.GT: label}
