from torch_geometric.data import Data
from .utils import BatchKeys
import pandas as pd
import torch
from safetensors.torch import load_file
from .utils import BatchKeys


class MMGNNDataset:
    def __init__(
        self,
        graph_root: str,
        split_root: str,
        dataset: str,
        fold_id: int,
    ):
        self.graph_root = graph_root
        self.split_root = split_root
        self.dataset = pd.read_csv(dataset).sort_values("node_id").reset_index()
        self.fold = fold_id

    def __call__(self) -> dict[str, torch.Tensor]:
        graph_content = load_file(
            f"{self.graph_root}/{self.fold}/{BatchKeys.GRAPH}.safetensors"
        )
        mri_graph = Data(
            x=graph_content[BatchKeys.MRI],
            edge_index=graph_content[BatchKeys.EDGE_INDEX],
            edge_weight=graph_content[BatchKeys.EDGE_WEIGHT],
        )
        pet_graph = Data(
            x=graph_content[BatchKeys.PET],
            edge_index=graph_content[BatchKeys.EDGE_INDEX],
            edge_weight=graph_content[BatchKeys.EDGE_WEIGHT],
        )
        y = torch.from_numpy(self.dataset[BatchKeys.LABEL].values)

        mri_mask = torch.from_numpy(self.dataset[BatchKeys.HAS_MRI].values).float()
        pet_mask = torch.from_numpy(self.dataset[BatchKeys.HAS_PET].values).float()

        return {
            BatchKeys.MRI: mri_graph,
            BatchKeys.PET: pet_graph,
            BatchKeys.GT: y,
            BatchKeys.HAS_MRI: mri_mask,
            BatchKeys.HAS_PET: pet_mask,
        }

    def _get_split_dataset(self, split: str) -> pd.DataFrame:
        if split == "test":
            return pd.read_csv(f"{self.split_root}/{split}.csv")
        return pd.read_csv(f"{self.split_root}/folds/{self.fold}/{split}.csv")

    def get_mask(self, split: str) -> torch.Tensor:
        split_df = self._get_split_dataset(split)
        return torch.from_numpy(
            self.dataset[BatchKeys.SAFETENSOR_PTH]
            .isin(split_df[BatchKeys.SAFETENSOR_PTH])
            .values
        )
