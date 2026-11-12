from torch_geometric.data import Data
from .utils import BatchKeys
import pandas as pd
import torch


class PopulationGraphDataset:
    def __init__(
        self,
        dataset_root: str,
        topology_root: str,
        fold_id: int,
    ):
        self.dataset_root = dataset_root
        self.topology_root = topology_root
        self.fold_id = fold_id

    def __call__(self) -> Data:
        topology = torch.load(f"{self.topology_root}/topology.pt")
        edge_index = topology[BatchKeys.EDGE_INDEX]
        edge_weight = topology[BatchKeys.EDGE_WEIGHT]
        x = torch.load(f"{self.topology_root}/{self.fold_id}/features.pt", weights_only=False)
        x = torch.from_numpy(x)
        population_dataset = pd.read_csv(f"{self.topology_root}/population_dataset.csv")
        population_dataset = population_dataset.sort_values(
            by="node_id", ascending=True
        )
        y = population_dataset["label"].values
        y = torch.from_numpy(y)
        train_dataset = pd.read_csv(
            f"{self.dataset_root}/folds/{self.fold_id}/train.csv"
        )
        train_mask = (
            population_dataset["safetensor_file"]
            .isin(train_dataset["safetensor_file"])
            .values
        )
        train_mask = torch.from_numpy(train_mask).bool()
        val_dataset = pd.read_csv(f"{self.dataset_root}/folds/{self.fold_id}/val.csv")
        val_mask = (
            population_dataset["safetensor_file"]
            .isin(val_dataset["safetensor_file"])
            .values
        )
        val_mask = torch.from_numpy(val_mask).bool()
        test_dataset = pd.read_csv(f"{self.dataset_root}/test.csv")
        test_mask = (
            population_dataset["safetensor_file"]
            .isin(test_dataset["safetensor_file"])
            .values
        )
        test_mask = torch.from_numpy(test_mask).bool()

        return Data(
            x=x.float(),
            edge_index=edge_index.long(),
            edge_weight=edge_weight.float(),
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
