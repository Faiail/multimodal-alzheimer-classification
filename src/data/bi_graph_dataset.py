import torch
from torch.utils.data import Dataset
from .brain_graph_dataset import BrainGraphLoader
from .utils import BatchKeys
import pandas as pd
from safetensors.torch import load_file
from torch_geometric.data import Data


class BiGraphDataset(Dataset):
    def __init__(
        self,
        topology_root: str,
        roi_feature_root: str,
        dataset_root: str,
        fold_id: int,
    ) -> None:
        super().__init__()
        self.topology_root = topology_root
        self.roi_feature_root = roi_feature_root
        self.dataset_root = dataset_root
        self.fold_id = fold_id
        self.dataset = pd.read_csv(
            f"{self.topology_root}/population_dataset.csv", index_col="node_id"
        )

    def _build_brain_graph(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        N = x.shape[0]  # 166
        # Pairwise distance squared across 4D ROI features: (166, 166)
        dist_sq = torch.cdist(x, x, p=2).pow(2)
        # RBF Gaussian similarity using median off-diagonal distance
        off_diag = ~torch.eye(N, dtype=torch.bool, device=x.device)
        sigma_sq = dist_sq[off_diag].median().clamp(min=1e-6)
        sim = torch.exp(-dist_sq / (2.0 * sigma_sq))
        sim.fill_diagonal_(0.0)  # no self-loops
        # Top-k strongest connections per ROI + symmetrize
        vals, idx = torch.topk(sim, k=self.top_k, dim=1)
        adj = torch.zeros_like(sim).scatter_(1, idx, vals)
        adj = torch.maximum(adj, adj.t())
        # Extract COO edge_index and raw positive weights
        edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        raw_weights = adj[edge_index[0], edge_index[1]]
        # Row-normalize incoming edge weights: e_tilde_{ji} = e_{ji} / sum_{k} e_{ki}
        dst = edge_index[1]
        deg_sum = torch.zeros(N, dtype=raw_weights.dtype, device=raw_weights.device)
        deg_sum.index_add_(0, dst, raw_weights)
        edge_weight = raw_weights / deg_sum[dst].clamp(min=1e-12)
        return edge_index, edge_weight

    def __getitem__(self, index):
        raw_item = self.dataset.loc[index]
        roi_features = load_file(
            f"{self.roi_feature_root}/{raw_item[BatchKeys.SAFETENSOR_PTH]}"
        )[BatchKeys.ROI]
        edge_index, edge_weight = self._build_brain_graph(roi_features)
        y = raw_item[BatchKeys.LABEL]
        return {
            BatchKeys.GRAPH: Data(
                x=roi_features, edge_index=edge_index, edge_weight=edge_weight
            ),
            BatchKeys.GT: y,
        }

    def __len__(self):
        return len(self.dataset)

    def get_split_dataset(self, split: str) -> pd.DataFrame:
        if split == "test":
            return pd.read_csv(f"{self.dataset_root}/{split}.csv")
        return pd.read_csv(f"{self.dataset_root}/folds/{self.fold_id}/{split}.csv")

    def get_mask(self, split: str) -> torch.Tensor:
        split_df = self.get_split_dataset(split)
        return torch.from_numpy(
            self.dataset[BatchKeys.SAFETENSOR_PTH]
            .isin(split_df[BatchKeys.SAFETENSOR_PTH])
            .values
        )

    def get_base_topology(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.load(f"{self.topology_root}/topology.pt", weights_only=False)


class BiGraphLoader(BrainGraphLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor=None,
        persistent_workers=False,
        pin_memory_device="",
        follow_batch=None,
        exclude_keys=None,
    ):
        super().__init__(
            dataset,
            batch_size,
            False,
            sampler,
            batch_sampler,
            num_workers,
            pin_memory,
            drop_last,
            timeout,
            worker_init_fn,
            multiprocessing_context,
            generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )
