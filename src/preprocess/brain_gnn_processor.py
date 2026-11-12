from safetensors.torch import load_file
from src.data.utils import BatchKeys
import os
from src.utils import ParameterKeys
import torch
from tqdm import tqdm
from torch_geometric.data import Data


class BrainGNNProcessor:
    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.init()

    def init(self):
        self.in_dir = self.parameters.get(ParameterKeys.IN_DIR)
        self.out_dir = self.parameters.get(ParameterKeys.OUT_DIR, "./")
        os.makedirs(self.out_dir, exist_ok=True)
        self.pbar = self.parameters.get(ParameterKeys.PBAR, False)
        self.topk = self.parameters.get(ParameterKeys.TOPK, 0.1)
        self.skip = self.parameters.get(ParameterKeys.SKIP, False)

    def get_pbar(self, data: list, desc: str):
        return (
            data
            if not self.pbar
            else tqdm(
                data,
                total=len(data),
                desc=desc,
            )
        )

    def get_edges(
        self, roi_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk = int(roi_features.size(0) * self.topk)
        N = roi_features.shape[0]  # 166
        # 1. Pairwise squared Euclidean distance between ROI feature vectors: (166, 166)
        dist_sq = torch.cdist(roi_features, roi_features, p=2).pow(2)
        # 2. Scale-free RBF kernel bandwidth (median off-diagonal distance)
        off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=roi_features.device)
        sigma_sq = dist_sq[off_diag_mask].median().clamp(min=1e-6)
        sim = torch.exp(-dist_sq / (2.0 * sigma_sq))
        # Remove self-loops (self-transformation is handled by W_i * h_i in Eq. 1)
        sim.fill_diagonal_(0.0)
        # 3. Sparsification: keep top-k strongest connections per node (guarantees degree >= top_k)
        vals, idx = torch.topk(sim, k=topk, dim=1)
        adj = torch.zeros_like(sim)
        adj.scatter_(1, idx, vals)
        # 4. Symmetrize to make the brain graph undirected: (166, 166)
        adj = torch.maximum(adj, adj.t())
        # 5. Extract COO edge_index (2, E) and raw positive edge_weight (E,)
        edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        raw_weights = adj[edge_index[0], edge_index[1]]
        # 6. Normalize incoming edge weights per target node i (Eq. 1 of paper)
        # edge_index is [source, target]: target is edge_index[1]
        dst = edge_index[1]
        deg_sum = torch.zeros(N, dtype=raw_weights.dtype, device=raw_weights.device)
        deg_sum.index_add_(0, dst, raw_weights)
        edge_weight = raw_weights / deg_sum[dst].clamp(min=1e-12)
        return edge_index, edge_weight

    def launch(self):
        bar = self.get_pbar(os.listdir(self.in_dir), desc="Extracting brai gnn graphs")
        for fname in bar:
            out_fname = fname.replace(".safetensors", ".pt")
            if self.skip and os.path.exists(f"{self.out_dir}/{out_fname}"):
                continue
            roi_features = load_file(f"{self.in_dir}/{fname}")[BatchKeys.ROI]
            edge_index, edge_weight = self.get_edges(roi_features=roi_features)
            torch.save(
                Data(
                    x=roi_features.float(),
                    edge_index=edge_index.long(),
                    edge_weight=edge_weight.float(),
                    pos_id=torch.eye(roi_features.size(0), dtype=torch.float32),
                ),
                f"{self.out_dir}/{out_fname}",
            )
