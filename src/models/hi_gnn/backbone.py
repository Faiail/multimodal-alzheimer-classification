import src.models.gnn_models as gnn_pkg
import torch
from torch_geometric.nn import global_mean_pool, global_max_pool


class GNNBackbone(torch.nn.Module):
    def __init__(
        self,
        gnn_name: str,
        gnn_params: dict,
        hidden_dim: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.model = gnn_pkg.__dict__[gnn_name](**gnn_params)
        self.proj = torch.nn.Linear(hidden_dim * 2, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        z = self.model(x=x, edge_index=edge_index, edge_weight=edge_weight)
        mean_pool = global_mean_pool(z, batch)
        max_pool = global_max_pool(z, batch)
        return self.proj(torch.cat([mean_pool, max_pool], dim=-1))
    
