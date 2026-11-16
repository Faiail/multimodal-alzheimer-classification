import src.models.gnn_models as gnn_backbone
import torch


class PopulationClassfier(torch.nn.Module):
    def __init__(self, gnn_name: str, gnn_params: dict):
        super().__init__()
        self.model = gnn_backbone.__dict__[gnn_name](**gnn_params)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(x=x, edge_index=edge_index, edge_weigt=edge_weight)
