import src.models.gnn_models as gnn_pkg
import torch


class PopulationGCN(torch.nn.Module):
    def __init__(
        self,
        gnn_name: str,
        gnn_parameters: dict,
        hidden_size: int,
        dropout: float,
        num_classes: int,
    ):
        super().__init__()
        self.backbone = gnn_pkg.__dict__[gnn_name](**gnn_parameters)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        node_embedddings = self.backbone(x=x, edge_index=edge_index, edge_weight=edge_weight)
        return self.fc(self.dropout(node_embedddings))
