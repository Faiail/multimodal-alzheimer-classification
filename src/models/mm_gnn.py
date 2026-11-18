from torch_geometric.data import Data
from torch_geometric.nn.models.basic_gnn import BasicGNN
import torch
import src.models.gnn_models as gnn_pkg


class MMGNN(torch.nn.Module):
    def __init__(
        self,
        mri_branch_parameters: dict,
        pet_branch_parameters: dict,
    ):
        super().__init__()
        self.mri_gnn = self._init_gnn(**mri_branch_parameters)
        self.pet_gnn = self._init_gnn(**pet_branch_parameters)

    def _init_gnn(self, gnn_name: str, gnn_config: dict) -> BasicGNN:
        return gnn_pkg.__dict__[gnn_name](**gnn_config)

    def forward(
        self,
        mri_graph: Data,
        pet_graph: Data,
        mri_mask: torch.Tensor,
        pet_mask: torch.Tensor,
    ) -> torch.Tensor:
        mri_logits = self.mri_gnn(
            x=mri_graph.x,
            edge_index=mri_graph.edge_index,
            edge_weight=mri_graph.edge_weight,
        )
        pet_logits = self.pet_gnn(
            x=pet_graph.x,
            edge_index=pet_graph.edge_index,
            edge_weight=pet_graph.edge_weight,
        )
        return ((mri_logits * mri_mask.unsqueeze(-1)) + (pet_logits * pet_mask.unsqueeze(-1))) / (
            mri_mask.unsqueeze(-1) + pet_mask.unsqueeze(-1)
        )
