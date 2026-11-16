from .backbone import GNNBackbone
from .pop_classifier import PopulationClassfier
import torch


class HiGNN(torch.nn.Module):
    def __init__(self, backbone_params: dict, classifier_params: dict):
        super().__init__()
        self.backbone = GNNBackbone(**backbone_params)
        self.classifier = PopulationClassfier(**classifier_params)

    def forward(
        self,
        brain_embeddings: torch.Tensor,
        pop_edge_index: torch.Tensor,
        pop_edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = pop_edge_index[0], pop_edge_index[1]
        z_src = torch.nn.functional.normalize(brain_embeddings[src], p=2, dim=-1)
        z_dst = torch.nn.functional.normalize(brain_embeddings[dst], p=2, dim=-1)
        sim = torch.nn.functional.relu((z_src * z_dst).sum(dim=-1))
        pop_refined_weight = pop_edge_weight * sim
        return self.classifier(
            x=brain_embeddings,
            edge_index=pop_edge_index,
            edge_weight=pop_refined_weight,
        )

    def encode_brains(
        self,
        brain_x: torch.Tensor,
        brain_edge_index: torch.Tensor,
        brain_edge_weight: torch.Tensor,
        brain_batch: torch.Tensor,
    ) -> torch.Tensor:
        return self.backbone(
            x=brain_x,
            edge_index=brain_edge_index,
            edge_weight=brain_edge_weight,
            batch=brain_batch,
        )
