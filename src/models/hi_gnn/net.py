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
        chunk_size: int = 1_000_000,
    ) -> torch.Tensor:
        src, dst = pop_edge_index[0], pop_edge_index[1]
        num_edges = src.shape[0]

        with torch.no_grad():
            # 1. Normalize at node level: O(N * d) instead of O(E * d).
            # For ~5000 nodes this takes ~2.5 MB instead of allocating 9+ GB per tensor!
            z = torch.nn.functional.normalize(brain_embeddings, p=2, dim=-1)

            # 2. Compute edge similarity in chunks to keep VRAM usage < 1 GB
            sim = torch.empty(
                num_edges, dtype=torch.float32, device=brain_embeddings.device
            )
            for i in range(0, num_edges, chunk_size):
                end = min(i + chunk_size, num_edges)
                s_chunk = src[i:end]
                d_chunk = dst[i:end]
                sim[i:end] = torch.nn.functional.relu(
                    (z[s_chunk] * z[d_chunk]).sum(dim=-1)
                )

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
