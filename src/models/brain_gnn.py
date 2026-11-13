from typing import Any, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class RaGConv(nn.Module):
    """ROI-aware Graph Convolutional Layer (Ra-GConv).
    As formulated in Li et al. (Medical Image Analysis, 2021, Eq. 1-3),
    each anatomical ROI learns an individual embedding matrix W_i that
    is decomposed as a linear combination of K latent community bases:
        vec(W_i) = Theta2 * relu(Theta1 * r_i) + b
    where r_i is the one-hot anatomical identity vector of ROI i.
    """

    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        out_channels: int,
        num_communities: int = 8,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_communities = num_communities
        self.theta1 = nn.Linear(num_nodes, num_communities, bias=False)
        self.theta2 = nn.Linear(num_communities, out_channels * in_channels, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.theta1.weight)
        nn.init.xavier_uniform_(self.theta2.weight)
        nn.init.zeros_(self.theta2.bias)

    def forward(
        self,
        x: torch.Tensor,
        pos_id: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        # Eq. 2 & 3: generate node-specific projection matrix W_i
        alpha = F.relu(self.theta1(pos_id))  # (N_batch, K)
        w_vec = self.theta2(alpha)  # (N_batch, out_channels * in_channels)
        weight_mats = w_vec.view(-1, self.out_channels, self.in_channels)
        # Node-specific linear transformation: x'_i = W_i * x_i
        x_prime = torch.einsum("noi,ni->no", weight_mats, x)
        # Eq. 1: neighbor aggregation with incoming normalized edge weights
        src, dst = edge_index[0], edge_index[1]
        msg = edge_weight.unsqueeze(-1) * x_prime[src]
        agg = torch.zeros_like(x_prime).index_add_(0, dst, msg)
        return F.relu(x_prime + agg)


class RPool(nn.Module):
    """ROI-Selection TopK Pooling Layer (R-pool).
    Follows Li et al. (Medical Image Analysis, 2021, Eq. 4) with
    per-graph score z-normalization and TopK node pruning.
    """

    def __init__(
        self,
        in_channels: int,
        ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.w = nn.Parameter(torch.randn(in_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.w, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        pos_id: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor],
    ]:
        # Eq. 4: project onto normalized projection vector
        w_unit = self.w / self.w.norm(p=2).clamp(min=1e-12)
        score = torch.matmul(x, w_unit)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        norm_score = torch.zeros_like(score)
        topk_mask = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        sig_sorted_list = []
        for g in range(num_graphs):
            g_idx = (batch == g).nonzero(as_tuple=True)[0]
            if len(g_idx) == 0:
                continue
            s_g = score[g_idx]
            mean_g = s_g.mean()
            std_g = s_g.std(unbiased=False).clamp(min=1e-6)
            s_g_norm = (s_g - mean_g) / std_g
            norm_score[g_idx] = s_g_norm
            k = max(1, int(round(len(g_idx) * self.ratio)))
            _, top_rel = torch.topk(s_g_norm, k=k)
            topk_mask[g_idx[top_rel]] = True
            # Track sorted sigmoid scores for TPK loss
            sig_s = torch.sigmoid(s_g_norm)
            sig_sorted, _ = torch.sort(sig_s, descending=True)
            sig_sorted_list.append(sig_sorted)
        perm = topk_mask.nonzero(as_tuple=True)[0]
        gate = torch.sigmoid(norm_score[perm]).unsqueeze(-1)
        x_pooled = x[perm] * gate
        pos_id_pooled = pos_id[perm]
        batch_pooled = batch[perm]
        # Slice induced subgraph edges
        src, dst = edge_index[0], edge_index[1]
        edge_mask = topk_mask[src] & topk_mask[dst]
        sub_src, sub_dst = src[edge_mask], dst[edge_mask]
        sub_ew = edge_weight[edge_mask]
        node_map = torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)
        node_map[perm] = torch.arange(len(perm), device=x.device)
        edge_index_pooled = torch.stack([node_map[sub_src], node_map[sub_dst]], dim=0)
        # Re-normalize incoming edge weights for each surviving node
        dst_pooled = edge_index_pooled[1]
        deg_sum = torch.zeros(len(perm), dtype=sub_ew.dtype, device=sub_ew.device)
        deg_sum.index_add_(0, dst_pooled, sub_ew)
        edge_weight_pooled = sub_ew / deg_sum[dst_pooled].clamp(min=1e-12)
        return (
            x_pooled,
            pos_id_pooled,
            edge_index_pooled,
            edge_weight_pooled,
            batch_pooled,
            norm_score,
            sig_sorted_list,
        )


class BrainGNN(nn.Module):
    """Brain Graph Neural Network (BrainGNN) architecture.
    Follows Li et al. (Medical Image Analysis, 2021) with 2 sequential
    blocks of [Ra-GConv -> R-pool -> Readout], followed by summary
    concatenation and an MLP classification head.
    """

    def __init__(
        self,
        num_nodes: int = 166,
        in_channels: int = 4,
        hidden_dim1: int = 32,
        hidden_dim2: int = 32,
        num_classes: int = 3,
        num_communities: int = 8,
        pooling_ratio: float = 0.5,
        mlp_hidden_dim: int = 8,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.num_classes = num_classes
        # Block 1
        self.conv1 = RaGConv(num_nodes, in_channels, hidden_dim1, num_communities)
        self.pool1 = RPool(hidden_dim1, pooling_ratio)
        # Block 2
        self.conv2 = RaGConv(num_nodes, hidden_dim1, hidden_dim2, num_communities)
        self.pool2 = RPool(hidden_dim2, pooling_ratio)
        # Readout concatenation: 2 * (hidden_dim1 + hidden_dim2) -> mlp_hidden_dim -> num_classes
        readout_dim = 2 * (hidden_dim1 + hidden_dim2)
        self.mlp = nn.Sequential(
            nn.Linear(readout_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, num_classes),
        )

    @staticmethod
    def _readout(x: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        """Eq. 5: Concatenated mean and max global pooling per graph."""
        z_list = []
        for g in range(num_graphs):
            idx = (batch == g).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                z_list.append(
                    torch.zeros(2 * x.shape[1], device=x.device, dtype=x.dtype)
                )
            else:
                xg = x[idx]
                z_list.append(torch.cat([xg.mean(dim=0), xg.max(dim=0)[0]], dim=-1))
        return torch.stack(z_list, dim=0)

    def forward(
        self,
        x: Union[torch.Tensor, Any],
        pos_id: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, Any]]]:
        # Unpack PyG Data/Batch object if passed directly: model(batch)
        if not isinstance(x, torch.Tensor):
            data = x
            x = data.x
            pos_id = getattr(data, "pos_id", None)
            if pos_id is None:
                pos_id = getattr(data, "R", None)
            edge_index = data.edge_index
            edge_weight = getattr(data, "edge_weight", None)
            batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        num_graphs = int(batch.max().item()) + 1
        # Fallback for pos_id if not pre-populated in Data
        if pos_id is None:
            pos_id = torch.eye(self.num_nodes, device=x.device).repeat(num_graphs, 1)
        # Block 1
        h1 = self.conv1(x, pos_id, edge_index, edge_weight)
        x_p1, pos_p1, ei_p1, ew_p1, b_p1, norm_s1, sig_sorted1 = self.pool1(
            h1, pos_id, edge_index, edge_weight, batch
        )
        z1 = self._readout(x_p1, b_p1, num_graphs)
        # Block 2
        h2 = self.conv2(x_p1, pos_p1, ei_p1, ew_p1)
        x_p2, _, _, _, b_p2, _, sig_sorted2 = self.pool2(h2, pos_p1, ei_p1, ew_p1, b_p1)
        z2 = self._readout(x_p2, b_p2, num_graphs)
        # Readout concatenation and classification MLP
        z = torch.cat([z1, z2], dim=-1)
        logits = self.mlp(z)
        if not return_aux:
            return logits
        aux = {
            "norm_s1": norm_s1,
            "sig_sorted1": sig_sorted1,
            "sig_sorted2": sig_sorted2,
            "batch": batch,
            "num_graphs": num_graphs,
            "pool1_w": self.pool1.w,
            "pool2_w": self.pool2.w,
        }
        return logits, aux
