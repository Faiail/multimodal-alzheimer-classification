import torch
from torch import nn
from typing import Any
from .focal_loss import FocalLoss


class BrainGNNLoss(nn.Module):
    """Full 4-term regularization loss function for BrainGNN (Li et al., Eq. 10)."""

    def __init__(
        self,
        alpha: float = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        lambda1: float = 0.1,
        lambda2: float = 0.1,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.eps = eps
        self.ce = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)

    def forward(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        aux: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        l_ce = self.ce(logits, y)
        # Eq. 7: Unit loss
        l_unit = (aux["pool1_w"].norm(p=2) - 1.0) ** 2 + (
            aux["pool2_w"].norm(p=2) - 1.0
        ) ** 2
        # Eq. 9: TopK pooling loss across both pooling layers
        l_tpk = 0.0
        for sig_sorted_list in [aux["sig_sorted1"], aux["sig_sorted2"]]:
            layer_tpk = 0.0
            num_m = len(sig_sorted_list)
            for s_m in sig_sorted_list:
                n_nodes = s_m.shape[0]
                k = max(1, int(round(n_nodes * 0.5)))
                s_top = s_m[:k].clamp(min=self.eps, max=1.0 - self.eps)
                s_bottom = s_m[k:].clamp(min=self.eps, max=1.0 - self.eps)
                term = torch.log(s_top).sum() + torch.log(1.0 - s_bottom).sum()
                layer_tpk = layer_tpk - (term / n_nodes)
            l_tpk = l_tpk + (layer_tpk / max(1, num_m))
        # Eq. 8: Group-Level Consistency loss on 1st pooling layer
        l_glc = 0.0
        num_graphs = aux["num_graphs"]
        batch = aux["batch"]
        norm_s1 = aux["norm_s1"]
        s_list = []
        for g in range(num_graphs):
            idx = (batch == g).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                s_list.append(norm_s1[idx])
        if len(s_list) > 0:
            s_mat = torch.stack(s_list, dim=0)
            num_classes = logits.shape[1]
            for c in range(num_classes):
                c_mask = y == c
                m_c = int(c_mask.sum().item())
                if m_c > 1:
                    s_c = s_mat[c_mask]
                    mean_sc = s_c.mean(dim=0, keepdim=True)
                    l_glc = l_glc + 2.0 * m_c * torch.sum((s_c - mean_sc) ** 2)
            l_glc = l_glc / max(1, num_graphs)
        total = l_ce + l_unit + self.lambda1 * l_tpk + self.lambda2 * l_glc
        return {
            "loss": total,
            "loss_ce": l_ce,
            "loss_unit": l_unit,
            "loss_tpk": l_tpk,
            "loss_glc": l_glc,
        }
