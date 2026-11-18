import torch
import torch.nn as nn
from torch_geometric.nn.conv import ChebConv, MessagePassing
from torch_geometric.nn.models.basic_gnn import BasicGNN


class FastChebConv(nn.Module):
    """Memory-efficient Chebyshev Graph Convolution (exact match to ChebConv,

    lambda_max=2.0, normalization='sym'). Projects X @ W_k BEFORE graph
    propagation and uses torch.sparse.mm so no [E, in_channels] tensor is
    ever allocated.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K

        self.lins = nn.ModuleList(
            [nn.Linear(in_channels, out_channels, bias=False) for _ in range(K)]
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    @staticmethod
    def _build_scaled_laplacian(
        num_nodes: int,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """Builds sparse L_tilde = (2 / 2) * L_sym - I = -D^{-1/2} A D^{-1/2}."""
        row, col = edge_index
        if edge_weight is None:
            edge_weight = torch.ones(
                row.size(0), device=edge_index.device, dtype=torch.float32
            )

        deg = torch.zeros(num_nodes, device=edge_index.device, dtype=edge_weight.dtype)
        deg.scatter_add_(0, row, edge_weight)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0.0)

        norm_weight = -deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        return torch.sparse_coo_tensor(
            edge_index, norm_weight, size=(num_nodes, num_nodes)
        ).coalesce()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        N = x.size(0)
        L_tilde = self._build_scaled_laplacian(N, edge_index, edge_weight)

        # k = 0: T_0(L) (X W_0) = X W_0
        out = self.lins[0](x)

        # k = 1: T_1(L) (X W_1) = L_tilde @ (X W_1)
        if self.K > 1:
            h1 = self.lins[1](x)
            out = out + torch.sparse.mm(L_tilde, h1)

        # k >= 2: T_k(L) (X W_k) via recurrence on [N, out_channels]
        for k in range(2, self.K):
            hk = self.lins[k](x)
            t_prev2 = hk
            t_prev1 = torch.sparse.mm(L_tilde, hk)
            for _ in range(2, k + 1):
                t_curr = 2.0 * torch.sparse.mm(L_tilde, t_prev1) - t_prev2
                t_prev2, t_prev1 = t_prev1, t_curr
            out = out + t_curr

        if self.bias is not None:
            out = out + self.bias
        return out


class FastChebNet(BasicGNN):
    supports_edge_weight: bool = True
    supports_edge_attr: bool = False

    def init_conv(
        self, in_channels: int, out_channels: int, **kwargs
    ) -> MessagePassing:
        return FastChebConv(in_channels, out_channels, **kwargs)
