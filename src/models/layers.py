import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KANLinear(nn.Module):
    """Layer KAN fondamentale (Linear + Spline)"""
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, enable_standalone_scale_spline=True, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).expand(in_features, -1).contiguous())
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            grid_points = self.grid.T[self.spline_order : -self.spline_order]
            noise = (torch.rand(grid_points.size(0), self.in_features, self.out_features) - 1 / 2) * self.scale_noise / self.grid_size
            self.spline_weight.data.copy_((self.scale_spline if not self.enable_standalone_scale_spline else 1.0) * self.curve2coeff(grid_points, noise))
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        x = x.unsqueeze(-1)
        bases = ((x >= self.grid[:, :-1]) & (x < self.grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (x - self.grid[:, : -(k + 1)]) / (self.grid[:, k:-1] - self.grid[:, : -(k + 1)]) * bases[:, :, :-1] + (self.grid[:, k + 1 :] - x) / (self.grid[:, k + 1 :] - self.grid[:, 1:(-k)]) * bases[:, :, 1:]
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        batch_size = x.size(0)
        A_list = []
        for i in range(batch_size):
            A_i = self.b_splines(x[i:i+1])
            A_list.append(A_i.squeeze(0))
        A = torch.stack(A_list, dim=0)
        
        ret = torch.linalg.lstsq(A.transpose(0, 1), y.transpose(0, 1)).solution.permute(2, 0, 1)
        return ret

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0)

    def forward(self, x: torch.Tensor):
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_basis = self.b_splines(x).view(x.size(0), -1)
        spline_output = F.linear(spline_basis, self.scaled_spline_weight.view(self.out_features, -1))
        return base_output + spline_output

class HierarchicalPooling(nn.Module):
    def forward(self, x):
        return torch.cat([torch.mean(x, dim=1), torch.max(x, dim=1)[0]], dim=1)

class GraphConvLayer(nn.Module):
    """Standard GCN Layer (Linear)"""
    def __init__(self, in_features, out_features, use_residual=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bn = nn.BatchNorm1d(out_features)
        self.use_residual = use_residual and (in_features == out_features)

    def forward(self, x, adj):
        identity = x
        support = torch.matmul(x, self.linear.weight.t()) 
        output = torch.matmul(adj, support) if adj.dim() == 2 else torch.bmm(adj, support)
        
        B, N, F_ = output.shape
        output = self.bn(output.view(-1, F_)).view(B, N, F_)
        if self.use_residual: output += identity
        return output

class RobustGraphAttentionLayer(nn.Module):
    """Standard GAT Layer (Linear Projections)"""
    def __init__(self, in_features, out_features, dropout, alpha=0.2, n_heads=2):
        super().__init__()
        self.n_heads = n_heads
        self.heads = nn.ModuleList([nn.Linear(in_features, out_features, bias=False) for _ in range(n_heads)])
        self.attentions = nn.ParameterList([nn.Parameter(torch.zeros(size=(2 * out_features, 1))) for _ in range(n_heads)])
        for i in range(n_heads):
            nn.init.xavier_uniform_(self.heads[i].weight)
            nn.init.xavier_uniform_(self.attentions[i])
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.norm = nn.LayerNorm(out_features * n_heads)
        self.dropout = dropout
        self.capture_attention = False
        self.last_attention = None

    def forward(self, h, adj=None):
        B, N, _ = h.size()
        head_outs = []
        captured_attention = []
        for i in range(self.n_heads):
            Wh = self.heads[i](h)
            a_input = torch.cat([Wh.repeat_interleave(N, dim=1), Wh.repeat(1, N, 1)], dim=2)
            e = self.leakyrelu(torch.matmul(a_input, self.attentions[i])).view(B, N, N)
            
            if adj is not None:
                zero_vec = -9e15 * torch.ones_like(e)
                attention = torch.where(adj > 0, e, zero_vec)
            else:
                attention = e
                
            attention = F.softmax(attention, dim=2)
            if self.capture_attention:
                captured_attention.append(attention.detach())
            attention = F.dropout(attention, self.dropout, training=self.training)
            head_outs.append(torch.matmul(attention, Wh))
        self.last_attention = captured_attention if self.capture_attention else None
        return self.norm(torch.cat(head_outs, dim=2))

class KANGraphConvLayer(nn.Module):
    """KAN GCN Layer"""
    def __init__(self, in_features, out_features, use_residual=True):
        super().__init__()
        self.kan_trans = KANLinear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.use_residual = use_residual and (in_features == out_features)

    def forward(self, x, adj):
        identity = x
        B, N, F_ = x.shape
        support = self.kan_trans(x.reshape(-1, F_)).reshape(B, N, -1)
        output = torch.matmul(adj, support) if adj.dim() == 2 else torch.bmm(adj, support)
        
        output = self.bn(output.reshape(-1, output.shape[-1])).reshape(B, N, -1)
        if self.use_residual: output += identity
        return output

class KANRobustGraphAttentionLayer(nn.Module):
    """KAN GAT Layer"""
    def __init__(self, in_features, out_features, dropout, alpha=0.2, n_heads=2):
        super().__init__()
        self.n_heads = n_heads
        self.kan_heads = nn.ModuleList([KANLinear(in_features, out_features) for _ in range(n_heads)])
        self.attentions = nn.ParameterList([nn.Parameter(torch.zeros(size=(2 * out_features, 1))) for _ in range(n_heads)])
        for i in range(n_heads): nn.init.xavier_uniform_(self.attentions[i])
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.norm = nn.LayerNorm(out_features * n_heads)
        self.dropout = dropout
        self.capture_attention = False
        self.last_attention = None

    def forward(self, h, adj=None):
        B, N, _ = h.size()
        head_outs = []
        captured_attention = []
        for i in range(self.n_heads):
            Wh = self.kan_heads[i](h.reshape(-1, h.shape[-1])).reshape(B, N, -1)
            
            a_input = torch.cat([Wh.repeat_interleave(N, dim=1), Wh.repeat(1, N, 1)], dim=2)
            e = self.leakyrelu(torch.matmul(a_input, self.attentions[i])).view(B, N, N)
            
            if adj is not None:
                zero_vec = -9e15 * torch.ones_like(e)
                attention = torch.where(adj > 0, e, zero_vec)
            else:
                attention = e
            
            attention = F.softmax(attention, dim=2)
            if self.capture_attention:
                captured_attention.append(attention.detach())
            attention = F.dropout(attention, self.dropout, training=self.training)
            head_outs.append(torch.matmul(attention, Wh))
        self.last_attention = captured_attention if self.capture_attention else None
        return self.norm(torch.cat(head_outs, dim=2))

class FocalLoss(nn.Module):
    """Focal Loss per gestire class imbalance e hard examples
    
    Args:
        alpha: Weights per classe (opzionale)
        gamma: Focusing parameter (default=2.0). Valori più alti penalizzano maggiormente gli easy examples
        reduction: 'mean', 'sum' o 'none'
    
    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
