import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing, global_max_pool, global_mean_pool, to_hetero
from torch_geometric.utils import softmax, to_dense_batch
from .layers import (GraphConvLayer, RobustGraphAttentionLayer,
                     KANGraphConvLayer, KANRobustGraphAttentionLayer,
                     KANLinear)


class PyGGraphConv(MessagePassing):
    """Weighted message passing used by each relation of the heterogeneous GCN."""

    def __init__(self, in_features, out_features, use_kan=False):
        super().__init__(aggr='add')
        self.transform = KANLinear(in_features, out_features) if use_kan else nn.Linear(
            in_features, out_features, bias=False
        )
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x, edge_index, edge_weight):
        if isinstance(x, tuple):
            source_x, target_x = x
            support = (self.transform(source_x), None)
            size = (source_x.size(0), target_x.size(0))
        else:
            support = self.transform(x)
            size = None
        out = self.propagate(
            edge_index, x=support, edge_weight=edge_weight, size=size
        )
        return self.bn(out)

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.view(-1, 1)


class PyGGraphAttentionConv(MessagePassing):
    """Sparse relation-aware equivalent of the project's dense GAT layer."""

    def __init__(self, in_features, out_features, dropout, n_heads=1, use_kan=False):
        super().__init__(aggr='add', node_dim=0)
        projection = KANLinear if use_kan else lambda i, o: nn.Linear(i, o, bias=False)
        self.heads = nn.ModuleList([projection(in_features, out_features) for _ in range(n_heads)])
        self.attentions = nn.ParameterList([
            nn.Parameter(torch.empty(2 * out_features)) for _ in range(n_heads)
        ])
        for attention in self.attentions:
            nn.init.xavier_uniform_(attention.view(1, -1))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.norm = nn.LayerNorm(out_features * n_heads)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        del edge_weight  # GAT uses relation membership; weights remain available to GCN.
        outputs = []
        for projection, attention in zip(self.heads, self.attentions):
            if isinstance(x, tuple):
                source_x, target_x = x
                projected = (projection(source_x), projection(target_x))
                size = (source_x.size(0), target_x.size(0))
            else:
                projected = projection(x)
                size = None
            outputs.append(
                self.propagate(
                    edge_index, x=projected, attention=attention, size=size
                )
            )
        return self.norm(torch.cat(outputs, dim=-1))

    def message(self, x_i, x_j, attention, index, ptr, size_i):
        scores = self.leaky_relu((torch.cat([x_i, x_j], dim=-1) * attention).sum(dim=-1))
        coefficients = softmax(scores, index, ptr, size_i)
        coefficients = F.dropout(coefficients, self.dropout, training=self.training)
        return x_j * coefficients.unsqueeze(-1)


class HeterogeneousInputEncoder(nn.Module):
    """Project each node type from its own feature width to a shared hidden width."""

    def __init__(self, input_dims, hidden_dim):
        super().__init__()
        self.projections = nn.ModuleDict({
            node_type: nn.Linear(input_dim, hidden_dim)
            for node_type, input_dim in input_dims.items()
        })
        self.normalizations = nn.ModuleDict({
            node_type: nn.BatchNorm1d(hidden_dim) for node_type in input_dims
        })

    def forward(self, x_dict):
        return {
            node_type: F.relu(self.normalizations[node_type](self.projections[node_type](x)))
            for node_type, x in x_dict.items()
        }


class HeterogeneousGCNBackbone(nn.Module):
    def __init__(self, hidden_dim, n_layers, dropout, use_kan):
        super().__init__()
        self.layers = nn.ModuleList([
            PyGGraphConv(hidden_dim, hidden_dim, use_kan=use_kan)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight):
        for layer in self.layers:
            identity = x
            x = F.relu(layer(self.dropout(x), edge_index, edge_weight) + identity)
        return x


class HeterogeneousGATBackbone(nn.Module):
    def __init__(self, hidden_dim, n_layers, dropout, use_kan):
        super().__init__()
        self.layers = nn.ModuleList()
        if n_layers == 1:
            self.layers.append(PyGGraphAttentionConv(hidden_dim, hidden_dim, dropout, 1, use_kan))
        else:
            self.layers.append(PyGGraphAttentionConv(hidden_dim, hidden_dim, dropout, 2, use_kan))
            for _ in range(n_layers - 2):
                self.layers.append(PyGGraphAttentionConv(hidden_dim * 2, hidden_dim, dropout, 2, use_kan))
            self.layers.append(PyGGraphAttentionConv(hidden_dim * 2, hidden_dim, dropout, 1, use_kan))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight):
        for index, layer in enumerate(self.layers):
            x = layer(self.dropout(x), edge_index, edge_weight)
            if index < len(self.layers) - 1:
                x = F.elu(x)
        return x


def _heterogeneous_embedding(backbone, input_encoder, graph):
    encoded_x = input_encoder(graph.x_dict)
    x_dict = backbone(encoded_x, graph.edge_index_dict, graph.edge_weight_dict)
    node_x = x_dict['brain_region']
    batch = graph['brain_region'].batch
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=r"The usage of `scatter\(reduce='max'\)` can be accelerated.*",
            category=UserWarning,
            module=r'torch_geometric\.utils\._scatter',
        )
        max_pooled = global_max_pool(node_x, batch)
    return torch.cat([global_mean_pool(node_x, batch), max_pooled], dim=-1)


class TabularMLP(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features * 2),
            nn.BatchNorm1d(out_features * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(out_features * 2, out_features),
            nn.BatchNorm1d(out_features), nn.ReLU()
        )
    def forward(self, x): return self.net(x)

class StandardHead(nn.Module):
    """Testa di classificazione classica (MLP) per confronto fair con KAN"""
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
    def forward(self, x): return self.net(x)

class GCN_KAN_Classifier(nn.Module):
    def __init__(self, in_features, n_clinical_features, hidden_dim, n_classes=2, dropout=0.4,
                 use_kan_backbone=False, use_kan_head=True, n_layers=2, graph_type='static',
                 hetero_metadata=None, hetero_input_dims=None):
        super().__init__()
        assert n_layers >= 1, "n_layers must be >= 1"
        self.use_clinical = n_clinical_features > 0
        self.graph_type = graph_type
        self.is_heterogeneous = hetero_metadata is not None

        if self.is_heterogeneous:
            if hetero_input_dims is None:
                if len(hetero_metadata[0]) != 1:
                    raise ValueError("hetero_input_dims is required for multiple node types")
                hetero_input_dims = {hetero_metadata[0][0]: in_features}
            self.hetero_input_encoder = HeterogeneousInputEncoder(
                hetero_input_dims, hidden_dim
            )
            homogeneous = HeterogeneousGCNBackbone(
                hidden_dim, n_layers, dropout, use_kan_backbone
            )
            self.hetero_backbone = to_hetero(homogeneous, hetero_metadata, aggr='sum')
        else:
            self.input_proj = nn.Linear(in_features, hidden_dim)
            self.bn_in = nn.BatchNorm1d(hidden_dim)
            ConvLayer = KANGraphConvLayer if use_kan_backbone else GraphConvLayer
            self.gcn_layers = nn.ModuleList([
                ConvLayer(hidden_dim, hidden_dim, use_residual=True) for _ in range(n_layers)
            ])

        if self.use_clinical:
            self.tabular_encoder = TabularMLP(n_clinical_features, hidden_dim, dropout)
            self.fusion_dim = hidden_dim * 3
        else:
            self.fusion_dim = hidden_dim * 2
        self.dropout = nn.Dropout(dropout)
        self.classifier_layer = (
            KANLinear(self.fusion_dim, hidden_dim)
            if use_kan_head else StandardHead(self.fusion_dim, hidden_dim)
        )
        self.final_proj = nn.Linear(hidden_dim, n_classes)

    def forward(self, x_graph, adj, x_clin=None):
        if self.is_heterogeneous:
            if not isinstance(x_graph, HeteroData):
                raise TypeError("heterogeneous mode expects a batched HeteroData object")
            emb_graph = _heterogeneous_embedding(self.hetero_backbone, self.hetero_input_encoder, x_graph)
        else:
            x = F.relu(self.bn_in(self.input_proj(x_graph).transpose(1, 2)).transpose(1, 2))
            for index, layer in enumerate(self.gcn_layers):
                if index > 0:
                    x = self.dropout(x)
                x = F.relu(layer(x, adj))
            emb_graph = torch.cat([torch.mean(x, dim=1), torch.max(x, dim=1)[0]], dim=1)

        emb_fused = torch.cat([emb_graph, self.tabular_encoder(x_clin)], dim=1) \
            if self.use_clinical and x_clin is not None else emb_graph
        return self.final_proj(self.dropout(self.classifier_layer(emb_fused)))


class GAT_KAN_Classifier(nn.Module):
    def __init__(self, in_features, n_clinical_features, hidden_dim, n_classes=2, dropout=0.5,
                 use_kan_backbone=False, use_kan_head=True, n_layers=2, graph_type='static',
                 hetero_metadata=None, hetero_input_dims=None):
        super().__init__()
        assert n_layers >= 1, "n_layers must be >= 1"
        self.n_heads = 2
        self.use_clinical = n_clinical_features > 0
        self.graph_type = graph_type
        self.is_heterogeneous = hetero_metadata is not None

        if self.is_heterogeneous:
            if hetero_input_dims is None:
                if len(hetero_metadata[0]) != 1:
                    raise ValueError("hetero_input_dims is required for multiple node types")
                hetero_input_dims = {hetero_metadata[0][0]: in_features}
            self.hetero_input_encoder = HeterogeneousInputEncoder(
                hetero_input_dims, hidden_dim
            )
            homogeneous = HeterogeneousGATBackbone(
                hidden_dim, n_layers, dropout, use_kan_backbone
            )
            self.hetero_backbone = to_hetero(homogeneous, hetero_metadata, aggr='sum')
        else:
            self.input_proj = nn.Linear(in_features, hidden_dim)
            GATLayer = KANRobustGraphAttentionLayer if use_kan_backbone else RobustGraphAttentionLayer
            self.gat_layers = nn.ModuleList()
            if n_layers == 1:
                self.gat_layers.append(GATLayer(hidden_dim, hidden_dim, dropout, alpha=0.2, n_heads=1))
            else:
                self.gat_layers.append(GATLayer(hidden_dim, hidden_dim, dropout, alpha=0.2, n_heads=2))
                for _ in range(n_layers - 2):
                    self.gat_layers.append(GATLayer(hidden_dim * 2, hidden_dim, dropout, alpha=0.2, n_heads=2))
                self.gat_layers.append(GATLayer(hidden_dim * 2, hidden_dim, dropout, alpha=0.2, n_heads=1))

        if self.use_clinical:
            self.tabular_encoder = TabularMLP(n_clinical_features, hidden_dim, dropout)
            self.fusion_dim = hidden_dim * 3
        else:
            self.fusion_dim = hidden_dim * 2
        self.dropout = nn.Dropout(dropout)
        self.classifier_layer = (
            KANLinear(self.fusion_dim, hidden_dim)
            if use_kan_head else StandardHead(self.fusion_dim, hidden_dim)
        )
        self.final_proj = nn.Linear(hidden_dim, n_classes)

    def forward(self, x_graph, adj, x_clin=None):
        if self.is_heterogeneous:
            if not isinstance(x_graph, HeteroData):
                raise TypeError("heterogeneous mode expects a batched HeteroData object")
            emb_graph = _heterogeneous_embedding(self.hetero_backbone, self.hetero_input_encoder, x_graph)
        else:
            x = F.relu(self.input_proj(x_graph))
            for index, layer in enumerate(self.gat_layers):
                if index > 0:
                    x = self.dropout(x)
                x = layer(x, adj)
                if index < len(self.gat_layers) - 1:
                    x = F.elu(x)
            emb_graph = torch.cat([torch.mean(x, dim=1), torch.max(x, dim=1)[0]], dim=1)

        emb_fused = torch.cat([emb_graph, self.tabular_encoder(x_clin)], dim=1) \
            if self.use_clinical and x_clin is not None else emb_graph
        return self.final_proj(self.dropout(self.classifier_layer(emb_fused)))


class DualGraph_KAN_Classifier(nn.Module):
    def __init__(self, in_features, n_clinical_features, hidden_dim, n_classes=2, dropout=0.5,
                 use_kan_backbone=False, use_kan_head=True, n_layers=2,
                 graph_type='static', hetero_metadata=None, hetero_input_dims=None):
        super().__init__()
        self.n_heads = 2
        self.graph_type = graph_type
        self.is_heterogeneous = hetero_metadata is not None
        self.clin_proj = nn.Linear(1, hidden_dim)
        self.type_emb_brain = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.type_emb_clin = nn.Parameter(torch.randn(1, 1, hidden_dim))

        GATLayer = KANRobustGraphAttentionLayer if use_kan_backbone else RobustGraphAttentionLayer
        if self.is_heterogeneous:
            if hetero_input_dims is None:
                raise ValueError("hetero_input_dims is required for heterogeneous DualGraph")
            self.hetero_input_encoder = HeterogeneousInputEncoder(
                hetero_input_dims, hidden_dim
            )
            homogeneous = HeterogeneousGATBackbone(
                hidden_dim, n_layers, dropout, use_kan_backbone
            )
            self.hetero_backbone = to_hetero(
                homogeneous, hetero_metadata, aggr='sum'
            )
            clinical_heads = 1
            union_input_dim = hidden_dim
        else:
            self.brain_proj = nn.Linear(in_features, hidden_dim)
            self.brain_gat = GATLayer(
                hidden_dim, hidden_dim, dropout, n_heads=self.n_heads
            )
            clinical_heads = self.n_heads
            union_input_dim = hidden_dim * self.n_heads

        self.clin_gat = GATLayer(
            hidden_dim, hidden_dim, dropout, n_heads=clinical_heads
        )
        self.union_gat = GATLayer(
            union_input_dim, hidden_dim, dropout, n_heads=1
        )
        self.dropout = nn.Dropout(dropout)
        pool_dim = hidden_dim * 2
        self.classifier_layer = (
            KANLinear(pool_dim, hidden_dim)
            if use_kan_head else StandardHead(pool_dim, hidden_dim)
        )
        self.final_proj = nn.Linear(hidden_dim, n_classes)

    def forward(self, x_graph, adj, x_clin):
        if x_clin is None:
            raise ValueError("DualGraph requires clinical features")
        if self.is_heterogeneous:
            if not isinstance(x_graph, HeteroData):
                raise TypeError("heterogeneous DualGraph expects batched HeteroData")
            encoded = self.hetero_input_encoder(x_graph.x_dict)
            encoded['brain_region'] = encoded['brain_region'] + self.type_emb_brain.squeeze(0)
            output_dict = self.hetero_backbone(
                encoded, x_graph.edge_index_dict, x_graph.edge_weight_dict
            )
            h_brain, _ = to_dense_batch(
                output_dict['brain_region'], x_graph['brain_region'].batch
            )
        else:
            h_brain = F.relu(self.brain_proj(x_graph)) + self.type_emb_brain
            h_brain = self.brain_gat(h_brain, adj)

        h_clin = F.relu(self.clin_proj(x_clin.unsqueeze(-1))) + self.type_emb_clin
        h_clin = self.clin_gat(h_clin, adj=None)
        h_union = torch.cat([h_brain, h_clin], dim=1)
        h_final = self.union_gat(self.dropout(h_union), adj=None)
        embedding = torch.cat(
            [torch.mean(h_final, dim=1), torch.max(h_final, dim=1)[0]], dim=1
        )
        return self.final_proj(self.dropout(self.classifier_layer(embedding)))
