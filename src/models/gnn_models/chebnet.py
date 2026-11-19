from torch_geometric.nn.conv import ChebConv, MessagePassing
from torch_geometric.nn.models.basic_gnn import BasicGNN


class ChebNet(BasicGNN):
    supports_edge_weight: bool = True
    supports_edge_attr: bool = False

    def init_conv(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs,
    ) -> MessagePassing:
        return ChebConv(in_channels, out_channels, **kwargs)
