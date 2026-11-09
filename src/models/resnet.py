import monai.networks.nets as monai_nets
import torch


class ResNet(torch.nn.Module):
    def __init__(
        self,
        n_layers: int = 18,
        spatial_dims: int = 3,
        in_channels: int = 1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.resnet = getattr(monai_nets, f"resnet{n_layers}")(
            spatial_dims=spatial_dims,
            n_input_channels=in_channels,
            num_classes=num_classes,
            pretrained=False,
        )

    def forward(self, x):
        return self.resnet(x)