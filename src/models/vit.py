from monai.networks.nets import ViT as MonaiViT
import torch


class ViT(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        spatial_dims: int,
        img_size: int | tuple[int],
        patch_size: int | tuple[int],
        num_classes: int,
    ) -> None:
        self.model = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            spatial_dims=spatial_dims,
            classification=True,
            num_classes=num_classes,
            post_activation=None,
        )

    def forward(self, x):
        return self.model(x)[0]
