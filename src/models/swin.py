from typing import Sequence, Tuple, Union
import torch
import torch.nn as nn
from monai.networks.nets.swin_unetr import SwinTransformer


class Swin(nn.Module):
    """
    Hierarchical 3D Swin Vision Transformer classifier for multiclass
    neuroimaging diagnosis (e.g., CN vs. MCI vs. AD).

    Built on top of MONAI's SwinTransformer backbone with local shifted-window
    self-attention, progressive patch merging, 3D adaptive average pooling,
    and a linear classification head.

    Args:
        in_channels (int): Number of input channels (1 for sMRI, 2 for sMRI+PET early fusion).
        num_classes (int): Number of target diagnostic classes (default: 3).
        embed_dim (int): Number of linear projection channels at stage 0 (default: 24).
        patch_size (Sequence[int]): Spatial patch size for tokenization (default: (2, 2, 2)).
        window_size (Sequence[int]): Local 3D shifted-window size (default: (7, 7, 7)).
        depths (Sequence[int]): Number of Swin Transformer blocks per stage (default: (2, 2, 2, 2)).
        num_heads (Sequence[int]): Number of attention heads per stage (default: (3, 6, 12, 24)).
        dropout_rate (float): Dropout rate for embeddings and projection layers (default: 0.1).
        attn_drop_rate (float): Dropout rate for attention weights (default: 0.0).
        drop_path_rate (float): Stochastic depth rate (default: 0.1).
        use_checkpoint (bool): Use PyTorch gradient checkpointing to save VRAM (default: False).
    """

    def __init__(
        self,
        in_channels: int,
        spatial_dims: int,
        num_classes: int,
        embed_dim: int = 24,
        patch_size: Sequence[int] = (2, 2, 2),
        window_size: Sequence[int] = (7, 7, 7),
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        dropout_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depths = depths

        # 1. 3D Swin Backbone
        self.backbone = SwinTransformer(
            in_chans=in_channels,
            embed_dim=embed_dim,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            drop_rate=dropout_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            use_checkpoint=use_checkpoint,
        )

        # Bottleneck channel dimension after 4 downsampling stages:
        # embed_dim * (2 ** len(depths)) -> 24 * 16 = 384 (or 48 * 16 = 768)
        self.bottleneck_dim = int(embed_dim * (2 ** len(depths)))

        # 2. Classification Head
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.norm = nn.LayerNorm(self.bottleneck_dim)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        self.classifier = nn.Linear(self.bottleneck_dim, num_classes)

        # 3. Initialize Head Weights from Scratch
        self._init_head_weights()

    def _init_head_weights(self):
        """Initializes classification head layers with truncated normal distribution."""
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0.0)
        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.constant_(self.norm.bias, 0.0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts global pooled latent representation from the bottleneck stage.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W).

        Returns:
            torch.Tensor: Normalized feature vector of shape (B, bottleneck_dim).
        """
        # Backbone returns multi-scale features: [x0, x1, x2, x3, x4]
        features = self.backbone(x)
        bottleneck = features[-1]                   # Deepest stage: (B, C_bot, D_bot, H_bot, W_bot)
        pooled = self.pool(bottleneck).flatten(1)   # (B, C_bot)
        latent = self.norm(pooled)                  # (B, C_bot)
        return latent

    def forward(
        self,
        x: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass producing raw unbounded logits for CrossEntropyLoss.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W).
            return_features (bool): If True, also returns latent embeddings.

        Returns:
            torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
                - logits: Unbounded prediction scores of shape (B, num_classes).
                - latent (optional): Feature vector of shape (B, bottleneck_dim).
        """
        latent = self.extract_features(x)
        logits = self.classifier(self.dropout(latent))
        return logits