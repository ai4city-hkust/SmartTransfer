import torch
import torch.nn as nn
from .decoder import ProjectionHead
import segmentation_models_pytorch as smp

class ResNetSegModel(nn.Module):
    def __init__(self, encoder_name: str = "resnet18", proj_dim: int = 50, feat_idx: int = -2):    # resnet 18 152
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None
        )
        self.feat_idx = feat_idx
        for p in self.unet.encoder.parameters():
            p.requires_grad_(True)
        in_dim = self.unet.encoder.out_channels[feat_idx]
        self.proj_head = ProjectionHead(in_dim=in_dim, proj_dim=proj_dim)

    def forward(self, x: torch.Tensor):
        logits = self.unet(x)

        return logits, None, None
    