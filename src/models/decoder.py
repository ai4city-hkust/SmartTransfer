import torch
import torch.nn as nn


def conv_bn_relu(in_c, out_c, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(in_c, out_c, k, s, p, bias=False),
                         nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))

class Decoder(nn.Module):
    def __init__(self, in_ch=1024, mid_ch=256):
        super().__init__()
        self.block1 = conv_bn_relu(in_ch, mid_ch)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.block2 = conv_bn_relu(mid_ch, mid_ch)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.block3 = conv_bn_relu(mid_ch, mid_ch)
        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.block4 = conv_bn_relu(mid_ch, mid_ch)
        self.up4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.head = nn.Conv2d(mid_ch, 1, kernel_size=1)

    def forward(self, x):
        x = self.block1(x); x = self.up1(x)
        x = self.block2(x); x = self.up2(x)
        x = self.block3(x); x = self.up3(x)
        x = self.block4(x); x = self.up4(x)
        feat = x                    # [B,mid_ch,H,W]
        logits = self.head(x)       # [B,1,H,W]    

        return logits, feat

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 1024, proj_dim: int = 50, bias: bool = True):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, proj_dim, kernel_size=1, bias=bias)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.proj(x)
