import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=None, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class YOLOTinyBackbone(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        # stem
        self.stem = nn.Sequential(
            ConvBNAct(in_ch, base, 3, 2),      # /2
            ConvBNAct(base, base, 3, 1),
        )
        # stages
        self.s2 = nn.Sequential(
            ConvBNAct(base, base*2, 3, 2),     # /4
            ConvBNAct(base*2, base*2, 3, 1),
        )
        self.s3 = nn.Sequential(
            ConvBNAct(base*2, base*4, 3, 2),   # /8
            ConvBNAct(base*4, base*4, 3, 1),
            ConvBNAct(base*4, base*4, 3, 1),
        )
        self.s4 = nn.Sequential(
            ConvBNAct(base*4, base*8, 3, 2),   # /16
            ConvBNAct(base*8, base*8, 3, 1),
            ConvBNAct(base*8, base*8, 3, 1),
        )
        self.s5 = nn.Sequential(
            ConvBNAct(base*8, base*16, 3, 2),  # /32
            ConvBNAct(base*16, base*16, 3, 1),
            ConvBNAct(base*16, base*16, 3, 1),
        )

    def forward(self, x):
        x = self.stem(x)     # /2
        x = self.s2(x)       # /4
        c3 = self.s3(x)      # /8
        c4 = self.s4(c3)     # /16
        c5 = self.s5(c4)     # /32
        return c3, c4, c5

class YOLOFPN(nn.Module):
    def __init__(self, c3_ch, c4_ch, c5_ch, fpn_ch=128):
        super().__init__()
        self.lat5 = ConvBNAct(c5_ch, fpn_ch, k=1, s=1, p=0)
        self.lat4 = ConvBNAct(c4_ch, fpn_ch, k=1, s=1, p=0)
        self.lat3 = ConvBNAct(c3_ch, fpn_ch, k=1, s=1, p=0)

        self.out4 = nn.Sequential(
            ConvBNAct(fpn_ch*2, fpn_ch, 3, 1),
            ConvBNAct(fpn_ch, fpn_ch, 3, 1),
        )
        self.out3 = nn.Sequential(
            ConvBNAct(fpn_ch*2, fpn_ch, 3, 1),
            ConvBNAct(fpn_ch, fpn_ch, 3, 1),
        )

        # /8 -> /4
        self.up_to4 = nn.Sequential(
            ConvBNAct(fpn_ch, fpn_ch, 3, 1),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBNAct(fpn_ch, fpn_ch, 3, 1),
        )

    def forward(self, c3, c4, c5):
        p5 = self.lat5(c5)  # /32
        p4 = self.lat4(c4)  # /16
        p3 = self.lat3(c3)  # /8

        p5_up = F.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        p4 = self.out4(torch.cat([p4, p5_up], dim=1))  # /16

        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode="nearest")
        p3 = self.out3(torch.cat([p3, p4_up], dim=1))  # /8

        f4 = self.up_to4(p3)  # /4
        return f4

class YOLOSegModel(nn.Module):
    def __init__(self, img_size=128, base=32, fpn_ch=128):
        super().__init__()
        self.img_size = int(img_size)
        self.backbone = YOLOTinyBackbone(in_ch=3, base=base)

        # c3=/8: base*4, c4=/16: base*8, c5=/32: base*16
        self.fpn = YOLOFPN(c3_ch=base*4, c4_ch=base*8, c5_ch=base*16, fpn_ch=fpn_ch)

        # seg head：/4 -> /1
        self.seg_head = nn.Sequential(
            ConvBNAct(fpn_ch, fpn_ch, 3, 1),
            ConvBNAct(fpn_ch, fpn_ch, 3, 1),
            nn.Upsample(scale_factor=4, mode="nearest"),   # /4 -> /1
            nn.Conv2d(fpn_ch, 1, kernel_size=1),
        )

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        f4 = self.fpn(c3, c4, c5)            # /4
        logits = self.seg_head(f4)           # /1
        
        return logits, None, None

