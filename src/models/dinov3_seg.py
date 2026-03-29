import math
import torch
import torch.nn as nn
from .decoder import Decoder, ProjectionHead

class DINOv3Encoder(nn.Module):
    def __init__(self, repo_dir: str, weight_path: str, img_size: int = 128, hidden_dim: int = 1024):
        super().__init__()
        self.img_size = img_size
        self.hidden_dim = hidden_dim
        self.encoder = torch.hub.load(repo_dir, 'dinov3_vitl16', source='local', weights=weight_path)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder.get_intermediate_layers(x, n=1)[0]  # [B, L, D]
        B, L, D = feats.shape
        assert D == self.hidden_dim
        ph = max(1, self.img_size // 16); pw = max(1, self.img_size // 16); expected = ph * pw
        if L == expected:        tokens, (h,w) = feats, (ph,pw)
        elif L == expected + 1:  tokens, (h,w) = feats[:,1:], (ph,pw)
        else:
            def best_hw(n):
                r = int(math.sqrt(n))
                for dh in range(r+1):
                    for cand in (r-dh, r+dh):
                        if cand>0 and n%cand==0: return (cand, n//cand)
                return None
            hw, use_cls = best_hw(L), False
            if hw is None:
                hw, use_cls = best_hw(L-1), True
                if hw is None: raise ValueError(f"Unexpected token length {L}")
            tokens, (h,w) = (feats[:,1:], hw) if use_cls else (feats, hw)

        return tokens.permute(0,2,1).reshape(B, D, h, w)  # [B,1024,h,w]

class DINOv3SegModel(nn.Module):
    def __init__(self, repo_dir: str, weight_path: str, img_size: int = 128, in_ch=1024, mid_ch=256, proj_dim: int = 50):
        super().__init__()
        self.backbone = DINOv3Encoder(repo_dir, weight_path, img_size=img_size, hidden_dim=in_ch)
        self.decoder  = Decoder(in_ch=in_ch, mid_ch=mid_ch)
        self.proj_head = ProjectionHead(in_dim=in_ch, proj_dim=proj_dim)

    def forward(self, x):
        tokens = self.backbone(x)      # [B,D,h,w]
        logits, feat = self.decoder(tokens)  # [B,1,H,W]
        fb = self.proj_head(tokens)    # [B,proj_dim,h,w]

        return logits, fb, feat # fb->pc feat->dpt
