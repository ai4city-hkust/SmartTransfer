import torch
import torch.nn as nn


class BinaryFocalLoss2d(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        super().__init__(); self.alpha, self.gamma, self.reduction = alpha, gamma, reduction
    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - pt) ** self.gamma * bce
        
        return loss.mean() if self.reduction=="mean" else loss.sum()
