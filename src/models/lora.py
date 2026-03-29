import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scale = self.alpha / max(1, self.r)
        self.dropout = nn.Dropout(dropout) if (dropout and dropout > 0) else None

        # --- expose Linear-like attributes ---
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.bias = base.bias  # may be None

        # Freeze base params
        for p in self.base.parameters():
            p.requires_grad = False

        # Create LoRA params on same device/dtype as base weight
        dev = self.base.weight.device
        dt  = self.base.weight.dtype

        self.A = nn.Parameter(torch.empty(self.r, self.in_features, device=dev, dtype=dt))
        self.B = nn.Parameter(torch.empty(self.out_features, self.r, device=dev, dtype=dt))

        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        nn.init.zeros_(self.B)

    @property
    def weight(self):
        # allow code that reads linear.weight
        return self.base.weight

    def forward(self, x):
        y = self.base(x)
        if self.dropout is not None:
            x = self.dropout(x)
        # x: [..., in_features]
        lora = (x @ self.A.t()) @ self.B.t()  # [..., out_features]
        return y + self.scale * lora
        
def _get_parent_module(root: nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
        
    return parent, parts[-1]

def apply_lora_to_linear(
    root: nn.Module,
    target_substr=("qkv", "proj", "fc1", "fc2"), 
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    verbose: bool = True,
):
    replaced = 0
    wrapped_linear_ids = set()

    for name, m in list(root.named_modules()):
        if not isinstance(m, nn.Linear):
            continue
        if target_substr is not None and not any(s in name for s in target_substr):
            continue

        mid = id(m)
        if mid in wrapped_linear_ids:
            continue

        parent, child = _get_parent_module(root, name)
        cur = getattr(parent, child)

        if isinstance(cur, LoRALinear):
            wrapped_linear_ids.add(id(cur.base))
            continue

        setattr(parent, child, LoRALinear(m, r=r, alpha=alpha, dropout=dropout))
        wrapped_linear_ids.add(mid)
        replaced += 1

    return replaced
