import os
import torch.nn as nn
from src.models.lora import apply_lora_to_linear


def _norm_tid(tid) -> str:
    return str(tid).replace("\\", "/").split("/")[-1]

def dump_tile_list(save_path: str, tile_list, header: str = None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        if header:
            f.write(f"# {header}\n")
        for t in tile_list:
            f.write(_norm_tid(t) + "\n")
    print(f"[Tiles] Saved: {save_path} (n={len(tile_list)})")

def get_tile_ids_from_indices(dataset, indices):
    out = []
    for ii in indices:
        _, _, tid = dataset[int(ii)]
        out.append(_norm_tid(tid))
    return out

def set_trainable(model: nn.Module, mode: str):
    for p in model.parameters():
        p.requires_grad = False

    if hasattr(model, "decoder") and (model.decoder is not None):
        if mode == "fewshot":
            model.decoder.eval()
        else:
            for p in model.decoder.parameters():
                p.requires_grad = True
            model.decoder.train()

    enc = None
    if hasattr(model, "backbone") and hasattr(model.backbone, "encoder"):
        enc = model.backbone.encoder

    if mode == "zeroshot":
        if enc is not None:
            enc.eval()
        return

    if mode == "fewshot":
        if enc is None:
            raise ValueError("Model has no backbone.encoder for LoRA fewshot")

        if not getattr(model, "_lora_injected", False):
            apply_lora_to_linear(enc)
            model._lora_injected = True

        for n, p in enc.named_parameters():
            if n.endswith(".A") or n.endswith(".B") or (".A" in n) or (".B" in n):
                p.requires_grad = True

        enc.eval()
        return

    raise ValueError(f"Unknown mode: {mode}")
