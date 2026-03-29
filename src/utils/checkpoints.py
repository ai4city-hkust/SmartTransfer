import os
import torch
import torch.nn as nn
from src.models.dinov3_seg import DINOv3SegModel
from src.models.lora import LoRALinear
from src.utils.io import set_trainable


def load_decoder_only_into_model(model: nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and ("decoder" in ckpt):
        model.decoder.load_state_dict(ckpt["decoder"], strict=True)
        return

    if isinstance(ckpt, dict) and ("model_state" in ckpt):
        model.load_state_dict(ckpt["model_state"], strict=True)
        return

    if isinstance(ckpt, dict):
        model.load_state_dict(ckpt, strict=False)
        return

    raise ValueError(f"Unrecognized checkpoint format: {ckpt_path}")

def save_decoder_only(model, save_path, info: dict):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    if isinstance(model, DINOv3SegModel):
        to_save = {"decoder": model.decoder.state_dict(), **info}
    else:
        to_save = {"model_state": model.state_dict(), **info}

    torch.save(to_save, save_path)

def _collect_lora_state_dict(enc: nn.Module) -> dict:
    lora_sd = {}
    for name, m in enc.named_modules():
        if isinstance(m, LoRALinear):
            lora_sd[f"{name}.A"] = m.A.detach().cpu()
            lora_sd[f"{name}.B"] = m.B.detach().cpu()

    return lora_sd

def _load_lora_state_dict(enc: nn.Module, lora_sd: dict, device: torch.device):
    missing = 0
    for name, m in enc.named_modules():
        if not isinstance(m, LoRALinear):
            continue
        kA = f"{name}.A"
        kB = f"{name}.B"
        if (kA in lora_sd) and (kB in lora_sd):
            m.A.data.copy_(lora_sd[kA].to(device=device, dtype=m.A.dtype))
            m.B.data.copy_(lora_sd[kB].to(device=device, dtype=m.B.dtype))
        else:
            missing += 1
    if missing > 0:
        print(f"[LoRA] warn: {missing} LoRALinear modules missing keys in ckpt (ok if arch/targets changed).")

def save_fewshot_ckpt(model: nn.Module, save_path: str, info: dict = None):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    if not hasattr(model, "backbone") or not hasattr(model.backbone, "encoder"):
        raise ValueError("save_fewshot_ckpt: model has no backbone.encoder")

    enc = model.backbone.encoder
    to_save = {
        "decoder": model.decoder.state_dict() if hasattr(model, "decoder") and model.decoder is not None else None,
        "lora": _collect_lora_state_dict(enc),
        "info": info or {},
    }
    torch.save(to_save, save_path)
    print(f"[Fewshot CKPT] saved -> {save_path}")

def load_fewshot_ckpt_into_model(model: nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    set_trainable(model, mode="fewshot") 
    model.eval()

    if isinstance(ckpt, dict) and ckpt.get("decoder", None) is not None:
        model.decoder.load_state_dict(ckpt["decoder"], strict=True)

    if not hasattr(model, "backbone") or not hasattr(model.backbone, "encoder"):
        raise ValueError("load_fewshot_ckpt_into_model: model has no backbone.encoder")
    enc = model.backbone.encoder
    lora_sd = ckpt.get("lora", {}) if isinstance(ckpt, dict) else {}
    _load_lora_state_dict(enc, lora_sd, device=device)

    return ckpt
