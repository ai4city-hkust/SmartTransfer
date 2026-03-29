import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from PIL import Image
from src.utils.io import _norm_tid


def _tensor_to_uint8_img(x):  # x: [3,H,W] tensor after SAT normalization
    mean = np.array([0.430, 0.411, 0.296]).reshape(3,1,1)
    std  = np.array([0.213, 0.156, 0.143]).reshape(3,1,1)
    x = x.detach().cpu().numpy()
    x = (x * std + mean).clip(0, 1)
    x = (x * 255.0).round().astype(np.uint8)        # [3,H,W]
    return np.transpose(x, (1, 2, 0))               # [H,W,3]

def export_infer_images(
    model,
    dataset,
    indices,
    out_dir,
    thr=0.5,
    tag="",
    device="cuda"
):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    for ii in indices:
        img_t, msk_t, tid = dataset[int(ii)]
        tid = _norm_tid(tid)

        inp = img_t.unsqueeze(0).to(device, non_blocking=True)
        logits, _, _ = model(inp)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        pred = (prob > thr).astype(np.uint8)

        gt = (msk_t[0].cpu().numpy() > 0.5).astype(np.uint8)

        img_vis = _tensor_to_uint8_img(img_t)
        Image.fromarray(img_vis).save(
            os.path.join(out_dir, f"{tid}_input.png")
        )

        Image.fromarray((gt * 255).astype(np.uint8)).save(
            os.path.join(out_dir, f"{tid}_gt.png")
        )

        Image.fromarray((pred * 255).astype(np.uint8)).save(
            os.path.join(out_dir, f"{tid}_pred.png")
        )

        tp = (pred == 1) & (gt == 1)
        fp = (pred == 1) & (gt == 0)
        fn = (pred == 0) & (gt == 1)

        vis = np.zeros((*gt.shape, 3), dtype=np.uint8)
        vis[tp] = (255, 255, 255)   # TP: white
        vis[fp] = (255, 0, 0)       # FP: red
        vis[fn] = (0, 0, 255)       # FN: blue

        Image.fromarray(vis).save(
            os.path.join(out_dir, f"{tid}_tp_fp_fn.png")
        )
