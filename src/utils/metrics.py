import torch
from typing import Dict


def seg_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, thr=0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs > thr).long(); gts = targets.long()
    TP = ((preds==1)&(gts==1)).sum().item()
    TN = ((preds==0)&(gts==0)).sum().item()
    FP = ((preds==1)&(gts==0)).sum().item()
    FN = ((preds==0)&(gts==1)).sum().item()
    prec = TP / (TP + FP + 1e-9)
    rec  = TP / (TP + FN + 1e-9)
    acc  = (TP + TN) / (TP + TN + FP + FN + 1e-9)
    f1   = 2*prec*rec / (prec+rec + 1e-9)
    iou1 = TP / (TP + FP + FN + 1e-9)
    iou0 = TN / (TN + FN + FP + 1e-9)
    miou = 0.5*(iou0 + iou1)
    return {
        "precision": float(prec),
        "recall": float(rec),
        "accuracy": float(acc),
        "f1": float(f1),
        "iou0": float(iou0),
        "iou1": float(iou1),
        "miou": float(miou),
    }
