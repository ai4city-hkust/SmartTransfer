import sys
import torch
from tqdm import tqdm
from src.utils.metrics import seg_metrics_from_logits
from src.datasets.split import region_from_tile_id


def evaluate_with_region(model, loader, loss_fn, epoch, thr=0.5, split="Val", device=None):
    model.eval()
    total_loss, tot = 0.0, 0

    agg_logits, agg_masks = [], []

    reg_logits = {}
    reg_masks  = {}

    for imgs, masks, tile_ids in tqdm(loader, desc=f"[{split} {epoch}]", ncols=100, file=sys.__stderr__):
        imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        logits, _, _ = model(imgs)
        loss = loss_fn(logits, masks)

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        tot += bs

        # overall
        agg_logits.append(logits.detach().cpu())
        agg_masks.append(masks.detach().cpu())

        # per-region
        tile_ids = [str(t).replace("\\", "/").split("/")[-1] for t in tile_ids]
        for b in range(bs):
            tid = tile_ids[b]
            rid = region_from_tile_id(tid)
            if rid not in reg_logits:
                reg_logits[rid] = []
                reg_masks[rid]  = []
            reg_logits[rid].append(logits[b:b+1].detach().cpu())
            reg_masks[rid].append(masks[b:b+1].detach().cpu())

    avg_loss = total_loss / max(1, tot)
    logits_all = torch.cat(agg_logits, dim=0)
    masks_all  = torch.cat(agg_masks, dim=0)
    overall_metrics = seg_metrics_from_logits(logits_all, masks_all, thr=thr)

    print(f"[{split} {epoch} Overall]: loss={avg_loss:.4f} "
          f"Prec={overall_metrics['precision']:.4f} Rec={overall_metrics['recall']:.4f} "
          f"Acc={overall_metrics['accuracy']:.4f} F1={overall_metrics['f1']:.4f} "
          f"IoU0={overall_metrics['iou0']:.4f} IoU1={overall_metrics['iou1']:.4f} mIoU={overall_metrics['miou']:.4f}")

    per_region_metrics = {}
    for rid in sorted(reg_logits.keys()):
        l = torch.cat(reg_logits[rid], dim=0)
        m = torch.cat(reg_masks[rid],  dim=0)
        met = seg_metrics_from_logits(l, m, thr=thr)
        per_region_metrics[rid] = met
        print(f"[{split} {epoch} {rid:>12s}]: "
              f"Prec={met['precision']:.4f} Rec={met['recall']:.4f} Acc={met['accuracy']:.4f} "
              f"F1={met['f1']:.4f} IoU0={met['iou0']:.4f} IoU1={met['iou1']:.4f} mIoU={met['miou']:.4f}")

    return avg_loss, overall_metrics, per_region_metrics

def evaluate_on_test(model, loader, loss_fn, thr=0.5, split="Test", device=None):
    model.eval()
    total_loss, tot = 0.0, 0
    agg_logits, agg_masks = [], []

    for imgs, masks, tile_ids in tqdm(loader, desc=split, ncols=100, file=sys.__stderr__):
        imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)

        logits, pixel_feats, _ = model(imgs)
        loss = loss_fn(logits, masks)

        total_loss += loss.item() * imgs.size(0)
        tot += imgs.size(0)
        agg_logits.append(logits.detach().cpu())
        agg_masks.append(masks.detach().cpu())

    avg_loss = total_loss / max(1, tot)
    logits = torch.cat(agg_logits, dim=0)
    masks  = torch.cat(agg_masks, dim=0)
    metrics = seg_metrics_from_logits(logits, masks, thr=thr)

    print(f"{split}: loss={avg_loss:.4f} "
          f"Prec={metrics['precision']:.4f} Rec={metrics['recall']:.4f} Acc={metrics['accuracy']:.4f} F1={metrics['f1']:.4f} "
          f"IoU0={metrics['iou0']:.4f} IoU1={metrics['iou1']:.4f} mIoU={metrics['miou']:.4f}")

    return avg_loss, metrics
