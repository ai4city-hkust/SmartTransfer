import os, sys, json, traceback
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from src.datasets.seg_dataset import SegPNGRegionDataset
from src.models.dinov3_seg import DINOv3SegModel
from src.models.resnet_seg import  ResNetSegModel
from src.models.yololiked_seg import YOLOSegModel
from src.utils.checkpoints import load_decoder_only_into_model, save_decoder_only
from src.utils.seed import set_seed
from src.utils.logger import setup_logging
from src.utils.io import set_trainable
from src.losses.focal import BinaryFocalLoss2d
from src.losses.pc import assign_proto_labels, compute_proto_consistency_loss, build_pixel_prototypes
from src.losses.dpt import build_tile_geo_index, load_footprints, get_bmask_batch_cached, patch_labels_3class_from_gt_and_building, batch_patch_pool_feats_full, dpt_patch_loss
from src.utils.metrics import seg_metrics_from_logits
from src.datasets.split import build_kfold_loaders, subsample_train_idx_keep_all_regions
from src.utils.io import get_tile_ids_from_indices, dump_tile_list
from src.utils.visual import export_infer_images
from src.engine.evaluator import evaluate_with_region


def train_one_epoch(
    model, loader, optimizer, scaler, loss_fn, epoch,
    prototypes_pos=None, prototypes_neg=None,
    tile_index=None, footprints=None, bmask_cache=None,
    cfg=None, device=None, use_amp=None
):
    if bmask_cache is None:
        bmask_cache = {}
    model.train()
    pbar = tqdm(loader, desc=f"[Train {epoch}]", ncols=100, file=sys.__stderr__)

    sum_total = 0.0
    sum_seg   = 0.0
    sum_proto = 0.0   
    sum_patch = 0.0  
    n_samples = 0

    total_patch_used = 0
    total_proto_pixels = 0
    total_pixels = 0

    for imgs, masks, tile_ids in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        bs = imgs.size(0)
        optimizer.zero_grad(set_to_none=True)

        batch_used = 0
        batch_total = 0

        with autocast(enabled=use_amp):
            logits, pixel_feats, patch_feats = model(imgs)

            seg_loss = loss_fn(logits, masks)

            L_proto = torch.tensor(0.0, device=device)
            if (prototypes_pos is not None) and (prototypes_neg is not None) and (cfg["lambda_pc"] > 0):
                target_h, target_w = logits.shape[2], logits.shape[3]
                
                y_proto = assign_proto_labels(pixel_feats, prototypes_pos, prototypes_neg, target_size=(target_h, target_w))
                mask_ok = (y_proto == masks).float()
                batch_total = mask_ok.numel()
                batch_used = int(mask_ok.sum().item())
                total_proto_pixels += batch_used
                total_pixels += batch_total
                L_proto = compute_proto_consistency_loss(logits, y_proto, mask=mask_ok)

            L_patch = torch.tensor(0.0, device=device)
            patch_used = 0

            if (cfg.get("lambda_dpt", 0) > 0) and (tile_ids is not None) and (tile_index is not None) and (footprints is not None):
                B, _, H, W = logits.shape
                grid = cfg["patch_grid"]
                if grid <= 0:
                    grid = 1

                bmask = get_bmask_batch_cached(
                    tile_ids=tile_ids,
                    tile_index=tile_index,
                    footprints=footprints,
                    H=H, W=W,
                    device=device,
                    all_touched=cfg["all_touched"],
                    cache=bmask_cache,          
                )  

                y_patch = patch_labels_3class_from_gt_and_building(
                    gt_mask=masks, bmask=bmask, grid=grid
                )  

                patch_feat = batch_patch_pool_feats_full(
                    feat_map=patch_feats, grid=grid
                )  
                patch_feat = patch_feat.reshape(B, grid * grid, -1)  
                patch_emb = patch_feat

                if cfg.get("normalize", True):
                    patch_emb = patch_emb / (patch_emb.norm(dim=-1, keepdim=True) + 1e-8)

                L_patch, patch_used = dpt_patch_loss(
                    patch_emb=patch_emb,
                    y_patch=y_patch,
                    grid=grid,
                    margin=cfg.get("dpt_margin", 0.2),
                    max_triplets_per_img=cfg.get("dpt_triplets_per_img", 16),
                    use_nonbuilding_as_neg=cfg.get("dpt_use_nonbuilding_as_neg", True),
                )

            total_loss = seg_loss + cfg["lambda_pc"] * L_proto + cfg.get("lambda_dpt", 0.0) * L_patch


        scaler.scale(total_loss).backward() 
        if cfg["grad_clip"] is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        sum_total += float(total_loss.detach().item()) * bs
        sum_seg   += float(seg_loss.detach().item())   * bs
        sum_proto += float(L_proto.detach().item())    * bs
        n_samples += bs
        sum_patch += float(L_patch.detach().item()) * bs
        total_patch_used += int(patch_used)

        m = seg_metrics_from_logits(logits, masks)

    avg_total = sum_total / max(1, n_samples)

    return avg_total

def run_one_fold(fold: int, cfg: dict, device: torch.device, use_amp: bool):
    fold_dir = os.path.join(cfg["save_dir"], f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    logger = setup_logging(fold_dir, fold=fold, mode=cfg.get("mode", "train"))
    try:
        set_seed(cfg["kfold_seed"] + fold)

        train_regions = cfg["all_regions"] if cfg.get("fullsup", False) else cfg["source_regions"]
        full_set = SegPNGRegionDataset(cfg["data_root"], train_regions, cfg["img_size"])

        train_loader, val_loader, tr_idx, va_idx = build_kfold_loaders(
            full_set,
            k=cfg["kfold_k"],
            fold=fold,
            batch_size=cfg["batch_size"],
            num_workers=cfg["num_workers"],
            seed=cfg["kfold_seed"],
            stratify=cfg["kfold_stratify"],
        )

        train_pct = float(cfg.get("train_pct", 1.0))
        if train_pct < 1.0:
            tr_idx = subsample_train_idx_keep_all_regions(
                dataset=full_set,
                tr_idx=tr_idx,
                train_pct=train_pct,
                seed=int(cfg.get("kfold_seed", 42) + fold),
                min_per_region=1
            )
            train_set = Subset(full_set, tr_idx.tolist())
            train_loader = DataLoader(
                train_set,
                batch_size=cfg["batch_size"],
                shuffle=True,
                num_workers=cfg["num_workers"],
                pin_memory=False
            )
            print(f"[TrainPct][keep-all-regions] fold={fold} pct={train_pct:.2f} use_train={len(train_set)}")

        with open(os.path.join(fold_dir, "split.json"), "w") as f:
            json.dump({"train_idx": tr_idx.tolist(), "val_idx": va_idx.tolist()}, f)

        val_tiles = get_tile_ids_from_indices(full_set, va_idx)
        dump_tile_list(
            os.path.join(fold_dir, "val_tiles.txt"),
            val_tiles,
            header=f"fullsup={cfg.get('fullsup', False)} fold={fold} split=val"
        )

        arch = str(cfg.get("arch", "dinov3")).lower()
        if arch == "dinov3":
            model = DINOv3SegModel(cfg["repo_dir"], cfg["weight_path"], img_size=cfg["img_size"]).to(device)
            set_trainable(model, mode="zeroshot")
        elif arch == "resnet18":
            model = ResNetSegModel(encoder_name=cfg.get("resnet_encoder", "resnet18")).to(device)
            for p in model.parameters():
                p.requires_grad = True
            model.train()
        elif arch == "resnet152":
            model = ResNetSegModel(encoder_name=cfg.get("resnet_encoder", "resnet152")).to(device)
            for p in model.parameters():
                p.requires_grad = True
            model.train()
        elif arch == "yololiked":
            model = YOLOSegModel(img_size=cfg["img_size"], base=cfg.get("yolo_base", 32),
                                 fpn_ch=cfg.get("yolo_fpn_ch", 128)).to(device)
            for p in model.parameters():
                p.requires_grad = True
            model.train()

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"]
        )
        scaler = GradScaler(enabled=use_amp)
        loss_fn = BinaryFocalLoss2d(alpha=cfg["focal_alpha"], gamma=cfg["focal_gamma"])

        prototypes_pos = prototypes_neg = None
        if cfg["lambda_pc"] > 0:
            model.eval()
            prototypes_pos, prototypes_neg = build_pixel_prototypes(
                model=model, dataset=full_set, tr_idx=tr_idx, device=device, cfg=cfg, fold=fold
            )

        tile_index = footprints = None
        if cfg.get("lambda_dpt", 0) > 0:
            tile_index, transformer, src_crs = build_tile_geo_index(cfg["geo_json_dir"])
            footprints = load_footprints(cfg["footprint_dir"], src_crs)

        best_epoch = -1
        best_miou = -1.0
        best_metrics = None
        best_region_metrics = None
        history = {"train_loss": [], "val_loss": [], "val_iou0": [], "val_iou1": [], "val_miou": [], "val_f1": []}
        bmask_cache = {}

        for epoch in range(1, cfg["epochs"] + 1):
            tr_loss = train_one_epoch(
                model, train_loader, optimizer, scaler, loss_fn, epoch,
                prototypes_pos=prototypes_pos, prototypes_neg=prototypes_neg,
                tile_index=tile_index, footprints=footprints, bmask_cache=bmask_cache,
                cfg=cfg, device=device, use_amp=use_amp
            )

            val_loss, val_metrics, val_region_metrics = evaluate_with_region(model, val_loader, loss_fn, epoch, split="Val", device=device)

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_metrics["f1"])
            history["val_iou0"].append(val_metrics["iou0"])
            history["val_iou1"].append(val_metrics["iou1"])
            history["val_miou"].append(val_metrics["miou"])

            if val_metrics["miou"] > best_miou:
                best_epoch = epoch
                best_miou = val_metrics["miou"]
                best_metrics = dict(val_metrics)
                if val_region_metrics is not None:
                    best_region_metrics = {k: dict(v) for k, v in val_region_metrics.items()}
                save_path = os.path.join(fold_dir, "best_decoder.pth")
                save_decoder_only(model, save_path, {"epoch": epoch, "miou": best_miou, "cfg": cfg, "fold": fold})

        try:
            best_ckpt = os.path.join(fold_dir, "best_decoder.pth")
            if os.path.isfile(best_ckpt):
                load_decoder_only_into_model(model, best_ckpt, device=device)
                viz_dir = os.path.join(fold_dir, "viz_val")
                export_infer_images(model, full_set, va_idx, viz_dir, thr=cfg.get("test_prob_thr", 0.5), tag=f"val_fold{fold}", device=device)
                print(f"[VIZ] Exported val -> {viz_dir}")
        except Exception:
            print("[WARN] export_infer_images (fullsup) failed")
            traceback.print_exc()

        print(f"[Fold {fold}] Best epoch={best_epoch}, best mIoU={best_miou:.4f}")
        return best_metrics, best_region_metrics

    except Exception:
        print("\n[ERROR] Exception occurred in run_one_fold!")
        traceback.print_exc()
        raise
    finally:
        sys.stdout = logger._stdout
        sys.stderr = logger._stderr
        logger.close()
