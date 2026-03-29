import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.losses.focal import BinaryFocalLoss2d
from src.losses.pc import assign_proto_labels, compute_proto_consistency_loss
from src.losses.dpt import patch_labels_3class_from_gt_and_building, batch_patch_pool_feats_full, dpt_patch_loss, get_bmask_batch_cached
from src.utils.io import set_trainable


def fewshot_adapt(model, fs_set, device, prototypes_pos=None, prototypes_neg=None, tile_index=None, footprints=None, cfg=None):
    if len(fs_set) == 0:
        print("[Few-shot] fs_set is empty, skip adaptation.")
        return

    model.train()
    set_trainable(model, mode="fewshot")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["fewshot_lr"]
    )
    loss_fn = BinaryFocalLoss2d(alpha=cfg["focal_alpha"], gamma=cfg["focal_gamma"])

    bs = min(int(cfg.get("fewshot_batch_size", cfg.get("fewshot_n", 1))), len(fs_set))
    loader = DataLoader(fs_set, batch_size=bs, shuffle=True, num_workers=0)

    bmask_cache = {}

    for ep in range(int(cfg["fewshot_epochs"])):
        for imgs, masks, tile_ids in loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            logits, pixel_feats, patch_feats = model(imgs)

            seg_loss = loss_fn(logits, masks)

            L_proto = torch.tensor(0.0, device=device)
            if (cfg.get("lambda_pc", 0) > 0) and (prototypes_pos is not None) and (prototypes_neg is not None):
                target_h, target_w = logits.shape[2], logits.shape[3]
                y_proto = assign_proto_labels(
                    pixel_feats, prototypes_pos, prototypes_neg,
                    target_size=(target_h, target_w)
                )
                mask_ok = (y_proto == masks).float()
                L_proto = compute_proto_consistency_loss(logits, y_proto, mask=mask_ok)

            L_patch = torch.tensor(0.0, device=device)
            if (cfg.get("lambda_dpt", 0) > 0) and (tile_index is not None) and (footprints is not None):
                B, _, H, W = logits.shape
                grid = cfg["patch_grid"]

                bmask = get_bmask_batch_cached(
                    tile_ids,
                    tile_index,
                    footprints,
                    H, W,
                    device,
                    cfg["all_touched"],
                    bmask_cache,
                )

                y_patch = patch_labels_3class_from_gt_and_building(masks, bmask, grid)

                patch_feat = batch_patch_pool_feats_full(patch_feats, grid).reshape(B, grid * grid, -1)
                if cfg.get("normalize", True):
                    patch_feat = patch_feat / (patch_feat.norm(dim=-1, keepdim=True) + 1e-8)

                L_patch, _ = dpt_patch_loss(
                    patch_emb=patch_feat,
                    y_patch=y_patch,
                    grid=grid,
                    margin=cfg["dpt_margin"],
                    max_triplets_per_img=cfg["dpt_triplets_per_img"],
                    use_nonbuilding_as_neg=cfg["dpt_use_nonbuilding_as_neg"],
                )

            loss = seg_loss + cfg.get("lambda_pc", 0) * L_proto + cfg.get("lambda_dpt", 0) * L_patch
            loss.backward()
            if cfg.get("grad_clip", None) is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
