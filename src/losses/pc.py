import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans


def extract_pixel_features_from_subset(
    model, subset, device, img_size,
    max_pixels_per_image, sample_pos_ratio,
    batch_size, num_workers, normalize
):
    model.eval()
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    feats_list, labels_list = [], []
    for imgs, masks, tile_ids in tqdm(loader, desc="Pixel feats", ncols=100, file=sys.__stderr__):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        tokens = model.backbone(imgs)          # [B,1024,h,w]
        pixel_feats = model.proj_head(tokens)  # [B,50,h,w]
        pixel_feats_up = F.interpolate(pixel_feats, size=(img_size, img_size), mode="nearest")

        B, D, H, W = pixel_feats_up.shape
        for b in range(B):
            feat_map = pixel_feats_up[b]  # [D,H,W]
            mask = masks[b]               # [1,H,W]

            feat_flat = feat_map.permute(1, 2, 0).reshape(-1, D).detach().cpu().numpy()
            label_flat = mask.permute(1, 2, 0).reshape(-1).detach().cpu().numpy().astype(np.int64)

            pos_idx = np.where(label_flat == 1)[0]
            neg_idx = np.where(label_flat == 0)[0]

            n_pos_target = int(max_pixels_per_image * sample_pos_ratio)
            n_neg_target = max_pixels_per_image - n_pos_target

            if len(pos_idx) > n_pos_target:
                pos_sel = np.random.choice(pos_idx, n_pos_target, replace=False)
            else:
                pos_sel = pos_idx

            if len(neg_idx) > n_neg_target:
                neg_sel = np.random.choice(neg_idx, n_neg_target, replace=False)
            else:
                neg_sel = neg_idx

            sel_idx = (
                np.concatenate([pos_sel, neg_sel])
                if (pos_sel.size + neg_sel.size) > 0
                else np.array([], dtype=np.int64)
            )
            if sel_idx.size == 0:
                continue

            sel_feats = feat_flat[sel_idx]
            sel_labels = label_flat[sel_idx]

            if normalize:
                sel_feats = sel_feats / (np.linalg.norm(sel_feats, axis=1, keepdims=True) + 1e-8)

            feats_list.append(sel_feats.astype(np.float32))
            labels_list.append(sel_labels.astype(np.int64))

    if len(feats_list) == 0:
        raise RuntimeError("No pixel features collected for this fold train subset.")
    feats_all = np.concatenate(feats_list, axis=0)
    labels_all = np.concatenate(labels_list, axis=0)

    return feats_all, labels_all

def kmeans_pixel_by_class(feats, labels, k_pos, k_neg, seed):
    pos = feats[labels == 1]
    neg = feats[labels == 0]
    if len(pos) < k_pos or len(neg) < k_neg:
        raise RuntimeError(f"Not enough samples for pixel kmeans: pos={len(pos)} neg={len(neg)}")

    pos = pos / (np.linalg.norm(pos, axis=1, keepdims=True) + 1e-8)
    neg = neg / (np.linalg.norm(neg, axis=1, keepdims=True) + 1e-8)

    kpos = MiniBatchKMeans(
        n_clusters=k_pos,
        random_state=seed
    ).fit(pos)

    kneg = MiniBatchKMeans(
        n_clusters=k_neg,
        random_state=seed
    ).fit(neg)

    return kpos.cluster_centers_.astype(np.float32), kneg.cluster_centers_.astype(np.float32)

def build_pixel_prototypes(model, dataset, tr_idx, device, cfg, fold: int):
    train_subset = Subset(dataset, tr_idx.tolist())

    feats, labels = extract_pixel_features_from_subset(
        model=model,
        subset=train_subset,
        device=device,
        img_size=cfg["img_size"],
        max_pixels_per_image=cfg["max_pixels_per_image"],
        sample_pos_ratio=cfg["sample_pos_ratio"],
        batch_size=cfg["pixel_batch_size"],
        num_workers=cfg["pixel_num_workers"],
        normalize=cfg["pixel_normalize"],
    )

    pp, pn = kmeans_pixel_by_class(
        feats, labels,
        k_pos=cfg["k_pos_pixel"],
        k_neg=cfg["k_neg_pixel"],
        seed=cfg["kfold_seed"] + fold,
    )

    pos = torch.from_numpy(pp).float().to(device)
    neg = torch.from_numpy(pn).float().to(device)

    return pos, neg

def assign_proto_labels(pixel_feats: torch.Tensor, prototypes_pos: torch.Tensor, prototypes_neg: torch.Tensor, target_size: tuple = None, normalize_feats: bool = True):
    B, D, h, w = pixel_feats.shape
    if target_size is not None and (h != target_size[0] or w != target_size[1]):
        pixel_feats = F.interpolate(pixel_feats, size=target_size, mode='nearest')
        h, w = target_size

    N = B * h * w
    F_flat = pixel_feats.permute(0,2,3,1).reshape(N, D)  # [N, D]

    P_pos = prototypes_pos
    P_neg = prototypes_neg
    if normalize_feats:
        F_flat = F_flat / (F_flat.norm(dim=1, keepdim=True) + 1e-8)
        P_pos = P_pos / (P_pos.norm(dim=1, keepdim=True) + 1e-8)
        P_neg = P_neg / (P_neg.norm(dim=1, keepdim=True) + 1e-8)

    P_all = torch.cat([P_pos, P_neg], dim=0)  # [K_all, D]
    K_pos = P_pos.shape[0]

    F_sq = (F_flat * F_flat).sum(dim=1, keepdim=True)  # [N,1]
    P_sq = (P_all * P_all).sum(dim=1).unsqueeze(0)     # [1,K_all]
    FP = F_flat @ P_all.t()                            # [N, K_all]
    dists = F_sq - 2.0 * FP + P_sq
    nearest = dists.argmin(dim=1)
    y_proto_flat = (nearest < K_pos).float()
    y_proto = y_proto_flat.view(B, 1, h, w)

    return y_proto

def compute_proto_consistency_loss(seg_logits: torch.Tensor, y_proto: torch.Tensor, mask: torch.Tensor = None):
    loss_map = F.binary_cross_entropy_with_logits(seg_logits, y_proto, reduction="none")
    if mask is not None:
        loss_map = loss_map * mask
        denom = mask.sum().clamp_min(1.0)
        return loss_map.sum() / denom
    else:
        return loss_map.mean()

# def compute_samplewise_proto_weights(
#     pixel_feats: torch.Tensor,         
#     logits: torch.Tensor,              
#     prototypes_pos: dict,              
#     prototypes_neg: dict,              
#     tau: float = 0.01,
#     target_size: tuple = None,
#     min_w: float = 0.0,
#     topk: int = 0,
#     cfg: dict = None
# ):
#     B = pixel_feats.size(0)
#     regions_order = [r for r in cfg["source_regions"] if r in prototypes_pos]
#     R = len(regions_order)
#     assert R > 0

#     probs = torch.sigmoid(logits)
#     pred = (probs > 0.5).float()

#     scores = torch.zeros((B, R), device=pixel_feats.device, dtype=torch.float32)

#     for j, r in enumerate(regions_order):
#         pp = prototypes_pos[r]
#         pn = prototypes_neg[r]
#         y_proto_r = assign_proto_labels(
#             pixel_feats, pp, pn,
#             target_size=target_size
#         )

#         agree = (pred == y_proto_r).float().mean(dim=(1,2,3))  
#         scores[:, j] = agree

#     w = torch.softmax(scores / max(tau, 1e-6), dim=1)

#     if min_w > 0:
#         w = torch.clamp(w, min=min_w)
#         w = w / (w.sum(dim=1, keepdim=True) + 1e-12)

#     if topk and topk > 0 and topk < R:
#         topv, topi = torch.topk(w, k=topk, dim=1)
#         w2 = torch.zeros_like(w)
#         w2.scatter_(1, topi, topv)
#         w = w2 / (w2.sum(dim=1, keepdim=True) + 1e-12)

#     return w, regions_order, scores
