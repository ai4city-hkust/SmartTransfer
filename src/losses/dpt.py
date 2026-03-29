import torch
import torch.nn.functional as F
from pathlib import Path
import json
from pyproj import CRS, Transformer
import geopandas as gpd
from shapely.geometry import box
import numpy as np
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from src.datasets.split import region_from_tile_id


def patch_labels_3class_from_gt_and_building(
    gt_mask: torch.Tensor,   
    bmask: torch.Tensor,     
    grid: int,
    tau_b: float = 0.02,     
    tau_u: float = 0.02,    
    tau_d: float = 0.10,    
    use_ignore: bool = True, 
):
    B, _, H, W = gt_mask.shape
    assert H % grid == 0 and W % grid == 0, f"H,W must be divisible by grid. got {(H,W)} grid={grid}"
    ph, pw = H // grid, W // grid
    patch_area = float(ph * pw)

    gt = (gt_mask > 0.5).float()
    bm = (bmask > 0.5).float()

    bm_blk = bm.view(B, 1, grid, ph, grid, pw)
    gt_blk = gt.view(B, 1, grid, ph, grid, pw)

    bc = bm_blk.sum(dim=(1,3,5))  

    dib = (gt_blk * bm_blk).sum(dim=(1,3,5))

    rb = bc / (patch_area + 1e-8)            
    is_building = (rb >= tau_b)              

    rd = torch.where(bc > 0, dib / (bc + 1e-8), torch.zeros_like(bc))  

    y = torch.full((B, grid, grid), 2, device=gt_mask.device, dtype=torch.long)

    if use_ignore:
        y[is_building] = -1
    else:
        y[is_building] = 0

    undmg = is_building & (rd <= tau_u)
    dmg   = is_building & (rd >= tau_d)
    y[undmg] = 0
    y[dmg]   = 1
    
    return y

def _patch_centers_xy01(grid: int, device: torch.device):
    xs = (torch.arange(grid, device=device).float() + 0.5) / grid
    ys = (torch.arange(grid, device=device).float() + 0.5) / grid
    yy, xx = torch.meshgrid(ys, xs, indexing="ij") 
    xy = torch.stack([xx, yy], dim=-1).reshape(grid * grid, 2) 

    return xy

def _sample_triplets_per_image(y_patch_1img: torch.Tensor, max_triplets: int, use_nonbuilding_as_neg: bool):
    y = y_patch_1img.reshape(-1) 
    N = y.numel()

    a_list, p_list, n_list = [], [], []

    valid = (y != 2) & (y != -1)
    anchor_candidates = valid.nonzero(as_tuple=False).flatten()
    if anchor_candidates.numel() == 0:
        return None

    perm = anchor_candidates[torch.randperm(anchor_candidates.numel(), device=y.device)]
    perm = perm[:max_triplets] if perm.numel() > max_triplets else perm

    for a in perm.tolist():
        ya = int(y[a].item())

        pos_pool = (y == ya).nonzero(as_tuple=False).flatten()
        pos_pool = pos_pool[pos_pool != a]
        if pos_pool.numel() == 0:
            continue

        if use_nonbuilding_as_neg:
            neg_pool = (y != ya).nonzero(as_tuple=False).flatten()
        else:
            if ya == 0:
                neg_pool = (y == 1).nonzero(as_tuple=False).flatten()
            elif ya == 1:
                neg_pool = (y == 0).nonzero(as_tuple=False).flatten()
            else:
                neg_pool = torch.empty(0, device=y.device, dtype=torch.long)

        if neg_pool.numel() == 0:
            continue

        p = pos_pool[torch.randint(0, pos_pool.numel(), (1,), device=y.device)].item()
        n = neg_pool[torch.randint(0, neg_pool.numel(), (1,), device=y.device)].item()

        a_list.append(a); p_list.append(p); n_list.append(n)

    if len(a_list) == 0:
        return None

    return (
        torch.tensor(a_list, device=y.device, dtype=torch.long),
        torch.tensor(p_list, device=y.device, dtype=torch.long),
        torch.tensor(n_list, device=y.device, dtype=torch.long),
    )

def dpt_patch_loss(
    patch_emb: torch.Tensor,   
    y_patch: torch.Tensor,     
    grid: int,
    margin: float = 0.2,
    max_triplets_per_img: int = 16,
    use_nonbuilding_as_neg: bool = True,
):
    B, N, D = patch_emb.shape
    assert N == grid * grid

    xy = _patch_centers_xy01(grid, device=patch_emb.device)

    losses = []
    used = 0

    for b in range(B):
        trip = _sample_triplets_per_image(
            y_patch[b], max_triplets=max_triplets_per_img,
            use_nonbuilding_as_neg=use_nonbuilding_as_neg
        )
        if trip is None:
            continue
        a_idx, p_idx, n_idx = trip
        used += a_idx.numel()

        ea = patch_emb[b, a_idx]  # [T,D]
        ep = patch_emb[b, p_idx]
        en = patch_emb[b, n_idx]

        dap = ((ea - ep) ** 2).sum(dim=1)
        dan = ((ea - en) ** 2).sum(dim=1)

        xa = xy[a_idx] 
        xp = xy[p_idx]
        xn = xy[n_idx]

        q_ap = ((xa - xp) ** 2).sum(dim=1).sqrt()
        q_an = ((xa - xn) ** 2).sum(dim=1).sqrt()
        q_pn = ((xp - xn) ** 2).sum(dim=1).sqrt()

        norm = (2.0 ** 0.5)
        P = (q_ap + q_an - q_pn) / norm

        l = F.relu(dap - dan + P + margin)
        losses.append(l.mean())

    if len(losses) == 0:
        return torch.tensor(0.0, device=patch_emb.device), 0

    return torch.stack(losses).mean(), used

def batch_patch_pool_feats_full(
    feat_map: torch.Tensor,
    grid: int,
):
    B, D, H, W = feat_map.shape
    assert H % grid == 0 and W % grid == 0, f"H,W must be divisible by grid. got {(H,W)} grid={grid}"
    ph, pw = H // grid, W // grid

    x = feat_map.view(B, D, grid, ph, grid, pw)
    v = x.mean(dim=(3, 5))                # [B,D,grid,grid]

    return v.permute(0, 2, 3, 1).contiguous()  # [B,grid,grid,D]

def build_tile_geo_index(json_dir: str):
    json_files = list(Path(json_dir).rglob("*.json"))
    if len(json_files) == 0:
        raise RuntimeError(f"No tiles_meta json found in {json_dir}")

    tile_index = {}
    src_crs = None

    for jf in json_files:
        data = json.loads(Path(jf).read_text())

        if src_crs is None:
            src_crs = CRS.from_wkt(data["crs"])

        tfm = data["transform"]
        a, b, c, d, e, f = map(float, tfm[:6])
        A = (a, b, c, d, e, f)

        for t in data["tiles"]:
            stem = Path(t["file"]).stem
            parts = Path(t["file"]).parts
            prefix = parts[0] if len(parts) > 1 else ""

            tile_index[stem] = {
                "x0": int(t["x0"]),
                "y0": int(t["y0"]),
                "w":  int(t["w"]),
                "h":  int(t["h"]),
                "prefix": prefix,
                "A": A,
            }

    transformer = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True)

    return tile_index, transformer, src_crs

def load_footprints(footprint_dir: str, src_crs: CRS):
    fp_dir = Path(footprint_dir)
    if not fp_dir.exists():
        raise RuntimeError(f"[FP] footprint_dir not exists: {footprint_dir}")

    files = list(fp_dir.rglob("*.geojson")) + list(fp_dir.rglob("*.GeoJSON"))
    if len(files) == 0:
        raise RuntimeError(f"[FP] No .geojson found under {footprint_dir}")

    out = {}
    for fp in sorted(files):
        key = fp.stem.split("_")[0]
        gdf = gpd.read_file(fp)

        gdf = gdf.to_crs(src_crs)
        gdf = gdf[gdf.geometry.notnull()].copy()
        gdf = gdf[gdf.is_valid].copy()
        out[key] = gdf
        _ = gdf.sindex

    if len(out) == 0:
        raise RuntimeError(f"[FP] 0 footprint layers loaded from {footprint_dir}")

    return out

def tile_bounds_from_meta(meta):
    a, b, c, d, e, f = meta["A"]
    x0, y0, w, h = meta["x0"], meta["y0"], meta["w"], meta["h"]

    origin_x, origin_y = c, f
    pixel_w, pixel_h = a, e 

    minx = origin_x + x0 * pixel_w
    maxx = origin_x + (x0 + w) * pixel_w
    maxy = origin_y + y0 * pixel_h
    miny = origin_y + (y0 + h) * pixel_h

    if minx > maxx: minx, maxx = maxx, minx
    if miny > maxy: miny, maxy = maxy, miny

    return minx, miny, maxx, maxy

def rasterize_buildings(meta, gdf, out_h, out_w, all_touched=True):
    minx, miny, maxx, maxy = tile_bounds_from_meta(meta)
    tile_geom = box(minx, miny, maxx, maxy)

    cand = gdf.cx[minx:maxx, miny:maxy]
    if len(cand) == 0:
        return np.zeros((out_h, out_w), np.uint8)

    sub = cand[cand.intersects(tile_geom)]
    if len(sub) == 0:
        return np.zeros((out_h, out_w), np.uint8)

    tfm = from_bounds(minx, miny, maxx, maxy, out_w, out_h)
    shapes = ((geom, 1) for geom in sub.geometry)

    mask = rasterize(
        shapes=shapes,
        out_shape=(out_h, out_w),
        transform=tfm,
        fill=0,
        all_touched=all_touched,
        dtype="uint8",
    )
    
    return (mask > 0).astype(np.uint8)

def get_bmask_batch_cached(
    tile_ids,
    tile_index,
    footprints,
    H: int,
    W: int,
    device: torch.device,
    all_touched: bool,
    cache: dict,
):
    bmask_list = []
    for tid_in in tile_ids:
        tid_in = str(tid_in)
        tid = str(tid_in).replace("\\","/").split("/")[-1]
        
        if tid in cache:
            bm = cache[tid]
        else:
            meta = tile_index.get(tid, None)
            if meta is None:
                bm = np.zeros((H, W), np.uint8)
            else:
                region = region_from_tile_id(tid) or meta.get("prefix", "")
                if (not region) or (region not in footprints):
                    bm = np.zeros((H, W), np.uint8)
                else:
                    bm = rasterize_buildings(
                        meta, footprints[region],
                        out_h=H, out_w=W,
                        all_touched=all_touched
                    ).astype(np.uint8)
            cache[tid] = bm

        bmask_list.append(bm[None, ...])

    bmask = torch.from_numpy(np.stack(bmask_list, axis=0)).float().to(device)

    return bmask
