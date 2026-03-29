import numpy as np
from collections import defaultdict
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import Subset, DataLoader

def region_from_tile_id(tile_id: str) -> str:
    if "_r" in tile_id:
        return tile_id.split("_r")[0]
    if "_" in tile_id:
        return tile_id.split("_")[0]
    return ""

def build_kfold_loaders(
    dataset,
    k: int = 5,
    fold: int = 0,
    batch_size: int = 16,
    num_workers: int = 0,
    seed: int = 42,
    shuffle_train: bool = True,
    pin_memory: bool = False,
    stratify: bool = False,
):
    n = len(dataset)
    if n == 0:
        raise RuntimeError("Empty dataset")


    reg2idx = defaultdict(list)
    for i in range(n):
        _, _, tid = dataset[i]
        r = region_from_tile_id(str(tid))
        reg2idx[r].append(i)


    regions = sorted([r for r in reg2idx.keys() if r])
    if len(regions) == 0:
        raise RuntimeError("No region parsed from tile_id. Check region_from_tile_id().")

    min_cnt = min(len(reg2idx[r]) for r in regions)
    k_eff = min(int(k), int(min_cnt))
    if k_eff < 2:
        raise RuntimeError(f"min_region_count={min_cnt} too small for KFold. Need >=2.")
    if k_eff != k:
        print(f"[WARN] k={k} > min_region_count={min_cnt}, "
              f"fallback to k_eff={k_eff} so every fold has every region in train/val.")
    k = k_eff

    if fold < 0 or fold >= k:
        raise ValueError(f"fold={fold} out of range for k={k}")

    reg_sizes = {r: len(reg2idx[r]) for r in regions}
    print(f"[Regions] sizes={reg_sizes}")

    tr_all, va_all = [], []


    for r in regions:
        idxs = np.array(reg2idx[r], dtype=np.int64)
        idxs = np.array(sorted(idxs.tolist(), key=lambda ii: str(dataset[ii][2])), dtype=np.int64)

        if stratify:
            ys = []
            for ii in idxs.tolist():
                _, m, _ = dataset[ii]
                ys.append(float(m.mean().item()))
            ys = np.array(ys, dtype=np.float32)
            bins = np.digitize(ys, bins=[0.0, 0.01, 0.05, 0.2])

            bincount = np.bincount(bins.astype(np.int64))
            ok_bins = bincount[bincount > 0]
            if (len(ok_bins) == 0) or (ok_bins.min() < k):
                splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
                splits = list(splitter.split(idxs))
            else:
                splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
                splits = list(splitter.split(idxs, bins))
        else:
            splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
            splits = list(splitter.split(idxs))

        tr_local, va_local = splits[fold]
        tr_all.append(idxs[tr_local])
        va_all.append(idxs[va_local])

    train_idx = np.concatenate(tr_all, axis=0)
    val_idx   = np.concatenate(va_all, axis=0)


    train_set = Subset(dataset, train_idx.tolist())
    val_set   = Subset(dataset, val_idx.tolist())

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle_train,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    print(f"[5-fold][region-wise] fold={fold}/{k-1} train={len(train_set)} val={len(val_set)} stratify={stratify}")
    return train_loader, val_loader, train_idx, val_idx

def build_single_region_fold_indices_like_fullsup(
    dataset, k: int, fold: int, seed: int, stratify: bool
):
    n = len(dataset)
    if n == 0:
        raise RuntimeError("Empty target dataset")

    all_idx = list(range(n))
    all_idx = sorted(all_idx, key=lambda ii: str(dataset[ii][2]))
    idxs = np.array(all_idx, dtype=np.int64)

    k_eff = min(int(k), int(n))
    if k_eff < 2:
        raise RuntimeError(f"n={n} too small for KFold. Need >=2.")
    if fold < 0 or fold >= k_eff:
        raise ValueError(f"fold={fold} out of range for k_eff={k_eff}")

    if stratify:
        ys = []
        for ii in idxs.tolist():
            _, m, _ = dataset[ii]
            ys.append(float(m.mean().item()))
        ys = np.array(ys, dtype=np.float32)

        bins = np.digitize(ys, bins=[0.0, 0.01, 0.05, 0.2])

        bincount = np.bincount(bins.astype(np.int64))
        ok_bins = bincount[bincount > 0]
        if (len(ok_bins) == 0) or (ok_bins.min() < k_eff):
            splitter = KFold(n_splits=k_eff, shuffle=True, random_state=seed)
            splits = list(splitter.split(idxs))
        else:
            splitter = StratifiedKFold(n_splits=k_eff, shuffle=True, random_state=seed)
            splits = list(splitter.split(idxs, bins))
    else:
        splitter = KFold(n_splits=k_eff, shuffle=True, random_state=seed)
        splits = list(splitter.split(idxs))

    tr_local, va_local = splits[fold]
    tr_idx = idxs[tr_local].tolist()
    va_idx = idxs[va_local].tolist()
    return tr_idx, va_idx, k_eff

def subsample_train_idx_keep_all_regions(dataset, tr_idx, train_pct, seed, min_per_region=1):
    tr_idx = np.array(tr_idx, dtype=np.int64)
    if train_pct is None:
        return tr_idx
    train_pct = float(train_pct)
    if train_pct >= 1.0:
        return tr_idx
    if train_pct <= 0:
        raise ValueError("train_pct must be in (0,1].")

    rng = np.random.RandomState(int(seed))

    reg2 = {}
    for i in tr_idx.tolist():
        _, _, tid = dataset[i]
        r = region_from_tile_id(str(tid))
        reg2.setdefault(r, []).append(int(i))

    regions = sorted([r for r in reg2.keys() if r])
    if len(regions) == 0:
        return tr_idx


    reg2ordered = {}
    for r in regions:
        lst = list(reg2[r])
        rng.shuffle(lst)
        reg2ordered[r] = lst

    total_available = len(tr_idx)
    total_n = int(round(total_available * train_pct))
    total_n = max(1, min(total_n, total_available))


    sizes = np.array([len(reg2ordered[r]) for r in regions], dtype=np.float32)
    props = sizes / max(1.0, sizes.sum())

    quotas = np.floor(props * total_n).astype(int)
    quotas = np.maximum(quotas, int(min_per_region))
    quotas = np.minimum(quotas, sizes.astype(int))


    while quotas.sum() > total_n:
        j = int(np.argmax(quotas))
        if quotas[j] > int(min_per_region):
            quotas[j] -= 1
        else:
            break
    while quotas.sum() < total_n:
        j = int(np.argmax(props))
        if quotas[j] < sizes[j]:
            quotas[j] += 1
        else:
            cand = np.where(quotas < sizes)[0]
            if cand.size == 0:
                break
            quotas[int(cand[0])] += 1

    picked = []
    for r, q in zip(regions, quotas.tolist()):
        picked.extend(reg2ordered[r][:q])


    picked = list(dict.fromkeys(picked))
    if len(picked) > total_n:
        picked = picked[:total_n]

    return np.array(picked, dtype=np.int64)
