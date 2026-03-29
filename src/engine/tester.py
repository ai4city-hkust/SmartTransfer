import os
import json
import traceback
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from src.utils.seed import set_seed
from src.datasets.seg_dataset import SegPNGRegionDataset
from src.datasets.split import build_single_region_fold_indices_like_fullsup
from src.utils.io import dump_tile_list, get_tile_ids_from_indices
from src.models.dinov3_seg import DINOv3SegModel
from src.utils.checkpoints import load_decoder_only_into_model, load_fewshot_ckpt_into_model, save_fewshot_ckpt
from src.losses.pc import build_pixel_prototypes
from src.losses.dpt import build_tile_geo_index, load_footprints
from src.engine.fewshot import fewshot_adapt
from src.losses.focal import BinaryFocalLoss2d
from src.engine.evaluator import evaluate_on_test
from src.utils.report import write_target_summary


def test(cfg=None, device=None):
    set_seed(cfg["kfold_seed"])
    thr = cfg.get("test_prob_thr", 0.5)
    mode = str(cfg.get("mode", "zeroshot"))
    fsn  = int(cfg.get("fewshot_n", 0))
    results = {}

    for target_region in cfg["target_regions"]:
        print(f"\n==============================")
        print(f" Target region: {target_region}")
        print(f"==============================")
        print(cfg["save_dir"])

        full_target_set = SegPNGRegionDataset(cfg["data_root"], regions=[target_region], img_size=cfg["img_size"])
        n = len(full_target_set)
        region_fold_metrics = []

        for fold in range(cfg["kfold_fold_start"], cfg["kfold_fold_end"]):
            fold_dir = os.path.join(cfg["save_dir"], f"fold_{fold}")
            ckpt_path = os.path.join(fold_dir, "best_decoder.pth")
            if not os.path.isfile(ckpt_path):
                print(f"[Fold {fold}] [Skip] checkpoint not found: {ckpt_path}")
                continue

            tr_pool_idx, test_idx, k_eff = build_single_region_fold_indices_like_fullsup(
                dataset=full_target_set,
                k=int(cfg["kfold_k"]),
                fold=int(fold),
                seed=int(cfg["kfold_seed"]),
                stratify=bool(cfg.get("kfold_stratify", False)),
            )

            fewshot_idx = []
            if mode == "fewshot":
                k_fs = min(int(cfg.get("fewshot_n", 0)), len(tr_pool_idx))
                if k_fs > 0:
                    rng = np.random.RandomState(int(cfg["fewshot_seed"]) + int(fold))
                    fewshot_idx = rng.choice(tr_pool_idx, size=k_fs, replace=False).tolist()
                    fewshot_set = Subset(full_target_set, fewshot_idx)
                else:
                    fewshot_set = []
            else:
                fewshot_set = []

            test_tiles = get_tile_ids_from_indices(full_target_set, test_idx)
            dump_tile_list(
                os.path.join(fold_dir, f"target_test_tiles_{target_region}.txt"),
                test_tiles,
                header=f"target={target_region} fold={fold} split=test"
            )

            print(f"[Fold {fold}] target={target_region} n={n} k_eff={k_eff} "
                  f"fewshot={len(fewshot_idx)} test={len(test_idx)}")
            

            model = DINOv3SegModel(cfg["repo_dir"], cfg["weight_path"]).to(device)
            fewshot_ckpt = os.path.join(fold_dir, f"fewshot_n{fsn}_ckpt.pth")

            if mode == "fewshot" and os.path.isfile(fewshot_ckpt):
                print(f"[SKIP-FEWSHOT] fold={fold} found existing fewshot ckpt: {fewshot_ckpt}")
                load_fewshot_ckpt_into_model(model, fewshot_ckpt, device=device)
            else:
                load_decoder_only_into_model(model, ckpt_path, device=device)

                proto_pos = proto_neg = None
                tile_index = footprints = None

                if (mode == "fewshot") and ((cfg.get("lambda_pc", 0) > 0) or (cfg.get("lambda_dpt", 0) > 0)):
                    if cfg.get("lambda_pc", 0) > 0:
                        split_path = os.path.join(fold_dir, "split.json")
                        if os.path.isfile(split_path):
                            with open(split_path, "r") as f:
                                split = json.load(f)
                            tr_idx = np.array(split["train_idx"], dtype=np.int64)

                            full_source_set = SegPNGRegionDataset(cfg["data_root"], cfg["source_regions"], cfg["img_size"])
                            tmp_model = DINOv3SegModel(cfg["repo_dir"], cfg["weight_path"]).to(device)
                            tmp_model.eval()
                            proto_pos, proto_neg = build_pixel_prototypes(
                                model=tmp_model, dataset=full_source_set, tr_idx=tr_idx, device=device, cfg=cfg, fold=fold
                            )
                            del tmp_model
                            torch.cuda.empty_cache()
                        else:
                            print(f"[WARN] split.json not found, skip proto for fewshot: {split_path}")

                    if cfg.get("lambda_dpt", 0) > 0:
                        tile_index, _, src_crs = build_tile_geo_index(cfg["geo_json_dir"])
                        footprints = load_footprints(cfg["footprint_dir"], src_crs)

                if mode == "fewshot":
                    fewshot_adapt(
                        model=model,
                        fs_set=fewshot_set,
                        device=device,
                        prototypes_pos=proto_pos,
                        prototypes_neg=proto_neg,
                        tile_index=tile_index,
                        footprints=footprints,
                        cfg=cfg
                    )

                    try:
                        save_fewshot_ckpt(
                            model=model,
                            save_path=fewshot_ckpt,
                            info={
                                "mode": "fewshot",
                                "fewshot_n": fsn,
                                "fewshot_seed": int(cfg.get("fewshot_seed", 0)),
                                "fold": int(fold),
                                "base_ckpt": ckpt_path,
                                "cfg": dict(cfg),
                            }
                        )
                    except Exception:
                        print("[WARN] save_fewshot_ckpt failed")
                        traceback.print_exc()

            loss_fn = BinaryFocalLoss2d(alpha=cfg["focal_alpha"], gamma=cfg["focal_gamma"])
            test_set = Subset(full_target_set, test_idx)
            test_loader = DataLoader(test_set, batch_size=cfg.get("test_batch_size", cfg["batch_size"]),
                                    shuffle=False, num_workers=cfg["num_workers"], pin_memory=False)
            _, metrics = evaluate_on_test(model, test_loader, loss_fn, thr=thr, split=f"Test({target_region}, fold={fold})", device=device)
            metrics = dict(metrics); metrics["fold"] = fold
            region_fold_metrics.append(metrics)

        if len(region_fold_metrics) > 0:
            results[target_region] = region_fold_metrics

    if results:
        if mode == "fewshot":
            write_target_summary(cfg["save_dir"], results, filename=f"fewshot_{cfg.get('fewshot_n')}_summary.log")
        else:
            write_target_summary(cfg["save_dir"], results, filename="target_summary.log")

    return results
