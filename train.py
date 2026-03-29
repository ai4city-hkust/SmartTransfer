import os
import torch
import argparse
import yaml
from src.utils.seed import set_seed
from src.engine.trainer import run_one_fold
from src.utils.report import write_region_macro_summary
from src.engine.tester import test
from src.config import get_default_cfg
from src.utils.env import setup_runtime_patches


def setup_env():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and torch.backends.cudnn.is_available()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return device, use_amp

def load_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = get_default_cfg()
    with open(args.config, "r") as f:
        yaml_cfg = yaml.safe_load(f)
    cfg.update(yaml_cfg)

    return cfg

def main():
    setup_runtime_patches()
    cfg = load_cfg()
    device, use_amp = setup_env()
    
    set_seed(cfg["kfold_seed"])
    print(
        f"save_dir={cfg.get('save_dir')}, "
        f"lambda_pc={cfg.get('lambda_pc')}, "
        f"lambda_dpt={cfg.get('lambda_dpt')}, "
        f"fullsup={cfg.get('fullsup', False)}"
    )

    region_fold_metrics = {}

    for fold in range(cfg["kfold_fold_start"], cfg["kfold_fold_end"]):
        best_overall, best_region = run_one_fold(fold, cfg, device, use_amp)

        if best_overall is None:
            continue

        if best_region is not None:
            for r, met in best_region.items():
                region_fold_metrics.setdefault(r, []).append(met)

    if region_fold_metrics:
        if cfg.get("fullsup", False):
            fname = "fullsup_region_summary.log"
            title = "FullSupervision Region-wise Summary"
        else:
            fname = "source_summary.log"
            title = "LODO Source Region-wise Summary"

        write_region_macro_summary(
            save_dir=cfg["save_dir"],
            results=region_fold_metrics,
            filename=fname,
            title=title
        )
    else:
        print("[SKIP] No folds trained in this run; keep existing region summary log unchanged.")

    if not cfg.get("fullsup", False):
        test(cfg, device)


if __name__ == "__main__":
    main()