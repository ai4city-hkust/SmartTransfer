import os
import numpy as np


def write_target_summary(save_dir: str, results: dict, filename: str = "target_summary.tsv", digits: int = 4):
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)
    keys = ["precision", "recall", "accuracy", "f1", "iou0", "iou1", "miou"]

    def _mean_std(vals):
        vals = [float(v) for v in vals]
        if len(vals) == 0:
            return None, None
        return float(np.mean(vals)), float(np.std(vals))

    with open(out_path, "w", encoding="utf-8") as f:
        # header
        f.write("region\t" + "\t".join(keys) + "\n")

        # per-region
        for region in sorted(results.keys()):
            fold_metrics = results[region]
            cells = []
            for k in keys:
                vals = [m[k] for m in fold_metrics if (m is not None and k in m)]
                mean, std = _mean_std(vals)
                if mean is None:
                    cells.append("-")
                else:
                    cells.append(f"{mean:.{digits}f} ± {std:.{digits}f}")
            f.write(region + "\t" + "\t".join(cells) + "\n")

        cells = []
        for k in keys:
            region_means = []
            for region, fold_metrics in results.items():
                vals = [m[k] for m in fold_metrics if (m is not None and k in m)]
                mean, _ = _mean_std(vals)
                if mean is not None:
                    region_means.append(mean)
            mean, std = _mean_std(region_means)
            if mean is None:
                cells.append("-")
            else:
                cells.append(f"{mean:.{digits}f} ± {std:.{digits}f}")
        f.write("OVERALL\t" + "\t".join(cells) + "\n")

    print(f"[Target Summary TSV] Saved to: {out_path}")

def write_region_macro_summary(
    save_dir: str,
    results: dict,
    filename: str = "region_summary.tsv",
    title: str = None,
    digits: int = 4,
):
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)
    keys = ["precision", "recall", "accuracy", "f1", "iou0", "iou1", "miou"]

    def _mean_std(vals):
        vals = [float(v) for v in vals]
        if len(vals) == 0:
            return None, None
        return float(np.mean(vals)), float(np.std(vals))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("region\t" + "\t".join(keys) + "\n")

        for region in sorted(results.keys()):
            ms = results[region]
            cells = []
            for k in keys:
                vals = [m[k] for m in ms if (m is not None and k in m)]
                mean, std = _mean_std(vals)
                if mean is None:
                    cells.append("-")
                else:
                    cells.append(f"{mean:.{digits}f} ± {std:.{digits}f}")
            f.write(region + "\t" + "\t".join(cells) + "\n")

        cells = []
        for k in keys:
            region_means = []
            for region, ms in results.items():
                vals = [m[k] for m in ms if (m is not None and k in m)]
                mean, _ = _mean_std(vals)
                if mean is not None:
                    region_means.append(mean)
            mean, std = _mean_std(region_means)
            if mean is None:
                cells.append("-")
            else:
                cells.append(f"{mean:.{digits}f} ± {std:.{digits}f}")
        f.write("OVERALL\t" + "\t".join(cells) + "\n")

    if title:
        print(f"[{title}] Saved to: {out_path}")
    else:
        print(f"[Region Summary TSV] Saved to: {out_path}")
