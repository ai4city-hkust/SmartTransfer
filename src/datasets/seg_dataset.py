import torch
from torchvision.transforms import v2
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def make_sat_transform(resize_size: int = 256):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.430, 0.411, 0.296),
            std=(0.213, 0.156, 0.143),
        ),
    ])

class SegPNGRegionDataset(Dataset):
    def __init__(self, root: str, regions, img_size: int = 128):
        self.root = Path(root)
        self.regions = list(regions)

        self.samples = []  # (img_path, lbl_path, region)
        for r in self.regions:
            img_dir = self.root / r / "images"
            lbl_dir = self.root / r / "labels"
            if (not img_dir.is_dir()) or (not lbl_dir.is_dir()):
                raise RuntimeError(f"Bad region structure: {r} (need images/labels)")

            img_paths = sorted(img_dir.glob("*.png"))
            if len(img_paths) == 0:
                raise RuntimeError(f"No PNG images in {img_dir}")

            for ip in img_paths:
                lp = lbl_dir / ip.name
                if not lp.exists():
                    raise RuntimeError(f"Missing label for {lp}")
                self.samples.append((ip, lp, r))

        self.tf_img = make_sat_transform(resize_size=img_size)
        self.resize_lbl = transforms.Resize((img_size, img_size), interpolation=Image.NEAREST)

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path, region = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        msk = Image.open(lbl_path).convert("L")

        img = self.tf_img(img)

        msk = self.resize_lbl(msk)
        arr = np.array(msk)
        if arr.ndim == 3:
            arr = arr[..., 0]
        bin_np = (arr > 0).astype(np.float32)
        msk = torch.from_numpy(bin_np).unsqueeze(0)

        tile_id = img_path.stem

        return img, msk, tile_id
