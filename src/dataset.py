"""Cornell Grasping Dataset wrapper.

Each sample consists of a `pcd*r.png` RGB image (640x480 native) and a
`pcd*cpos.txt` annotation listing positive grasp rectangles (4 corners each,
some with NaN entries that are skipped).

We:
  * resize images to 224x224 (matching ResNet input)
  * scale grasp corner coordinates with the same factor
  * apply joint image+grasp augmentation (hflip, ±15° rotation, color jitter)
    in TRAIN split only
  * convert to (x, y, theta, w, h) 5-parameter form
  * sample ONE positive grasp per image per epoch for the training target
  * return ALL positive grasps in eval mode for the Cornell metric
"""
from __future__ import annotations

import glob
import math
import os
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from src.utils import (
    IMG_SIZE,
    corners_to_params,
    parse_cornell_annotation,
)

# Native Cornell image size
NATIVE_W = 640
NATIVE_H = 480

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class CornellSample:
    """One Cornell image with its file paths and resized positive grasps.

    `grasps` are stored as (N, 4, 2) corner arrays in the resized 224x224 frame.
    """

    image_path: str
    cpos_path: str
    grasps: np.ndarray  # (N, 4, 2) in 224x224 coords


def _scale_corners(corners: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Scale corner x by sx, y by sy."""
    out = corners.copy()
    out[..., 0] *= sx
    out[..., 1] *= sy
    return out


def _flip_corners_h(corners: np.ndarray, width: int) -> np.ndarray:
    out = corners.copy()
    out[..., 0] = (width - 1) - out[..., 0]
    return out


def _rotate_corners(corners: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply 2x3 affine matrix M to (..., 4, 2) corner array."""
    flat = corners.reshape(-1, 2)
    ones = np.ones((flat.shape[0], 1), dtype=flat.dtype)
    h = np.concatenate([flat, ones], axis=1)  # (n, 3)
    out = (M @ h.T).T  # (n, 2)
    return out.reshape(corners.shape)


def _discover_samples(root: str) -> list[CornellSample]:
    """Find all (pcd*r.png, pcd*cpos.txt) pairs recursively under `root`."""
    pattern = os.path.join(root, "**", "pcd*r.png")
    img_paths = sorted(glob.glob(pattern, recursive=True))
    if not img_paths:
        # Some dataset layouts have files directly in root; try that too.
        img_paths = sorted(glob.glob(os.path.join(root, "pcd*r.png")))

    samples: list[CornellSample] = []
    sx = IMG_SIZE / NATIVE_W
    sy = IMG_SIZE / NATIVE_H
    for img_path in img_paths:
        cpos_path = img_path.replace("r.png", "cpos.txt")
        if not os.path.isfile(cpos_path):
            continue
        rects = parse_cornell_annotation(cpos_path)
        if not rects:
            continue
        scaled = np.stack([_scale_corners(r, sx, sy) for r in rects])
        samples.append(CornellSample(img_path, cpos_path, scaled))
    return samples


def _split_indices(
    n: int, seed: int = 42, ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
) -> dict[str, list[int]]:
    """Deterministic random 70/15/15 split by index."""
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n).tolist()
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    return {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }


class CornellGraspDataset(Dataset):
    """Cornell dataset returning (image_tensor, target_params, [all_params]).

    Args:
        root: directory containing pcd*r.png and pcd*cpos.txt
        split: "train" | "val" | "test"
        augment: apply hflip + rotation + color jitter (typically True for train only)
        return_all_grasps: if True, __getitem__ returns a third tuple element with all
            positive grasps (in 5-param form, in 224x224 coords) for the Cornell metric
        limit: cap the dataset size (for smoke tests)
        split_seed: seed for the deterministic split
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        augment: bool = False,
        return_all_grasps: bool = False,
        limit: int | None = None,
        split_seed: int = 42,
    ):
        super().__init__()
        assert split in ("train", "val", "test"), split
        self.root = root
        self.split = split
        self.augment = augment
        self.return_all_grasps = return_all_grasps

        all_samples = _discover_samples(root)
        if not all_samples:
            raise FileNotFoundError(
                f"No (pcd*r.png, pcd*cpos.txt) pairs found under {root!r}. "
                "Did scripts/download_data.sh run successfully?"
            )
        idx_map = _split_indices(len(all_samples), seed=split_seed)
        chosen = [all_samples[i] for i in idx_map[split]]
        if limit is not None:
            chosen = chosen[: int(limit)]
        self.samples: list[CornellSample] = chosen

        # Color jitter applied as PIL transform in TRAIN+augment mode
        self._color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> np.ndarray:
        """Load RGB image, resize to 224x224, return as uint8 numpy array."""
        img = Image.open(path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        return np.array(img)

    def _augment(
        self, img: np.ndarray, grasps: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply joint hflip + rotation + color jitter. `grasps` shape (N, 4, 2)."""
        h, w = img.shape[:2]

        # Random horizontal flip
        if random.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1, :])
            grasps = _flip_corners_h(grasps, w)

        # Random small rotation
        angle_deg = random.uniform(-15.0, 15.0)
        if abs(angle_deg) > 0.1:
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
            img = cv2.warpAffine(
                img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
            )
            grasps = _rotate_corners(grasps, M)

        # Color jitter (PIL roundtrip)
        pil = Image.fromarray(img)
        pil = self._color_jitter(pil)
        img = np.array(pil)

        return img, grasps

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """uint8 HWC -> normalized float CHW tensor."""
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        t = TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)
        return t

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = self._load_image(s.image_path)
        grasps = s.grasps.copy()  # (N, 4, 2)

        if self.augment:
            img, grasps = self._augment(img, grasps)

        # Convert each grasp's 4 corners to (x, y, theta, w, h)
        all_params = np.stack([corners_to_params(c) for c in grasps], axis=0).astype(
            np.float32
        )

        # Pick training target: random one in train mode, first in eval mode (deterministic)
        if self.split == "train":
            target_params = all_params[random.randrange(len(all_params))]
        else:
            target_params = all_params[0]

        img_tensor = self._to_tensor(img)
        target_tensor = torch.from_numpy(target_params)

        if self.return_all_grasps:
            return img_tensor, target_tensor, all_params
        return img_tensor, target_tensor


def collate_eval(batch):
    """Custom collate that handles variable-length all_grasps lists."""
    images = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    all_grasps = [b[2] for b in batch]
    return images, targets, all_grasps


# ---------------------------------------------------------------------------
# sanity visualization
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/cornell")
    ap.add_argument("--out", default="/tmp/sanity_dataset.png")
    ap.add_argument("--split", default="train")
    ap.add_argument("--augment", action="store_true")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.utils import params_to_corners as _ptc

    ds = CornellGraspDataset(
        root=args.root, split=args.split, augment=args.augment, return_all_grasps=True
    )
    print(f"[dataset] {len(ds)} samples in split={args.split}")
    if len(ds) == 0:
        raise SystemExit("Dataset is empty.")

    n_show = min(4, len(ds))
    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4))
    if n_show == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        img_tensor, target, all_params = ds[i]
        # Un-normalize for display
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        disp = (img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(disp)
        for p in all_params:
            c = _ptc(p)
            poly = np.concatenate([c, c[:1]], axis=0)
            ax.plot(poly[:, 0], poly[:, 1], "g-", lw=1)
        # Target rect in red
        c = _ptc(target.numpy())
        poly = np.concatenate([c, c[:1]], axis=0)
        ax.plot(poly[:, 0], poly[:, 1], "r-", lw=2)
        ax.set_title(
            f"#{i} target=(x={target[0]:.0f}, y={target[1]:.0f}, "
            f"θ={math.degrees(target[2]):.0f}°)"
        )
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=100)
    print(f"[dataset] wrote {args.out}")
