"""Evaluation entry point.

Computes the Cornell-metric grasp accuracy plus per-parameter MAE on the test split.

Run as:
    python -m src.evaluate --checkpoint results/pretrained/best.pth
    python -m src.evaluate --checkpoint results/pretrained/best.pth --visualize --num_samples 8
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from src.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    CornellGraspDataset,
)
from src.model import GraspNet
from src.utils import (
    angle_diff,
    denormalize_params,
    is_correct_grasp,
    params_to_corners,
    rect_iou,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", default="data/cornell")
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--seed", type=int, default=42, help="must match training seed for fair split")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--out_json", default=None, help="default: <ckpt_dir>/eval.json")
    p.add_argument("--out_png", default=None, help="default: <ckpt_dir>/qualitative.png")
    return p.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> GraspNet:
    state = torch.load(checkpoint_path, map_location=device)
    pretrained_flag = bool(state.get("args", {}).get("pretrained", False))
    # Build with random init regardless — we overwrite weights from the checkpoint.
    # We just match the architecture; the pretrained flag only controlled init.
    model = GraspNet(pretrained=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    return model, pretrained_flag


@torch.no_grad()
def evaluate_dataset(
    model: GraspNet, dataset: CornellGraspDataset, device: torch.device
) -> dict[str, Any]:
    correct = 0
    total = 0
    abs_errs = np.zeros(5, dtype=np.float64)
    per_image_records: list[dict[str, Any]] = []

    for i in tqdm(range(len(dataset)), desc="eval"):
        img_tensor, target_params, all_params = dataset[i]
        x = img_tensor.unsqueeze(0).to(device)
        pred_norm = model(x).cpu().numpy()[0]  # (5,) normalized
        pred_params = denormalize_params(pred_norm)

        # Cornell metric vs ALL positive ground-truth grasps
        ok = is_correct_grasp(pred_params, all_params)

        # Per-parameter MAE: compare against the closest ground-truth grasp by IoU
        best_iou = -1.0
        best_gt = None
        pred_corners = params_to_corners(pred_params)
        for gt in all_params:
            iou = rect_iou(pred_corners, params_to_corners(gt))
            if iou > best_iou:
                best_iou = iou
                best_gt = gt
        if best_gt is None:
            best_gt = target_params.numpy()
        diff = pred_params - best_gt
        diff[2] = angle_diff(pred_params[2], best_gt[2])  # circular for theta
        abs_errs += np.abs(diff)

        correct += int(ok)
        total += 1
        per_image_records.append(
            {
                "idx": i,
                "image": dataset.samples[i].image_path,
                "correct": bool(ok),
                "best_iou": float(best_iou),
                "pred": pred_params.tolist(),
                "best_gt": best_gt.tolist(),
            }
        )

    accuracy = correct / max(total, 1)
    mae = (abs_errs / max(total, 1)).tolist()
    return {
        "accuracy": float(accuracy),
        "n": int(total),
        "n_correct": int(correct),
        "mae_x": float(mae[0]),
        "mae_y": float(mae[1]),
        "mae_theta_rad": float(mae[2]),
        "mae_theta_deg": float(math.degrees(mae[2])),
        "mae_w": float(mae[3]),
        "mae_h": float(mae[4]),
        "per_image": per_image_records,
    }


def visualize(
    model: GraspNet,
    dataset: CornellGraspDataset,
    device: torch.device,
    out_png: str,
    num_samples: int,
    seed: int = 0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = random.Random(seed)
    n_show = min(num_samples, len(dataset))
    chosen = rng.sample(range(len(dataset)), n_show)

    cols = 4
    rows = math.ceil(n_show / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = np.atleast_2d(axes).reshape(rows, cols)

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    for k, idx in enumerate(chosen):
        ax = axes[k // cols, k % cols]
        img_tensor, target_params, all_params = dataset[idx]
        with torch.no_grad():
            pred_norm = model(img_tensor.unsqueeze(0).to(device)).cpu().numpy()[0]
        pred = denormalize_params(pred_norm)

        disp = (img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(disp)
        for gt in all_params:
            c = params_to_corners(gt)
            poly = np.concatenate([c, c[:1]], axis=0)
            ax.plot(poly[:, 0], poly[:, 1], "g-", lw=1)
        c = params_to_corners(pred)
        poly = np.concatenate([c, c[:1]], axis=0)
        ok = is_correct_grasp(pred, all_params)
        ax.plot(poly[:, 0], poly[:, 1], "-", color=("lime" if ok else "red"), lw=2)
        ax.set_title(f"#{idx} {'✓' if ok else '✗'}", fontsize=10)
        ax.axis("off")
    # Hide unused axes
    for k in range(n_show, rows * cols):
        axes[k // cols, k % cols].axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] device={device}, checkpoint={args.checkpoint}")

    model, was_pretrained = load_model(args.checkpoint, device)
    print(f"[evaluate] checkpoint trained with pretrained={was_pretrained}")

    dataset = CornellGraspDataset(
        root=args.data_dir,
        split=args.split,
        augment=False,
        return_all_grasps=True,
        limit=args.limit,
        split_seed=args.seed,
    )
    print(f"[evaluate] {len(dataset)} samples in split={args.split}")

    metrics = evaluate_dataset(model, dataset, device)
    metrics["checkpoint"] = os.path.abspath(args.checkpoint)
    metrics["split"] = args.split
    metrics["pretrained"] = was_pretrained

    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    out_json = args.out_json or os.path.join(ckpt_dir, "eval.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate] wrote {out_json}")

    print(
        f"[evaluate] accuracy={metrics['accuracy'] * 100:.2f}% "
        f"({metrics['n_correct']}/{metrics['n']})  "
        f"MAE x={metrics['mae_x']:.2f} y={metrics['mae_y']:.2f} "
        f"θ={metrics['mae_theta_deg']:.2f}° w={metrics['mae_w']:.2f} h={metrics['mae_h']:.2f}"
    )

    if args.visualize:
        out_png = args.out_png or os.path.join(ckpt_dir, "qualitative.png")
        visualize(model, dataset, device, out_png, args.num_samples)
        print(f"[evaluate] wrote {out_png}")


if __name__ == "__main__":
    main()
