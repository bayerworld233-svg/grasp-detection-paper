"""Training entry point.

Run as:
    python -m src.train --pretrained --output results/pretrained
    python -m src.train --output results/scratch  --epochs 30
    python -m src.train --pretrained --output results/smoke --limit 50 --epochs 2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import CornellGraspDataset
from src.model import GraspNet
from src.utils import normalize_params


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/cornell")
    p.add_argument("--output", required=True, help="dir for best.pth + train_log.json + train.log")
    p.add_argument("--pretrained", action="store_true", help="use ImageNet weights")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=5, help="early-stopping patience on val loss")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="cap dataset size for smoke tests")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_batch(targets: torch.Tensor) -> torch.Tensor:
    """Normalize per-row 5-vec targets via numpy (cheap, OK for batch sizes here)."""
    return torch.from_numpy(normalize_params(targets.numpy())).to(targets.device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    desc: str,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    loss_fn = nn.MSELoss()
    total = 0.0
    n = 0
    for images, targets in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        targets_norm = normalize_batch(targets).to(device, non_blocking=True)
        with torch.set_grad_enabled(is_train):
            preds = model(images)
            loss = loss_fn(preds, targets_norm)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += float(loss.item()) * images.size(0)
        n += images.size(0)
    return total / max(n, 1)


def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log = logging.getLogger(f"train.{os.path.basename(output_dir)}")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(os.path.join(output_dir, "train.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(sh)
    return log


def write_log_json(path: str, history: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    """Rewrite the full log JSON each epoch — robust to mid-training Colab disconnects."""
    payload = {"meta": meta, "history": history}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)
    log = setup_logging(args.output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device: {device}")
    log.info(f"args: {vars(args)}")

    train_set = CornellGraspDataset(
        root=args.data_dir, split="train", augment=True, limit=args.limit, split_seed=args.seed
    )
    val_set = CornellGraspDataset(
        root=args.data_dir, split="val", augment=False, limit=args.limit, split_seed=args.seed
    )
    log.info(f"train={len(train_set)} val={len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = GraspNet(pretrained=args.pretrained).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    history: list[dict[str, Any]] = []
    meta = {
        "pretrained": args.pretrained,
        "epochs_max": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "n_train": len(train_set),
        "n_val": len(val_set),
        "device": str(device),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "limit": args.limit,
        "seed": args.seed,
    }
    log_json_path = os.path.join(args.output, "train_log.json")
    ckpt_path = os.path.join(args.output, "best.pth")

    best_val = float("inf")
    epochs_since_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, device, f"train ep{epoch}")
        val_loss = run_epoch(model, val_loader, None, device, f"val   ep{epoch}")
        dt = time.time() - t0

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            epochs_since_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "args": vars(args),
                },
                ckpt_path,
            )
        else:
            epochs_since_improve += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val": best_val,
                "saved": improved,
                "elapsed_s": dt,
            }
        )
        write_log_json(log_json_path, history, meta)

        log.info(
            f"epoch {epoch:3d}/{args.epochs}  "
            f"train={train_loss:.5f}  val={val_loss:.5f}  best={best_val:.5f}  "
            f"saved={improved}  ({dt:.1f}s)"
        )

        if epochs_since_improve >= args.patience:
            log.info(f"early stopping at epoch {epoch} (no val improvement for {args.patience})")
            break

    log.info(f"done. best val loss: {best_val:.5f}. checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
