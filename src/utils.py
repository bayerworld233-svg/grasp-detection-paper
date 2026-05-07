"""Grasp rectangle utilities.

Two interchangeable representations:
  - corners: array shape (4, 2), the four (x, y) corners of an oriented rectangle
             in the order Cornell uses: corner pairs (0,1) and (2,3) are short
             "gripper-plate" edges; (1,2) and (3,0) are long "gripper-opening" edges.
  - 5-param: (x, y, theta, w, h) where (x, y) is center, theta is the orientation of
             the gripper-opening axis in radians, w is gripper opening (long side),
             and h is gripper plate width (short side).

Cornell evaluation metric (Jiang/Lenz/Saxena): a predicted grasp is correct iff
it matches some positive ground-truth grasp by:
  * Jaccard (IoU) > 0.25 on the oriented rectangles, AND
  * |angle_diff| < 30 degrees (mod pi).
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
from shapely.geometry import Polygon


# ---------------------------------------------------------------------------
# representation conversion
# ---------------------------------------------------------------------------


def corners_to_params(corners: np.ndarray) -> np.ndarray:
    """Convert (4, 2) Cornell-format corners to (x, y, theta, w, h).

    Cornell ordering: corners[0]-corners[1] is one gripper plate (short edge),
    corners[1]-corners[2] is the gripper opening (long edge / width direction).
    """
    corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    cx, cy = corners.mean(axis=0)
    # Long edge (gripper opening / width) = corners[1] -> corners[2]
    edge_w = corners[2] - corners[1]
    w = float(np.linalg.norm(edge_w))
    # Short edge (gripper plate / height) = corners[0] -> corners[1]
    edge_h = corners[1] - corners[0]
    h = float(np.linalg.norm(edge_h))
    # theta = angle of the long (width) edge
    theta = math.atan2(edge_w[1], edge_w[0])
    # Wrap to (-pi/2, pi/2] — opening direction is sign-invariant
    theta = wrap_angle(theta)
    return np.array([cx, cy, theta, w, h], dtype=np.float64)


def params_to_corners(params: Sequence[float]) -> np.ndarray:
    """Convert (x, y, theta, w, h) to (4, 2) corners in Cornell ordering."""
    cx, cy, theta, w, h = (float(v) for v in params)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    # Width axis (gripper opening) direction
    dx_w, dy_w = cos_t, sin_t
    # Height axis (gripper plate) is perpendicular
    dx_h, dy_h = -sin_t, cos_t
    hw, hh = w / 2.0, h / 2.0
    # Order matches Cornell convention used by corners_to_params (so it round-trips)
    p0 = (cx - hw * dx_w - hh * dx_h, cy - hw * dy_w - hh * dy_h)
    p1 = (cx - hw * dx_w + hh * dx_h, cy - hw * dy_w + hh * dy_h)
    p2 = (cx + hw * dx_w + hh * dx_h, cy + hw * dy_w + hh * dy_h)
    p3 = (cx + hw * dx_w - hh * dx_h, cy + hw * dy_w - hh * dy_h)
    return np.array([p0, p1, p2, p3], dtype=np.float64)


def wrap_angle(theta: float) -> float:
    """Wrap angle to (-pi/2, pi/2] — gripper orientation is direction-agnostic."""
    theta = (theta + math.pi / 2.0) % math.pi - math.pi / 2.0
    return theta


# ---------------------------------------------------------------------------
# metric primitives
# ---------------------------------------------------------------------------


def rect_iou(corners_a: np.ndarray, corners_b: np.ndarray) -> float:
    """Jaccard index of two oriented rectangles given by their 4 corners each."""
    poly_a = Polygon(corners_a).buffer(0)
    poly_b = Polygon(corners_b).buffer(0)
    if not poly_a.is_valid or not poly_b.is_valid:
        return 0.0
    union = poly_a.union(poly_b).area
    if union <= 0.0:
        return 0.0
    return float(poly_a.intersection(poly_b).area / union)


def angle_diff(a: float, b: float) -> float:
    """Smallest angle between two grasp orientations, in radians, in [0, pi/2]."""
    d = wrap_angle(a - b)
    return abs(d)


def is_correct_grasp(
    pred_params: Sequence[float],
    gt_params_list: Iterable[Sequence[float]],
    iou_threshold: float = 0.25,
    angle_threshold_deg: float = 30.0,
) -> bool:
    """Cornell metric: correct iff matches ANY GT by IoU > thresh AND angle < thresh."""
    angle_threshold = math.radians(angle_threshold_deg)
    pred_corners = params_to_corners(pred_params)
    pred_theta = float(pred_params[2])
    for gt in gt_params_list:
        if angle_diff(pred_theta, float(gt[2])) >= angle_threshold:
            continue
        gt_corners = params_to_corners(gt)
        if rect_iou(pred_corners, gt_corners) > iou_threshold:
            return True
    return False


# ---------------------------------------------------------------------------
# normalization (so the 5 dims contribute on similar scale to MSE loss)
# ---------------------------------------------------------------------------

# Targets are normalized to roughly [-1, 1] before the loss.
# x, y, w, h are in pixels of the *resized* 224x224 image; theta is in radians.
IMG_SIZE = 224


def normalize_params(params: np.ndarray) -> np.ndarray:
    """Normalize (x, y, theta, w, h) to ~[-1, 1] for stable regression."""
    p = np.asarray(params, dtype=np.float32).copy()
    p[..., 0] = (p[..., 0] / IMG_SIZE) * 2.0 - 1.0  # x
    p[..., 1] = (p[..., 1] / IMG_SIZE) * 2.0 - 1.0  # y
    p[..., 2] = p[..., 2] / (math.pi / 2.0)         # theta in (-1, 1]
    p[..., 3] = p[..., 3] / IMG_SIZE                # w (positive, ~0..1)
    p[..., 4] = p[..., 4] / IMG_SIZE                # h (positive, ~0..1)
    return p


def denormalize_params(p: np.ndarray) -> np.ndarray:
    """Inverse of normalize_params."""
    p = np.asarray(p, dtype=np.float32).copy()
    p[..., 0] = (p[..., 0] + 1.0) / 2.0 * IMG_SIZE
    p[..., 1] = (p[..., 1] + 1.0) / 2.0 * IMG_SIZE
    p[..., 2] = p[..., 2] * (math.pi / 2.0)
    p[..., 3] = p[..., 3] * IMG_SIZE
    p[..., 4] = p[..., 4] * IMG_SIZE
    return p


# ---------------------------------------------------------------------------
# Cornell annotation parsing
# ---------------------------------------------------------------------------


def parse_cornell_annotation(path: str) -> list[np.ndarray]:
    """Parse a pcd*cpos.txt or pcd*cneg.txt file.

    Each rectangle is 4 consecutive lines, one "x y" pair per line.
    Some rectangles contain NaN entries; those rectangles are skipped.
    """
    rects: list[np.ndarray] = []
    with open(path, "r") as f:
        coords: list[tuple[float, float]] = []
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            coords.append((x, y))
            if len(coords) == 4:
                arr = np.array(coords, dtype=np.float64)
                if np.isfinite(arr).all():
                    rects.append(arr)
                coords = []
    return rects


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Round-trip
    rng = np.random.default_rng(0)
    for _ in range(5):
        gt = np.array(
            [
                rng.uniform(50, 200),
                rng.uniform(50, 200),
                rng.uniform(-1.4, 1.4),
                rng.uniform(20, 80),
                rng.uniform(10, 40),
            ]
        )
        c = params_to_corners(gt)
        gt2 = corners_to_params(c)
        assert np.allclose(gt, gt2, atol=1e-6), f"round-trip failed: {gt} -> {gt2}"
    print("[utils] round-trip OK")

    # Self-IoU = 1.0
    gt = np.array([100.0, 100.0, 0.3, 60.0, 25.0])
    c = params_to_corners(gt)
    iou = rect_iou(c, c)
    assert abs(iou - 1.0) < 1e-6, f"self-IoU != 1: {iou}"
    print(f"[utils] self-IoU = {iou:.4f}")

    # Disjoint -> IoU 0
    far = np.array([400.0, 400.0, 0.3, 60.0, 25.0])
    iou0 = rect_iou(c, params_to_corners(far))
    assert iou0 == 0.0
    print("[utils] disjoint IoU = 0 OK")

    # Cornell metric
    pred_good = np.array([102.0, 99.0, 0.32, 58.0, 26.0])
    pred_bad_angle = np.array([100.0, 100.0, 1.2, 60.0, 25.0])
    pred_bad_iou = np.array([400.0, 400.0, 0.3, 60.0, 25.0])
    assert is_correct_grasp(pred_good, [gt])
    assert not is_correct_grasp(pred_bad_angle, [gt])
    assert not is_correct_grasp(pred_bad_iou, [gt])
    print("[utils] Cornell metric OK")

    # Normalization round-trip
    n = normalize_params(gt)
    dn = denormalize_params(n)
    assert np.allclose(gt, dn, atol=1e-4), (gt, dn)
    print("[utils] normalize round-trip OK")

    print("[utils] all sanity checks passed.")
