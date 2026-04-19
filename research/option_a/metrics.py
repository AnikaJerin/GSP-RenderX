from __future__ import annotations

import numpy as np
from scipy import ndimage


def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-5) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = np.count_nonzero(pred & target)
    denom = np.count_nonzero(pred) + np.count_nonzero(target)
    return float((2.0 * inter + eps) / (denom + eps))


def centerline_proxy(mask: np.ndarray, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    if np.count_nonzero(mask) == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    dist = ndimage.distance_transform_edt(mask.astype(bool), sampling=spacing)
    local_max = dist == ndimage.maximum_filter(dist, size=3, mode="nearest")
    centerline = (mask > 0) & local_max & (dist > max(spacing) * 0.35)
    if np.count_nonzero(centerline) == 0:
        centerline = mask > 0
    return centerline.astype(np.uint8)


def cldice_score(
    pred: np.ndarray,
    target: np.ndarray,
    pred_centerline: np.ndarray | None = None,
    target_centerline: np.ndarray | None = None,
    eps: float = 1e-5,
) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    pred_centerline = (
        centerline_proxy(pred) if pred_centerline is None else pred_centerline.astype(bool)
    )
    target_centerline = (
        centerline_proxy(target) if target_centerline is None else target_centerline.astype(bool)
    )

    tprec = (np.count_nonzero(pred_centerline & target) + eps) / (
        np.count_nonzero(pred_centerline) + eps
    )
    tsens = (np.count_nonzero(target_centerline & pred) + eps) / (
        np.count_nonzero(target_centerline) + eps
    )
    return float((2.0 * tprec * tsens) / (tprec + tsens + eps))


def endpoint_branchpoint_counts(centerline: np.ndarray) -> tuple[int, int]:
    centerline = centerline.astype(bool)
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    neighbors = ndimage.convolve(centerline.astype(np.uint8), kernel, mode="constant")
    neighbors = neighbors - centerline.astype(np.uint8)
    endpoints = int(np.count_nonzero(centerline & (neighbors == 1)))
    branchpoints = int(np.count_nonzero(centerline & (neighbors >= 3)))
    return endpoints, branchpoints


def topology_report(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, float]:
    pred_center = centerline_proxy(pred, spacing=spacing)
    target_center = centerline_proxy(target, spacing=spacing)
    pred_end, pred_branch = endpoint_branchpoint_counts(pred_center)
    target_end, target_branch = endpoint_branchpoint_counts(target_center)
    return {
        "dice": dice_score(pred, target),
        "cldice": cldice_score(pred, target, pred_centerline=pred_center, target_centerline=target_center),
        "pred_endpoints": float(pred_end),
        "target_endpoints": float(target_end),
        "pred_branchpoints": float(pred_branch),
        "target_branchpoints": float(target_branch),
        "endpoint_abs_error": float(abs(pred_end - target_end)),
        "branchpoint_abs_error": float(abs(pred_branch - target_branch)),
    }
