from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def soft_skeletonize(x: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    for _ in range(iterations):
        min_pool = -F.max_pool3d(-x, kernel_size=3, stride=1, padding=1)
        contour = F.relu(F.max_pool3d(min_pool, kernel_size=3, stride=1, padding=1) - min_pool)
        x = F.relu(x - contour)
    return x


def soft_cldice_loss(
    seg_logits: torch.Tensor,
    centerline_logits: torch.Tensor,
    target_mask: torch.Tensor,
    target_centerline: torch.Tensor,
    iterations: int = 10,
    eps: float = 1e-5,
) -> torch.Tensor:
    seg_prob = torch.sigmoid(seg_logits)
    center_prob = torch.sigmoid(centerline_logits)

    pred_skel = soft_skeletonize(seg_prob, iterations=iterations)
    target_skel = soft_skeletonize(target_mask, iterations=iterations)

    tprec = ((pred_skel * target_mask).sum() + eps) / (pred_skel.sum() + eps)
    tsens = ((target_skel * seg_prob).sum() + eps) / (target_skel.sum() + eps)

    # Auxiliary alignment between explicit centerline head and vessel topology targets.
    cprec = ((center_prob * target_centerline).sum() + eps) / (center_prob.sum() + eps)
    csens = ((target_centerline * center_prob).sum() + eps) / (target_centerline.sum() + eps)

    cldice = 1.0 - (2.0 * tprec * tsens) / (tprec + tsens + eps)
    center_align = 1.0 - (2.0 * cprec * csens) / (cprec + csens + eps)
    return 0.7 * cldice + 0.3 * center_align


def topology_centerline_loss(
    outputs: dict[str, torch.Tensor],
    target_mask: torch.Tensor,
    target_centerline: torch.Tensor,
    target_radius: torch.Tensor | None = None,
    target_branchpoints: torch.Tensor | None = None,
    bce_weight: float = 0.25,
    centerline_weight: float = 0.20,
    cldice_weight: float = 0.25,
    radius_weight: float = 0.15,
    branchpoint_weight: float = 0.15,
) -> tuple[torch.Tensor, dict[str, float]]:
    seg_logits = outputs["seg_logits"]
    center_logits = outputs["centerline_logits"]
    branch_logits = outputs.get("branchpoint_logits")
    radius_logits = outputs.get("radius_logits")

    dice = soft_dice_loss(seg_logits, target_mask)
    bce = F.binary_cross_entropy_with_logits(seg_logits, target_mask)
    center_bce = F.binary_cross_entropy_with_logits(center_logits, target_centerline)
    cldice = soft_cldice_loss(seg_logits, center_logits, target_mask, target_centerline)
    branch_bce = (
        F.binary_cross_entropy_with_logits(branch_logits, target_branchpoints)
        if (branch_logits is not None and target_branchpoints is not None)
        else seg_logits.new_tensor(0.0)
    )
    radius_l1 = (
        F.smooth_l1_loss(F.relu(radius_logits), target_radius)
        if (radius_logits is not None and target_radius is not None)
        else seg_logits.new_tensor(0.0)
    )

    base_weight = 1.0 - bce_weight - centerline_weight - cldice_weight - radius_weight - branchpoint_weight
    total = base_weight * dice
    total = (
        total
        + bce_weight * bce
        + centerline_weight * center_bce
        + cldice_weight * cldice
        + radius_weight * radius_l1
        + branchpoint_weight * branch_bce
    )

    stats = {
        "loss_total": float(total.detach().cpu()),
        "loss_dice": float(dice.detach().cpu()),
        "loss_bce": float(bce.detach().cpu()),
        "loss_centerline": float(center_bce.detach().cpu()),
        "loss_cldice": float(cldice.detach().cpu()),
        "loss_radius": float(radius_l1.detach().cpu()),
        "loss_branchpoint": float(branch_bce.detach().cpu()),
    }
    return total, stats
