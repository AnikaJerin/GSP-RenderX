from __future__ import annotations

import argparse
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.vessel_export import save_vessel_bundle
from research.option_a.model import TopologyCenterlineUNet
from research.option_a.topcow_data import normalize_mra


@torch.no_grad()
def _sliding_window_outputs(
    model: torch.nn.Module,
    image_t: torch.Tensor,
    device: torch.device,
    window_size: tuple[int, int, int] = (64, 64, 64),
    overlap: float = 0.5,
    use_amp: bool = True,
) -> dict[str, torch.Tensor]:
    _, _, depth, height, width = image_t.shape
    wz, wy, wx = window_size
    sz = max(1, int(wz * (1.0 - overlap)))
    sy = max(1, int(wy * (1.0 - overlap)))
    sx = max(1, int(wx * (1.0 - overlap)))

    def axis_starts(size: int, window: int, stride: int) -> list[int]:
        if size <= window:
            return [0]
        starts = list(range(0, max(size - window, 1), stride))
        if starts[-1] != size - window:
            starts.append(size - window)
        return starts

    z_starts = axis_starts(depth, wz, sz)
    y_starts = axis_starts(height, wy, sy)
    x_starts = axis_starts(width, wx, sx)

    accum = {
        "seg_logits": torch.zeros((1, 1, depth, height, width), dtype=torch.float32),
        "centerline_logits": torch.zeros((1, 1, depth, height, width), dtype=torch.float32),
        "branchpoint_logits": torch.zeros((1, 1, depth, height, width), dtype=torch.float32),
        "radius_logits": torch.zeros((1, 1, depth, height, width), dtype=torch.float32),
    }
    norm = torch.zeros((1, 1, depth, height, width), dtype=torch.float32)

    weight_patch = torch.ones((1, 1, wz, wy, wx), dtype=torch.float32)

    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                patch = image_t[:, :, z0 : z0 + wz, y0 : y0 + wy, x0 : x0 + wx].to(device)
                with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                    outputs = model(patch)

                z1, y1, x1 = z0 + patch.shape[-3], y0 + patch.shape[-2], x0 + patch.shape[-1]
                weight = weight_patch[:, :, : patch.shape[-3], : patch.shape[-2], : patch.shape[-1]]

                for key in accum:
                    accum[key][:, :, z0:z1, y0:y1, x0:x1] += outputs[key].detach().cpu() * weight
                norm[:, :, z0:z1, y0:y1, x0:x1] += weight

                del patch, outputs
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    for key in accum:
        accum[key] = accum[key] / torch.clamp(norm, min=1e-6)

    return accum


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    image_np: np.ndarray,
    device: torch.device,
    window_size: tuple[int, int, int] = (64, 64, 64),
    overlap: float = 0.5,
    use_amp: bool = True,
) -> dict[str, np.ndarray]:
    image_t = torch.from_numpy(normalize_mra(image_np)[None, None, ...].astype(np.float32))
    outputs = _sliding_window_outputs(
        model,
        image_t,
        device=device,
        window_size=window_size,
        overlap=overlap,
        use_amp=use_amp,
    )
    seg_prob = torch.sigmoid(outputs["seg_logits"]).squeeze().numpy().astype(np.float32)
    center_prob = torch.sigmoid(outputs["centerline_logits"]).squeeze().numpy().astype(np.float32)
    branch_prob = torch.sigmoid(outputs["branchpoint_logits"]).squeeze().numpy().astype(np.float32)
    radius_map = torch.relu(outputs["radius_logits"]).squeeze().numpy().astype(np.float32)
    vessel_mask = (seg_prob > 0.5).astype(np.uint8)
    center_mask = (center_prob > 0.5).astype(np.uint8)
    branch_mask = (branch_prob > 0.4).astype(np.uint8)

    structure_uncertainty = 1.0 - np.clip(np.abs(center_prob - branch_prob), 0.0, 1.0)
    logit_uncertainty = np.clip(1.0 - np.abs(seg_prob - 0.5) * 2.0, 0.0, 1.0)
    return {
        "vessel_prob": seg_prob,
        "vessel_mask": vessel_mask,
        "centerline_mask": center_mask,
        "branchpoint_map": branch_mask,
        "radius_map": radius_map,
        "uncertainty_map": np.clip(0.6 * logit_uncertainty + 0.4 * structure_uncertainty, 0.0, 1.0).astype(np.float32),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run TopCoW/TopBrain inference and export a local viewer bundle."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to trained .pt checkpoint")
    parser.add_argument("--image", required=True, help="Path to TopCoW MRA image .nii.gz")
    parser.add_argument("--out", required=True, help="Output .npz path")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--window-size", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision during inference")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = TopologyCenterlineUNet()
    payload = torch.load(args.checkpoint, map_location=device)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    image_nii = nib.load(args.image)
    image_np = np.asarray(image_nii.get_fdata(), dtype=np.float32)
    spacing = tuple(float(x) for x in image_nii.header.get_zooms()[:3])

    pred = run_inference(
        model,
        image_np,
        device=device,
        window_size=tuple(args.window_size),
        overlap=args.overlap,
        use_amp=not args.no_amp,
    )
    save_vessel_bundle(
        args.out,
        vessel_mask=pred["vessel_mask"],
        vessel_prob=pred["vessel_prob"],
        centerline_mask=pred["centerline_mask"],
        branchpoint_map=pred["branchpoint_map"],
        radius_map=pred["radius_map"],
        uncertainty_map=pred["uncertainty_map"],
        spacing=spacing,
        metadata={
            "checkpoint": args.checkpoint,
            "image": args.image,
            "method": "TopologyCenterlineUNetV2",
            "branchpoint_count_pred": int(np.count_nonzero(pred["branchpoint_map"])),
            "mean_radius_pred": float(np.mean(pred["radius_map"][pred["vessel_mask"] > 0])) if np.count_nonzero(pred["vessel_mask"]) > 0 else 0.0,
        },
    )
    print(f"Exported {args.out}")


if __name__ == "__main__":
    main()
