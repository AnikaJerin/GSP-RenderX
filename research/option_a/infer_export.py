from __future__ import annotations

import argparse
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.vessel_export import save_vessel_bundle
from research.option_a.model import TopologyCenterlineUNet
from research.option_a.topcow_data import normalize_mra


@torch.no_grad()
def run_inference(model: torch.nn.Module, image_np: np.ndarray, device: torch.device) -> dict[str, np.ndarray]:
    image_t = torch.from_numpy(normalize_mra(image_np)[None, None, ...].astype(np.float32)).to(device)
    outputs = model(image_t)
    seg_prob = torch.sigmoid(outputs["seg_logits"]).squeeze().cpu().numpy().astype(np.float32)
    center_prob = torch.sigmoid(outputs["centerline_logits"]).squeeze().cpu().numpy().astype(np.float32)
    vessel_mask = (seg_prob > 0.5).astype(np.uint8)
    center_mask = (center_prob > 0.5).astype(np.uint8)
    return {
        "vessel_prob": seg_prob,
        "vessel_mask": vessel_mask,
        "centerline_mask": center_mask,
        "uncertainty_map": np.clip(1.0 - np.abs(seg_prob - 0.5) * 2.0, 0.0, 1.0).astype(np.float32),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run TopCoW/TopBrain inference and export a local viewer bundle."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to trained .pt checkpoint")
    parser.add_argument("--image", required=True, help="Path to TopCoW MRA image .nii.gz")
    parser.add_argument("--out", required=True, help="Output .npz path")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
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

    pred = run_inference(model, image_np, device=device)
    save_vessel_bundle(
        args.out,
        vessel_mask=pred["vessel_mask"],
        vessel_prob=pred["vessel_prob"],
        centerline_mask=pred["centerline_mask"],
        uncertainty_map=pred["uncertainty_map"],
        spacing=spacing,
        metadata={"checkpoint": args.checkpoint, "image": args.image, "method": "TopologyCenterlineUNet"},
    )
    print(f"Exported {args.out}")


if __name__ == "__main__":
    main()
