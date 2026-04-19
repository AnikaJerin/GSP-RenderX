from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.app.services.vessel_export import save_vessel_bundle


def build_sample_case(shape: tuple[int, int, int] = (64, 64, 64)):
    vessel = np.zeros(shape, dtype=np.uint8)
    centerline = np.zeros(shape, dtype=np.uint8)
    radius = np.zeros(shape, dtype=np.float32)
    uncertainty = np.zeros(shape, dtype=np.float32)

    cx = shape[0] // 2
    cy = shape[1] // 2

    trunk_z = range(8, shape[2] - 10)
    left_branch_x = range(cx, 14, -1)
    right_branch_x = range(cx, shape[0] - 14)

    # Main trunk
    for z in trunk_z:
        centerline[cx, cy, z] = 1
        radius[cx, cy, z] = 2.6 if z < shape[2] * 0.45 else 2.0
        uncertainty[cx, cy, z] = 0.08

    branch_anchor = int(shape[2] * 0.45)

    # Left branch
    for i, x in enumerate(left_branch_x):
        y = cy - i // 4
        z = branch_anchor + i // 2
        if 0 <= y < shape[1] and 0 <= z < shape[2]:
            centerline[x, y, z] = 1
            radius[x, y, z] = max(0.9, 1.8 - i * 0.04)
            uncertainty[x, y, z] = min(0.65, 0.12 + i * 0.02)

    # Right branch
    for i, x in enumerate(right_branch_x):
        y = cy + i // 5
        z = branch_anchor + i // 2
        if 0 <= y < shape[1] and 0 <= z < shape[2]:
            centerline[x, y, z] = 1
            radius[x, y, z] = max(1.0, 1.9 - i * 0.03)
            uncertainty[x, y, z] = min(0.45, 0.1 + i * 0.015)

    offsets = [
        (0, 0, 0),
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
        (1, 1, 0),
        (-1, -1, 0),
    ]

    for x, y, z in np.argwhere(centerline > 0):
        local_r = int(max(1, round(radius[x, y, z])))
        for dx, dy, dz in offsets:
            px = x + dx * local_r
            py = y + dy * local_r
            pz = z + dz
            if 0 <= px < shape[0] and 0 <= py < shape[1] and 0 <= pz < shape[2]:
                vessel[px, py, pz] = 1

    vessel_prob = vessel.astype(np.float32) * 0.82
    vessel_prob += centerline.astype(np.float32) * 0.18
    vessel_prob = np.clip(vessel_prob, 0.0, 1.0)

    return vessel, vessel_prob, centerline, radius, uncertainty


def main():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic vessel prediction bundle for the local GSP viewer."
    )
    parser.add_argument(
        "--out",
        default="sample_vessel_case.npz",
        help="Output .npz path",
    )
    args = parser.parse_args()

    vessel, vessel_prob, centerline, radius, uncertainty = build_sample_case()

    metadata = {
        "case_name": "synthetic_option_a_demo",
        "source": "local_sample_generator",
        "notes": "Synthetic branching vessel for testing the centerline workflow.",
    }

    out_path = os.path.abspath(args.out)
    save_vessel_bundle(
        out_path,
        vessel_mask=vessel,
        vessel_prob=vessel_prob,
        centerline_mask=centerline,
        radius_map=radius,
        uncertainty_map=uncertainty,
        spacing=(0.45, 0.45, 0.6),
        metadata=metadata,
    )

    print(json.dumps({"written": out_path}, indent=2))


if __name__ == "__main__":
    main()
