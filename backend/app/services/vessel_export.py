from __future__ import annotations

from typing import Any

import numpy as np


def save_vessel_bundle(
    output_path: str,
    vessel_mask: np.ndarray | None = None,
    vessel_prob: np.ndarray | None = None,
    centerline_mask: np.ndarray | None = None,
    radius_map: np.ndarray | None = None,
    uncertainty_map: np.ndarray | None = None,
    spacing: tuple[float, float, float] | list[float] | np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Save a Colab-friendly vessel prediction bundle that the local GSP app can load.

    At least one of vessel_mask or vessel_prob must be provided.
    """
    if vessel_mask is None and vessel_prob is None:
        raise ValueError("Provide at least one of vessel_mask or vessel_prob.")

    payload: dict[str, Any] = {}

    if vessel_mask is not None:
        payload["vessel_mask"] = np.asarray(vessel_mask)
    if vessel_prob is not None:
        payload["vessel_prob"] = np.asarray(vessel_prob, dtype=np.float32)
    if centerline_mask is not None:
        payload["centerline_mask"] = np.asarray(centerline_mask)
    if radius_map is not None:
        payload["radius_map"] = np.asarray(radius_map, dtype=np.float32)
    if uncertainty_map is not None:
        payload["uncertainty_map"] = np.asarray(uncertainty_map, dtype=np.float32)

    spacing_value = spacing if spacing is not None else (1.0, 1.0, 1.0)
    payload["spacing"] = np.asarray(spacing_value, dtype=np.float32)

    if metadata is not None:
        payload["metadata_json"] = np.array(str(metadata), dtype=object)

    np.savez_compressed(output_path, **payload)
    return output_path
