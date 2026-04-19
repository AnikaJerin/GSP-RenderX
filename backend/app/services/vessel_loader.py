from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np


@dataclass
class VesselVolumeData:
    volume: np.ndarray
    centerline: np.ndarray | None = None
    radius_map: np.ndarray | None = None
    uncertainty_map: np.ndarray | None = None
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    metadata: dict[str, Any] | None = None


def _coerce_spacing(value: Any) -> tuple[float, float, float]:
    if value is None:
        return (1.0, 1.0, 1.0)

    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 1:
        scalar = float(arr[0])
        return (scalar, scalar, scalar)
    if arr.size >= 3:
        return (float(arr[0]), float(arr[1]), float(arr[2]))
    return (1.0, 1.0, 1.0)


def _extract_volume_from_npz(bundle: np.lib.npyio.NpzFile) -> VesselVolumeData:
    preferred_volume_keys = (
        "vessel_mask",
        "vessel_prob",
        "volume",
        "segmentation",
        "prediction",
        "mask",
    )
    volume = None
    for key in preferred_volume_keys:
        if key in bundle:
            volume = np.asarray(bundle[key])
            break

    if volume is None:
        available = ", ".join(bundle.files)
        raise ValueError(
            "Could not find a vessel volume in the .npz file. "
            f"Expected one of {preferred_volume_keys}, found: {available}"
        )

    centerline = np.asarray(bundle["centerline_mask"]) if "centerline_mask" in bundle else None
    radius_map = np.asarray(bundle["radius_map"]) if "radius_map" in bundle else None
    uncertainty_map = (
        np.asarray(bundle["uncertainty_map"]) if "uncertainty_map" in bundle else None
    )
    spacing = _coerce_spacing(bundle["spacing"]) if "spacing" in bundle else (1.0, 1.0, 1.0)

    metadata = {
        "available_keys": list(bundle.files),
    }

    if "metadata_json" in bundle:
        try:
            metadata["colab_metadata_json"] = str(bundle["metadata_json"].item())
        except Exception:
            metadata["colab_metadata_json"] = "unreadable"

    return VesselVolumeData(
        volume=volume,
        centerline=centerline,
        radius_map=radius_map,
        uncertainty_map=uncertainty_map,
        spacing=spacing,
        metadata=metadata,
    )


def load_vessel_volume(file_path: str) -> VesselVolumeData:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".npy":
        volume = np.load(file_path, allow_pickle=True)
        if isinstance(volume, np.ndarray) and volume.dtype != object:
            return VesselVolumeData(volume=np.asarray(volume), metadata={"source_ext": ext})

        if isinstance(volume, np.ndarray) and volume.dtype == object:
            obj = volume.item()
            if isinstance(obj, dict) and "volume" in obj:
                return VesselVolumeData(
                    volume=np.asarray(obj["volume"]),
                    centerline=np.asarray(obj["centerline"]) if "centerline" in obj else None,
                    radius_map=np.asarray(obj["radius_map"]) if "radius_map" in obj else None,
                    uncertainty_map=(
                        np.asarray(obj["uncertainty_map"]) if "uncertainty_map" in obj else None
                    ),
                    spacing=_coerce_spacing(obj.get("spacing")),
                    metadata={"source_ext": ext},
                )

        raise ValueError("Unsupported .npy vessel bundle format.")

    if ext == ".npz":
        with np.load(file_path, allow_pickle=True) as bundle:
            data = _extract_volume_from_npz(bundle)
            merged_meta = {"source_ext": ext}
            if data.metadata:
                merged_meta.update(data.metadata)
            data.metadata = merged_meta
            return data

    raise ValueError(f"Unsupported vessel volume type '{ext}'. Use .npy or .npz.")
