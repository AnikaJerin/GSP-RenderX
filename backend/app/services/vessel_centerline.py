from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from ..models.gaussian import GaussianPrimitiveSet
from ..models.structural import VesselStructuralFeatures
from .vessel_loader import VesselVolumeData


def _as_bool_volume(volume: np.ndarray, threshold: float) -> np.ndarray:
    arr = np.asarray(volume)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D vessel volume, got shape {arr.shape}")
    if arr.dtype == np.bool_:
        return arr
    return arr >= threshold


def _derive_centerline_and_radius(
    vessel_mask: np.ndarray,
    spacing: tuple[float, float, float],
    provided_centerline: np.ndarray | None = None,
    provided_radius: np.ndarray | None = None,
):
    distance = (
        np.asarray(provided_radius, dtype=np.float32)
        if provided_radius is not None
        else ndimage.distance_transform_edt(vessel_mask, sampling=spacing).astype(np.float32)
    )

    if provided_centerline is not None:
        centerline = np.asarray(provided_centerline, dtype=bool)
    else:
        local_max = distance == ndimage.maximum_filter(distance, size=3, mode="nearest")
        centerline = vessel_mask & local_max & (distance > max(spacing) * 0.35)

        if np.count_nonzero(centerline) == 0 and np.count_nonzero(vessel_mask) > 0:
            relaxed_max = distance == ndimage.maximum_filter(distance, size=5, mode="nearest")
            centerline = vessel_mask & relaxed_max & (distance > 0)

    if np.count_nonzero(centerline) == 0 and np.count_nonzero(vessel_mask) > 0:
        centerline = vessel_mask.copy()

    # Remove isolated noise voxels from the centerline proxy.
    neighbor_count = ndimage.convolve(
        centerline.astype(np.uint8), np.ones((3, 3, 3), dtype=np.uint8), mode="constant"
    ) - centerline.astype(np.uint8)
    centerline &= neighbor_count > 0

    if np.count_nonzero(centerline) == 0 and np.count_nonzero(vessel_mask) > 0:
        centerline = vessel_mask.copy()

    return centerline, distance


def _neighbor_statistics(centerline: np.ndarray):
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    neighbors = ndimage.convolve(centerline.astype(np.uint8), kernel, mode="constant")
    neighbors = neighbors - centerline.astype(np.uint8)
    endpoint_count = int(np.count_nonzero(centerline & (neighbors == 1)))
    branchpoint_count = int(np.count_nonzero(centerline & (neighbors >= 3)))
    return neighbors, endpoint_count, branchpoint_count


def extract_vessel_structural_features(
    vessel: VesselVolumeData, threshold: float = 0.5
) -> tuple[VesselStructuralFeatures, np.ndarray, np.ndarray, np.ndarray]:
    vessel_mask = _as_bool_volume(vessel.volume, threshold=threshold)
    centerline, radius_map = _derive_centerline_and_radius(
        vessel_mask,
        vessel.spacing,
        provided_centerline=vessel.centerline,
        provided_radius=vessel.radius_map,
    )
    _, endpoint_count, branchpoint_count = _neighbor_statistics(centerline)

    active_voxel_count = int(np.count_nonzero(vessel_mask))
    voxel_count = int(vessel_mask.size)
    vessel_density = float(active_voxel_count / max(voxel_count, 1))
    centerline_voxel_count = int(np.count_nonzero(centerline))

    if centerline_voxel_count > 0:
        centerline_radii = radius_map[centerline]
        mean_radius = float(np.mean(centerline_radii))
        max_radius = float(np.max(centerline_radii))
    else:
        mean_radius = 0.0
        max_radius = 0.0

    approximate_total_length = float(centerline_voxel_count * float(np.mean(vessel.spacing)))

    return (
        VesselStructuralFeatures(
            voxel_count=voxel_count,
            active_voxel_count=active_voxel_count,
            vessel_density=vessel_density,
            centerline_voxel_count=centerline_voxel_count,
            branchpoint_count=branchpoint_count,
            endpoint_count=endpoint_count,
            mean_radius=mean_radius,
            max_radius=max_radius,
            approximate_total_length=approximate_total_length,
            uncertainty_ready=vessel.uncertainty_map is not None,
        ),
        vessel_mask,
        centerline,
        radius_map,
    )


def _estimate_tangents(centerline: np.ndarray, coords: np.ndarray) -> np.ndarray:
    lookup = {tuple(coord.tolist()): idx for idx, coord in enumerate(coords)}
    tangents = np.zeros((len(coords), 3), dtype=np.float32)

    for i, coord in enumerate(coords):
        vectors = []
        cx, cy, cz = coord.tolist()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    neighbor = (cx + dx, cy + dy, cz + dz)
                    if lookup.get(neighbor) is not None:
                        vectors.append(np.array([dx, dy, dz], dtype=np.float32))

        if vectors:
            tangent = np.sum(vectors, axis=0)
            norm = float(np.linalg.norm(tangent))
            if norm > 1e-6:
                tangents[i] = tangent / norm
                continue

        tangents[i] = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    return tangents


def vessel_volume_to_gaussians(
    vessel: VesselVolumeData,
    threshold: float = 0.5,
    max_points: int = 120000,
) -> tuple[GaussianPrimitiveSet, VesselStructuralFeatures]:
    features, vessel_mask, centerline, radius_map = extract_vessel_structural_features(
        vessel, threshold=threshold
    )

    coords = np.argwhere(centerline)
    if len(coords) == 0:
        raise ValueError("No centerline voxels were found in the uploaded vessel volume.")

    if len(coords) > max_points:
        stride = int(math.ceil(len(coords) / max_points))
        coords = coords[::stride]

    spacing = np.asarray(vessel.spacing, dtype=np.float32)
    center = (np.asarray(vessel_mask.shape, dtype=np.float32) - 1.0) / 2.0
    positions = (coords.astype(np.float32) - center[None, :]) * spacing[None, :]
    normals = _estimate_tangents(centerline, coords)

    radii = radius_map[tuple(coords.T)].astype(np.float32)
    if np.max(radii) > 0:
        radii_norm = radii / np.max(radii)
    else:
        radii_norm = np.zeros_like(radii)

    uncertainty_map = vessel.uncertainty_map
    if uncertainty_map is not None and np.asarray(uncertainty_map).shape == vessel_mask.shape:
        uncertainty = np.asarray(uncertainty_map, dtype=np.float32)[tuple(coords.T)]
        uncertainty = np.clip(uncertainty, 0.0, 1.0)
    else:
        uncertainty = np.zeros(len(coords), dtype=np.float32)

    # Cool-to-warm color ramp: thick reliable trunks are brighter cyan, uncertain regions warm up.
    colors = np.stack(
        [
            0.20 + 0.65 * uncertainty,
            0.35 + 0.55 * radii_norm,
            0.55 + 0.35 * (1.0 - uncertainty),
        ],
        axis=1,
    ).astype(np.float32)
    colors = np.clip(colors, 0.0, 1.0)

    sizes = np.clip(radii * 0.65, 0.003, 0.03).astype(np.float32)

    return (
        GaussianPrimitiveSet(
            positions=positions.astype(np.float32),
            normals=normals.astype(np.float32),
            colors=colors,
            sizes=sizes,
            metadata={
                "generator": "vessel_centerline_proxy",
                "domain": "vessel",
                "status": "ready",
                "threshold": threshold,
                "max_points": max_points,
                "spacing": tuple(float(x) for x in vessel.spacing),
                "structural_features": features.as_dict(),
                "used_colab_centerline": vessel.centerline is not None,
                "used_colab_radius_map": vessel.radius_map is not None,
                "used_colab_uncertainty": vessel.uncertainty_map is not None,
                "source_metadata": vessel.metadata or {},
            },
        ),
        features,
    )
