import os
import uuid

import numpy as np
import trimesh

from ..models.gaussian import GaussianPrimitiveSet

STATIC_DIR = "static"


def _convex_hull_from_points(points: np.ndarray) -> trimesh.Trimesh:
    try:
        hull = trimesh.PointCloud(points).convex_hull
        if isinstance(hull, trimesh.Trimesh) and not hull.is_empty:
            return hull
    except Exception:
        pass
    return trimesh.Trimesh(vertices=points, process=False).convex_hull


def _estimate_pitch(points: np.ndarray) -> float:
    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    diag = np.linalg.norm(bounds_max - bounds_min)
    return max(diag / 36.0, 0.01)


def build_quick_mesh_from_gaussians(
    gaussian_set: GaussianPrimitiveSet,
    max_points: int = 12000,
) -> trimesh.Trimesh:
    points = np.asarray(gaussian_set.positions, dtype=np.float32)
    if len(points) < 4:
        raise ValueError("Need at least 4 Gaussian centers to build a quick mesh.")

    if len(points) > max_points:
        step = max(1, int(np.ceil(len(points) / max_points)))
        points = points[::step]

    pitch = _estimate_pitch(points)
    snapped = np.unique(np.round(points / pitch).astype(np.int32), axis=0).astype(np.float32) * pitch

    mesh = None
    if len(snapped) >= 8:
        try:
            mesh = trimesh.voxel.ops.points_to_marching_cubes(snapped, pitch=pitch)
        except Exception:
            mesh = None

    if mesh is None or mesh.is_empty or len(mesh.faces) == 0:
        mesh = _convex_hull_from_points(points)

    if mesh.is_empty or len(mesh.vertices) == 0:
        raise ValueError("Failed to build a quick mesh from Gaussian splats.")

    try:
        mesh.remove_degenerate_faces()
    except Exception:
        pass
    try:
        mesh.remove_duplicate_faces()
    except Exception:
        pass
    mesh.remove_unreferenced_vertices()
    mesh.process(validate=True)
    return mesh


def export_quick_mesh_from_gaussians(
    gaussian_set: GaussianPrimitiveSet,
    suffix: str | None = None,
) -> dict:
    os.makedirs(STATIC_DIR, exist_ok=True)
    mesh = build_quick_mesh_from_gaussians(gaussian_set)
    mesh_name = f"{uuid.uuid4().hex}{suffix or ''}.ply"
    mesh_path = os.path.join(STATIC_DIR, mesh_name)
    mesh.export(mesh_path)
    return {
        "mesh_url": f"/static/{mesh_name}",
        "mesh_type": "ply",
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "mesh_source": "gaussian_quick_mesh",
    }
