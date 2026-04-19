from __future__ import annotations

import math

import numpy as np
import trimesh

from ..models.structural import MeshStructuralFeatures, VesselStructuralFeatures


def extract_mesh_structural_features(
    mesh: trimesh.Trimesh, edge_angle_threshold: float = 35.0
) -> MeshStructuralFeatures:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    bbox_min, bbox_max = mesh.bounds
    bbox_extent = bbox_max - bbox_min
    bbox_diagonal = float(np.linalg.norm(bbox_extent))

    unique_edges = np.asarray(mesh.edges_unique)
    if len(unique_edges) > 0:
        edge_vectors = vertices[unique_edges[:, 1]] - vertices[unique_edges[:, 0]]
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        mean_edge_length = float(edge_lengths.mean())
        # A low edge-length to bbox ratio is a crude proxy for thin/high-detail structure.
        thin_structure_score = float(1.0 - min(1.0, mean_edge_length / (bbox_diagonal + 1e-8)))
    else:
        mean_edge_length = 0.0
        thin_structure_score = 0.0

    adjacency_angles = np.asarray(mesh.face_adjacency_angles)
    if adjacency_angles.size > 0:
        threshold_rad = math.radians(edge_angle_threshold)
        sharp_edge_count = int(np.count_nonzero(adjacency_angles > threshold_rad))
        sharp_edge_ratio = float(sharp_edge_count / len(adjacency_angles))
        mean_adjacency_angle_deg = float(np.degrees(adjacency_angles.mean()))
    else:
        sharp_edge_count = 0
        sharp_edge_ratio = 0.0
        mean_adjacency_angle_deg = 0.0

    return MeshStructuralFeatures(
        vertex_count=int(len(vertices)),
        face_count=int(len(faces)),
        bbox_extent=tuple(float(x) for x in bbox_extent.tolist()),
        bbox_diagonal=bbox_diagonal,
        sharp_edge_count=sharp_edge_count,
        sharp_edge_ratio=sharp_edge_ratio,
        mean_edge_length=mean_edge_length,
        thin_structure_score=thin_structure_score,
        mean_adjacency_angle_deg=mean_adjacency_angle_deg,
    )


def extract_vessel_structural_features(volume: np.ndarray) -> VesselStructuralFeatures:
    """
    Lightweight vessel-structure summary for future centerline-aware pipelines.
    This assumes a binary or probability volume and keeps computation cheap.
    """
    vol = np.asarray(volume)
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D vessel volume, got shape {vol.shape}")

    active = vol > 0
    active_voxel_count = int(np.count_nonzero(active))
    voxel_count = int(active.size)
    vessel_density = float(active_voxel_count / max(voxel_count, 1))

    # Full centerline extraction is intentionally deferred. For now, we expose
    # placeholders so the rest of the structural pipeline can be wired safely.
    return VesselStructuralFeatures(
        voxel_count=voxel_count,
        active_voxel_count=active_voxel_count,
        vessel_density=vessel_density,
        centerline_voxel_count=0,
        branchpoint_count=0,
        endpoint_count=0,
        mean_radius=0.0,
        max_radius=0.0,
        approximate_total_length=0.0,
        uncertainty_ready=True,
    )
