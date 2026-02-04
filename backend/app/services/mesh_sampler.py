import numpy as np
import trimesh


def _sample_edges(mesh, sharp_edges, target_points, edge_oversample=1.5):
    if len(sharp_edges) == 0 or target_points <= 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
        )

    v0 = mesh.vertices[sharp_edges[:, 0]]
    v1 = mesh.vertices[sharp_edges[:, 1]]
    n0 = mesh.vertex_normals[sharp_edges[:, 0]]
    n1 = mesh.vertex_normals[sharp_edges[:, 1]]
    lengths = np.linalg.norm(v1 - v0, axis=1)
    total_length = lengths.sum()
    if total_length <= 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
        )

    # allocate points proportional to edge length
    points = []
    normals = []
    for i in range(len(sharp_edges)):
        n = max(1, int(target_points * (lengths[i] / total_length) * edge_oversample))
        t = np.linspace(0, 1, n, endpoint=False)
        pts = v0[i] + (v1[i] - v0[i]) * t[:, None]
        ns = n0[i] + (n1[i] - n0[i]) * t[:, None]
        ns /= np.linalg.norm(ns, axis=1, keepdims=True) + 1e-8
        points.append(pts)
        normals.append(ns)
    return np.vstack(points).astype(np.float32), np.vstack(normals).astype(np.float32)


def sample_mesh_surface(mesh, n_points=50000, edge_angle_threshold=35, edge_oversample=1.5):
    """
    Converts a mesh into Gaussian splat data with edge preservation.
    Returns positions, normals, colors, and splat sizes.
    """

    # ---------------------------
    # 1️⃣ Surface Sampling
    # ---------------------------
    surface_pts, face_idx = trimesh.sample.sample_surface(mesh, n_points)
    surface_normals = mesh.face_normals[face_idx]

    # ---------------------------
    # 2️⃣ Detect Sharp Edges
    # ---------------------------
    adj_edges = mesh.face_adjacency_edges
    adj_angles = mesh.face_adjacency_angles

    sharp_mask = adj_angles > np.radians(edge_angle_threshold)
    sharp_edges = adj_edges[sharp_mask]

    # Sample edge points along sharp edges
    edge_pts, edge_pts_normals = _sample_edges(
        mesh, sharp_edges, target_points=int(n_points * 0.25), edge_oversample=edge_oversample
    )

    # ---------------------------
    # 3️⃣ Merge Surface + Edge Data
    # ---------------------------
    positions = np.vstack([surface_pts, edge_pts])
    normals = np.vstack([surface_normals, edge_pts_normals])

    # ---------------------------
    # 4️⃣ Generate Colors (Fake Lighting)
    # ---------------------------
    colors = (normals + 1.0) / 2.0

    # ---------------------------
    # 5️⃣ Assign Gaussian Sizes
    # Smaller splats on edges = sharper look
    # ---------------------------
    surface_sizes = np.full(len(surface_pts), 0.015)
    edge_sizes = np.full(len(edge_pts), 0.006)
    sizes = np.concatenate([surface_sizes, edge_sizes])
    

    return (
        positions.astype(np.float32),
        normals.astype(np.float32),
        colors.astype(np.float32),
        sizes.astype(np.float32),
    )
