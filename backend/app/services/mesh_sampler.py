import numpy as np
import trimesh


def sample_mesh_surface(mesh, n_points=50000, edge_angle_threshold=35):
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
    # ---------------------------

    adj_edges = mesh.face_adjacency_edges
    adj_angles = mesh.face_adjacency_angles

    sharp_mask = adj_angles > np.radians(edge_angle_threshold)
    sharp_edges = adj_edges[sharp_mask]

    edge_vertices = mesh.vertices[np.unique(sharp_edges)]

    # Approximate normals for edge points
    edge_normals = np.tile([0, 0, 1], (len(edge_vertices), 1))


    # Approximate normals for edge points
    edge_normals = np.tile([0, 0, 1], (len(edge_vertices), 1))

    # ---------------------------
    # 3️⃣ Merge Surface + Edge Data
    # ---------------------------
    positions = np.vstack([surface_pts, edge_vertices])
    normals = np.vstack([surface_normals, edge_normals])

    # ---------------------------
    # 4️⃣ Generate Colors (Fake Lighting)
    # ---------------------------
    colors = (normals + 1.0) / 2.0

    # ---------------------------
    # 5️⃣ Assign Gaussian Sizes
    # Smaller splats on edges = sharper look
    # ---------------------------
    surface_sizes = np.full(len(surface_pts), 0.015)
    edge_sizes = np.full(len(edge_vertices), 0.006)
    sizes = np.concatenate([surface_sizes, edge_sizes])
    

    return (
        positions.astype(np.float32),
        normals.astype(np.float32),
        colors.astype(np.float32),
        sizes.astype(np.float32),
    )

