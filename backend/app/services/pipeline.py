from .mesh_loader import load_mesh
from .mesh_sampler import sample_mesh_surface
from .gsp_encoder import write_gsp
import os
import uuid

STATIC_DIR = "static"


def convert_mesh_to_gaussians(
    file_path: str,
    samples=50000,
    edge_angle=35,
    edge_oversample=1.5,
):
    mesh = load_mesh(file_path)
    positions, normals, colors, sizes = sample_mesh_surface(
        mesh, samples, edge_angle_threshold=edge_angle, edge_oversample=edge_oversample
    )

    os.makedirs(STATIC_DIR, exist_ok=True)
    out_name = f"{uuid.uuid4().hex}.gsp"
    out_path = os.path.join(STATIC_DIR, out_name)

    write_gsp(positions, normals, colors, sizes, out_path)

    return {
        "gsp_url": f"/static/{out_name}",
        "count": int(len(positions)),
    }




# def convert_mesh_to_gaussians(file_path: str, samples=50000):
#     mesh = load_mesh(file_path)
#     positions, normals, colors, sizes = sample_mesh_surface(mesh, samples)
#     gaussian_data = build_gaussians(positions, normals, colors, sizes)
#     return gaussian_data
