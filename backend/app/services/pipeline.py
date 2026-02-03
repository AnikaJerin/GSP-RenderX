from .mesh_loader import load_mesh
from .mesh_sampler import sample_mesh_surface
from .gaussian_fitter import build_gaussians

from .gsp_encoder import write_gsp
import os

# def convert_mesh_to_gaussians(file_path: str, samples=50000):
#     mesh = load_mesh(file_path)
#     positions, normals, colors, sizes = sample_mesh_surface(mesh, samples)

#     out_path = file_path.replace(".obj", ".gsp").replace(".stl", ".gsp")
#     write_gsp(positions, normals, colors, sizes, out_path)

#     return {"gsp_url": f"/static/{os.path.basename(out_path)}"}


def convert_mesh_to_gaussians(file_path: str, samples=50000):
    mesh = load_mesh(file_path)
    positions, normals, colors, sizes = sample_mesh_surface(mesh, samples)
    gaussian_data = build_gaussians(positions, normals, colors, sizes)
    return gaussian_data
