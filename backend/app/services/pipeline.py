from .gsp_encoder import write_gsp
from .generator_registry import GeneratorContext, get_generator
from .quick_mesh import export_quick_mesh_from_gaussians
from .vessel_centerline import vessel_volume_to_gaussians
from .vessel_loader import load_vessel_volume
import os
import uuid

STATIC_DIR = "static"


def convert_mesh_to_gaussians(
    file_path: str,
    samples=50000,
    edge_angle=35,
    edge_oversample=1.5,
    generator_name: str = "heuristic_mesh",
):
    generator = get_generator(generator_name)
    gaussian_set = generator.generate(
        GeneratorContext(
            source_path=file_path,
            samples=samples,
            edge_angle=edge_angle,
            edge_oversample=edge_oversample,
        )
    )

    os.makedirs(STATIC_DIR, exist_ok=True)
    out_name = f"{uuid.uuid4().hex}.gsp"
    out_path = os.path.join(STATIC_DIR, out_name)

    write_gsp(
        gaussian_set.positions,
        gaussian_set.normals,
        gaussian_set.colors,
        gaussian_set.sizes,
        out_path,
    )
    quick_mesh = export_quick_mesh_from_gaussians(gaussian_set)

    return {
        "gsp_url": f"/static/{out_name}",
        "count": int(len(gaussian_set.positions)),
        "generator": generator_name,
        "metadata": gaussian_set.metadata or {},
        **quick_mesh,
    }


def convert_vessel_volume_to_gaussians(
    file_path: str,
    threshold: float = 0.5,
    max_points: int = 120000,
):
    vessel = load_vessel_volume(file_path)
    gaussian_set, features = vessel_volume_to_gaussians(
        vessel,
        threshold=threshold,
        max_points=max_points,
    )

    os.makedirs(STATIC_DIR, exist_ok=True)
    out_name = f"{uuid.uuid4().hex}.gsp"
    out_path = os.path.join(STATIC_DIR, out_name)

    write_gsp(
        gaussian_set.positions,
        gaussian_set.normals,
        gaussian_set.colors,
        gaussian_set.sizes,
        out_path,
    )
    quick_mesh = export_quick_mesh_from_gaussians(gaussian_set)

    return {
        "gsp_url": f"/static/{out_name}",
        "count": int(len(gaussian_set.positions)),
        "generator": "vessel_centerline_proxy",
        "metadata": gaussian_set.metadata or {},
        "structural_features": features.as_dict(),
        **quick_mesh,
    }
