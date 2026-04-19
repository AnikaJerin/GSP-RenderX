from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

from ..models.gaussian import GaussianPrimitiveSet
from .model_state import get_learned_mesh_model_state
from .mesh_loader import load_mesh
from .mesh_sampler import sample_mesh_surface
from .structural_features import extract_mesh_structural_features
from .vessel_centerline import vessel_volume_to_gaussians
from .vessel_loader import load_vessel_volume


@dataclass
class GeneratorContext:
    source_path: str
    samples: int
    edge_angle: float
    edge_oversample: float


class GaussianGenerator(ABC):
    name: str
    description: str

    @abstractmethod
    def generate(self, context: GeneratorContext) -> GaussianPrimitiveSet:
        raise NotImplementedError


class HeuristicMeshGenerator(GaussianGenerator):
    name = "heuristic_mesh"
    description = "Current edge-aware mesh sampler used by the existing pipeline."

    def generate(self, context: GeneratorContext) -> GaussianPrimitiveSet:
        mesh = load_mesh(context.source_path)
        positions, normals, colors, sizes = sample_mesh_surface(
            mesh,
            context.samples,
            edge_angle_threshold=context.edge_angle,
            edge_oversample=context.edge_oversample,
        )
        return GaussianPrimitiveSet(
            positions=positions,
            normals=normals,
            colors=colors,
            sizes=sizes,
            metadata={
                "generator": self.name,
                "domain": "mesh",
                "status": "ready",
            },
        )


class LearnedMeshGenerator(GaussianGenerator):
    name = "learned_mesh"
    description = (
        "Research scaffold for a learned geometry-aware mesh Gaussian allocator. "
        "Currently extracts structural features and falls back to the heuristic sampler."
    )

    def generate(self, context: GeneratorContext) -> GaussianPrimitiveSet:
        mesh = load_mesh(context.source_path)
        structural_features = extract_mesh_structural_features(
            mesh, edge_angle_threshold=context.edge_angle
        )
        model_state = get_learned_mesh_model_state()
        positions, normals, colors, sizes = sample_mesh_surface(
            mesh,
            context.samples,
            edge_angle_threshold=context.edge_angle,
            edge_oversample=context.edge_oversample,
        )

        return GaussianPrimitiveSet(
            positions=positions,
            normals=normals,
            colors=colors,
            sizes=sizes,
            metadata={
                "generator": self.name,
                "domain": "mesh",
                "status": "fallback_ready",
                "inference_mode": "heuristic_fallback",
                "training_required": True,
                "model_state": model_state,
                "target_problem": (
                    "Learn structural Gaussian allocation for edge and thin-structure fidelity "
                    "under fixed storage/rendering budgets."
                ),
                "structural_features": structural_features.as_dict(),
            },
        )


class VesselVolumeGenerator(GaussianGenerator):
    name = "vessel_volume"
    description = (
        "Placeholder for a topology-aware, uncertainty-aware vessel-to-Gaussian pipeline "
        "centered on centerline recovery."
    )

    def generate(self, context: GeneratorContext) -> GaussianPrimitiveSet:
        vessel = load_vessel_volume(context.source_path)
        gaussian_set, _ = vessel_volume_to_gaussians(
            vessel,
            threshold=0.5,
            max_points=context.samples,
        )
        return gaussian_set


_GENERATORS: Dict[str, GaussianGenerator] = {
    generator.name: generator
    for generator in (
        HeuristicMeshGenerator(),
        LearnedMeshGenerator(),
        VesselVolumeGenerator(),
    )
}


def get_generator(name: str) -> GaussianGenerator:
    try:
        return _GENERATORS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_GENERATORS))
        raise ValueError(f"Unknown generator '{name}'. Available generators: {available}") from exc


def list_generators() -> Dict[str, Dict[str, str]]:
    return {
        name: {
            "description": generator.description,
        }
        for name, generator in sorted(_GENERATORS.items())
    }
