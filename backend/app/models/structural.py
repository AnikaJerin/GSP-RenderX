from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple


@dataclass
class MeshStructuralFeatures:
    vertex_count: int
    face_count: int
    bbox_extent: Tuple[float, float, float]
    bbox_diagonal: float
    sharp_edge_count: int
    sharp_edge_ratio: float
    mean_edge_length: float
    thin_structure_score: float
    mean_adjacency_angle_deg: float

    def as_dict(self) -> Dict[str, float | int | Tuple[float, float, float]]:
        return asdict(self)

    def feature_vector(self) -> List[float]:
        return [
            float(self.vertex_count),
            float(self.face_count),
            float(self.bbox_extent[0]),
            float(self.bbox_extent[1]),
            float(self.bbox_extent[2]),
            float(self.bbox_diagonal),
            float(self.sharp_edge_count),
            float(self.sharp_edge_ratio),
            float(self.mean_edge_length),
            float(self.thin_structure_score),
            float(self.mean_adjacency_angle_deg),
        ]

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "vertex_count",
            "face_count",
            "bbox_extent_x",
            "bbox_extent_y",
            "bbox_extent_z",
            "bbox_diagonal",
            "sharp_edge_count",
            "sharp_edge_ratio",
            "mean_edge_length",
            "thin_structure_score",
            "mean_adjacency_angle_deg",
        ]


@dataclass
class VesselStructuralFeatures:
    voxel_count: int
    active_voxel_count: int
    vessel_density: float
    centerline_voxel_count: int
    branchpoint_count: int
    endpoint_count: int
    mean_radius: float
    max_radius: float
    approximate_total_length: float
    uncertainty_ready: bool

    def as_dict(self) -> Dict[str, float | int | bool]:
        return asdict(self)

    def feature_vector(self) -> List[float]:
        return [
            float(self.voxel_count),
            float(self.active_voxel_count),
            float(self.vessel_density),
            float(self.centerline_voxel_count),
            float(self.branchpoint_count),
            float(self.endpoint_count),
            float(self.mean_radius),
            float(self.max_radius),
            float(self.approximate_total_length),
            float(self.uncertainty_ready),
        ]

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "voxel_count",
            "active_voxel_count",
            "vessel_density",
            "centerline_voxel_count",
            "branchpoint_count",
            "endpoint_count",
            "mean_radius",
            "max_radius",
            "approximate_total_length",
            "uncertainty_ready",
        ]
