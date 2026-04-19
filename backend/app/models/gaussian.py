from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class GaussianPrimitiveSet:
    """
    Canonical in-memory Gaussian representation used by all generators.
    """

    positions: np.ndarray
    normals: np.ndarray
    colors: np.ndarray
    sizes: np.ndarray
    metadata: Dict[str, object] | None = None

    def as_json_dict(self) -> Dict[str, List]:
        return {
            "positions": self.positions.tolist(),
            "normals": self.normals.tolist(),
            "colors": self.colors.tolist(),
            "sizes": self.sizes.tolist(),
            "metadata": self.metadata or {},
        }
