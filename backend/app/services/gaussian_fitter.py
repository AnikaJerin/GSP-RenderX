import numpy as np

def build_gaussians(positions, normals, colors, sizes):
    return {
        "positions": positions.tolist(),
        "normals": normals.tolist(),
        "colors": colors.tolist(),
        "sizes": sizes.tolist()
    }

