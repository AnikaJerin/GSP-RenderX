import struct
import numpy as np


def write_gsp(positions, normals, colors, sizes, filepath):
    count = len(positions)

    positions = np.asarray(positions, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    colors = (np.asarray(colors) * 255).astype(np.uint8)
    sizes = (np.asarray(sizes) * 1000).astype(np.uint16)  # quantized

    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)

    with open(filepath, "wb") as f:
        # Header
        f.write(b"GSP1")
        f.write(struct.pack("I", count))
        f.write(struct.pack("3f", *bbox_min))
        f.write(struct.pack("3f", *bbox_max))

        # Data blocks
        f.write(positions.tobytes())
        f.write(normals.tobytes())
        f.write(colors.tobytes())
        f.write(sizes.tobytes())
