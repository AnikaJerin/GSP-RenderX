
import os
import sys
import numpy as np
import trimesh

# Add backend to sys.path
sys.path.append(os.path.abspath("backend"))

from app.services.face_gen import generate_face_mesh
from app.services.pipeline import convert_mesh_to_gaussians
from app.services.mesh_loader import load_mesh

def create_dummy_image(path):
    import cv2
    # Create a simple red/blue gradient image
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    for i in range(512):
        img[i, :, 0] = i // 2 # Blue
        img[:, i, 2] = i // 2 # Red
    cv2.imwrite(path, img)
    print(f"Created dummy image at {path}")

def debug_pipeline():
    mesh_out = "debug_synthetic.ply"
    
    print("--- 1. Creating Synthetic Colored Mesh ---")
    # Create a simple triangle
    vertices = np.array([[0,0,0], [1,0,0], [0,1,0]], dtype=np.float64)
    faces = np.array([[0,1,2]])
    # Red, Green, Blue
    colors = np.array([
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255]
    ], dtype=np.uint8)
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
    mesh.export(mesh_out, file_type='ply')
    print(f"Exported {mesh_out} with Colors.")

    print("--- 2. Inspecting Generated Mesh (Trimesh) ---")
    mesh_loaded = trimesh.load(mesh_out, force='mesh')
    print(f"Loaded Mesh Type: {type(mesh_loaded)}")
    
    if hasattr(mesh_loaded.visual, 'vertex_colors'):
        loaded_colors = mesh_loaded.visual.vertex_colors
        print(f"Loaded Colors Shape: {loaded_colors.shape}")
        print(f"Sample Color: {loaded_colors[0]}")
        
        if len(loaded_colors) > 0 and loaded_colors[0][0] > 200:
             print("✅ Colors preserved in PLY.")
        else:
             print("❌ Colors LOST in PLY.")
    else:
        print("❌ Mesh has no 'visual.vertex_colors' attribute.")

    print("--- 3. Testing Mesh Sampler Logic ---")
    from app.services.mesh_sampler import sample_mesh_surface
    try:
        p, n, c, s = sample_mesh_surface(mesh_loaded, n_points=10)
        print(f"Sampled Colors Shape: {c.shape}")
        
        # Check if sampled colors are consistent with input
        # Note: sample_mesh interpolates, but for a triangle with R,G,B corners, 
        # sampled points should be non-gray.
        # Fake lighting (normals) on a flat triangle would be constant uniform color (light blue-ish).
        # Real colors will vary.
        
        print(f"Sampled Color 0: {c[0]}")
        if np.std(c, axis=0).sum() < 0.01:
             print("⚠️ Sampled colors are uniform (Likely Fake Lighting/Normals).")
        else:
             print("✅ Sampled colors show variation (Likely Vertex Colors).")

    except Exception as e:
        print(f"Sampling Failed: {e}")

if __name__ == "__main__":
    debug_pipeline()
