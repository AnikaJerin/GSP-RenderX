
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import trimesh
import os

def generate_face_mesh(image_path: str, output_path: str):
    # Determine path to model file (in same directory as this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'face_landmarker.task')

    if not os.path.exists(model_path):
         raise FileNotFoundError(f"face_landmarker.task not found at {model_path}")

    # 1. Setup Options
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1)
    
    detector = vision.FaceLandmarker.create_from_options(options)

    # 2. Load Image
    image = mp.Image.create_from_file(image_path)

    # 3. Detect Landmarks
    detection_result = detector.detect(image)
    
    if not detection_result.face_landmarks:
        raise ValueError("No face detected in the image.")

    # 4. Process Geometry
    # Get the first face detected
    face_landmarks = detection_result.face_landmarks[0]
    
    # MediaPipe Tasks provides normalized coordinates (0.0 to 1.0)
    # We scale them based on the image dimensions for a 1:1 pixel mesh
    try:
        h, w = image.height, image.width
    except:
        h, w = 1024, 1024

    vertices = np.array([
        [lm.x * w, (1 - lm.y) * h, lm.z * w] 
        for lm in face_landmarks
    ])

    # Center the mesh
    vertices -= np.mean(vertices, axis=0)

    # 4b. Color Sampling (Skin)
    # Map vertices back to UVs or just sample pixel at (lm.x, lm.y)
    vertex_colors = []
    
    # We need to access pixel data. mp.Image -> numpy
    img_np = image.numpy_view() # (H, W, 3) usually RGB
    
    for lm in face_landmarks:
        # lm.x, lm.y are normalized [0,1]
        px = int(min(max(lm.x * w_img, 0), w_img - 1))
        py = int(min(max(lm.y * h_img, 0), h_img - 1))
        
        # Sample color
        # img_np is RGB
        rgb = img_np[py, px]
        vertex_colors.append(rgb)
        
    vertex_colors = np.array(vertex_colors, dtype=np.uint8)
    
    # Add alpha 255
    ones = np.ones((len(vertex_colors), 1), dtype=np.uint8) * 255
    vertex_colors_rgba = np.hstack([vertex_colors, ones])

    # 5. Build Mesh (Delaunay)
    from scipy.spatial import Delaunay
    tri = Delaunay(vertices[:, :2])
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=tri.simplices, vertex_colors=vertex_colors_rgba)
    
    # Export as PLY (Better color support)
    mesh.export(output_path, file_type='ply')
    
    return output_path
