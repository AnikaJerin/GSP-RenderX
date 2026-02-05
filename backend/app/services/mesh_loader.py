import trimesh


def _convex_hull_from_points(points):
    """
    Robust convex hull fallback that works across trimesh versions.
    """
    # PointCloud.convex_hull is widely supported.
    try:
        hull = trimesh.PointCloud(points).convex_hull
        if isinstance(hull, trimesh.Trimesh) and not hull.is_empty:
            return hull
    except Exception:
        pass

    # Final fallback: build a Trimesh and ask for its convex hull.
    hull = trimesh.Trimesh(vertices=points, process=False).convex_hull
    return hull

def load_mesh(file_path: str) -> trimesh.Trimesh:
    """
    Load a mesh from file, handling both meshes and point clouds.
    If the file is a point cloud (vertices only), convert it to a mesh.
    """
    # First, try loading as a mesh
    try:
        mesh = trimesh.load(file_path, force="mesh")
        
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
            )
        
        # If mesh has faces, use it directly
        if not mesh.is_empty and hasattr(mesh, "faces") and len(mesh.faces) > 0:
            if not mesh.is_watertight:
                print("⚠ Mesh not watertight — continuing anyway")
            mesh.fill_holes()
            mesh.process(validate=True)
            return mesh
    except Exception as e:
        print(f"⚠ Failed to load as mesh: {e}, trying as point cloud...")
    
    # If we get here, try loading as a point cloud
    try:
        # Load without forcing mesh type to see what we get
        loaded = trimesh.load(file_path)
        
        # Check if it's a PointCloud
        if isinstance(loaded, trimesh.PointCloud):
            print("⚠ Detected point cloud, converting to mesh...")
            points = loaded.vertices
            
            if len(points) < 4:
                raise ValueError(f"Point cloud has too few points ({len(points)}), need at least 4")
            
            # Try multiple methods to convert point cloud to mesh
            mesh = None
            
            # Method 1: Try ball pivoting (best quality, but may fail)
            if hasattr(trimesh.creation, 'ball_pivoting'):
                try:
                    print("  → Trying ball pivoting algorithm...")
                    mesh = trimesh.creation.ball_pivoting(
                        points=points,
                        radii=[0.005, 0.01, 0.02, 0.05]
                    )
                    if not mesh.is_empty:
                        print("  ✓ Ball pivoting succeeded")
                except Exception as bp_error:
                    print(f"  ⚠ Ball pivoting failed: {bp_error}")
            
            # Method 2: Convex hull (always works, but may not preserve details)
            if mesh is None or mesh.is_empty:
                print("  → Using convex hull (fallback)...")
                mesh = _convex_hull_from_points(points)
            
            if mesh.is_empty:
                raise ValueError("Failed to convert point cloud to mesh")
            
            mesh.process(validate=True)
            return mesh
        
        # If it's a Trimesh but empty or has no faces, try to extract vertices
        if isinstance(loaded, trimesh.Trimesh):
            if loaded.is_empty or (hasattr(loaded, "faces") and len(loaded.faces) == 0):
                print("⚠ Mesh has no faces, treating as point cloud...")
                points = loaded.vertices
                if len(points) == 0:
                    raise ValueError("Mesh has no vertices")
                
                if len(points) < 4:
                    raise ValueError(f"Point cloud has too few points ({len(points)}), need at least 4")
                
                # Convert point cloud to mesh
                mesh = None
                if hasattr(trimesh.creation, 'ball_pivoting'):
                    try:
                        print("  → Trying ball pivoting algorithm...")
                        mesh = trimesh.creation.ball_pivoting(
                            points=points,
                            radii=[0.005, 0.01, 0.02, 0.05]
                        )
                        if not mesh.is_empty:
                            print("  ✓ Ball pivoting succeeded")
                    except Exception:
                        pass
                
                if mesh is None or mesh.is_empty:
                    print("  → Using convex hull (fallback)...")
                    mesh = _convex_hull_from_points(points)
                
                mesh.process(validate=True)
                return mesh
            
            # If it has faces, use it
            if not loaded.is_watertight:
                print("⚠ Mesh not watertight — continuing anyway")
            loaded.fill_holes()
            loaded.process(validate=True)
            return loaded
            
    except Exception as e:
        raise ValueError(f"Failed to load mesh or convert point cloud: {e}")
    
    # Final fallback: if we still don't have a valid mesh
    raise ValueError("Mesh is empty or invalid, and could not be converted from point cloud")
