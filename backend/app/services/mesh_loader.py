import trimesh

def load_mesh(file_path: str) -> trimesh.Trimesh:
    # force='mesh' ensures we get a Trimesh object
    mesh = trimesh.load(file_path, force="mesh")

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        )

    if mesh.is_empty:
        raise ValueError("Mesh is empty or invalid")

    if not mesh.is_watertight:
        print("⚠ Mesh not watertight — continuing anyway")

    # FIX: Use fill_holes and process instead of the removed method
    mesh.fill_holes()
    mesh.process(validate=True)

    return mesh
