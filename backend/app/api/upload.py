from fastapi import APIRouter, UploadFile, File, Query, HTTPException
import shutil
import os
import uuid
import zipfile
from ..services.pipeline import convert_mesh_to_gaussians

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

SUPPORTED_EXTS = {".stl", ".obj", ".ply", ".glb", ".gltf"}


def _find_mesh_file(root_dir: str):
    for root, _, files in os.walk(root_dir):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in SUPPORTED_EXTS:
                return os.path.join(root, name)
    return None


@router.post("/convert")
async def convert_model(
    file: UploadFile = File(...),
    target_splats: int = Query(500000, ge=50000, le=2000000),
    edge_angle: float = Query(35.0, ge=5.0, le=80.0),
    edge_oversample: float = Query(1.5, ge=0.5, le=4.0),
):
    upload_id = uuid.uuid4().hex
    original_name = file.filename or "upload"
    ext = os.path.splitext(original_name)[1].lower()

    work_dir = os.path.join(UPLOAD_DIR, upload_id)
    os.makedirs(work_dir, exist_ok=True)

    file_path = os.path.join(work_dir, original_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if ext == ".zip":
        with zipfile.ZipFile(file_path, "r") as zf:
            zf.extractall(work_dir)
        mesh_path = _find_mesh_file(work_dir)
        if not mesh_path:
            raise HTTPException(status_code=400, detail="No supported mesh found in zip")
    else:
        if ext not in SUPPORTED_EXTS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
        mesh_path = file_path

    mesh_ext = os.path.splitext(mesh_path)[1].lower()
    mesh_out = f"{uuid.uuid4().hex}{mesh_ext}"
    mesh_out_path = os.path.join(STATIC_DIR, mesh_out)
    shutil.copyfile(mesh_path, mesh_out_path)

    gaussian_data = convert_mesh_to_gaussians(
        mesh_path,
        samples=target_splats,
        edge_angle=edge_angle,
        edge_oversample=edge_oversample,
    )

    return {
        **gaussian_data,
        "mesh_url": f"/static/{mesh_out}",
        "mesh_type": mesh_ext.replace(".", ""),
    }
