from fastapi import APIRouter, UploadFile, File, Query, HTTPException
import shutil
import os
import uuid
import zipfile
import httpx
from ..services.pipeline import convert_mesh_to_gaussians

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

SUPPORTED_EXTS = {".stl", ".obj", ".ply", ".glb", ".gltf"}

# Remote reconstruction API endpoints (e.g. Colab with FastAvatar / Pixel3DMM).
# Configure these via environment variables when you launch the backend.
FASTAVATAR_API = os.getenv("FASTAVATAR_API")  # e.g. https://<ngrok>.ngrok.io
PIXEL3DMM_API = os.getenv("PIXEL3DMM_API")    # e.g. https://<ngrok>.ngrok.io


def _find_mesh_file(root_dir: str):
    for root, _, files in os.walk(root_dir):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in SUPPORTED_EXTS:
                return os.path.join(root, name)
    return None


async def _fetch_remote_mesh(image_path: str, mode: str) -> str:
  """
  Send the image to the appropriate remote reconstruction API and save
  the returned mesh under STATIC_DIR. Returns the local mesh file path.
  This assumes the remote API returns raw mesh bytes with an optional
  X-Mesh-Format header (obj/ply/glb).
  """
  if mode == "fastavatar":
      if not FASTAVATAR_API:
          raise HTTPException(status_code=500, detail="FASTAVATAR_API is not configured")
      url = f"{FASTAVATAR_API.rstrip('/')}/api/reconstruct/face"
  elif mode == "pixel3dmm":
      if not PIXEL3DMM_API:
          raise HTTPException(status_code=500, detail="PIXEL3DMM_API is not configured")
      url = f"{PIXEL3DMM_API.rstrip('/')}/api/reconstruct/medical"
  else:
      raise HTTPException(status_code=400, detail="Unknown reconstruction mode")

  with open(image_path, "rb") as f:
      files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
      async with httpx.AsyncClient(timeout=120.0) as client:
          resp = await client.post(url, files=files)

  if resp.status_code != 200:
      raise HTTPException(status_code=502, detail=f"Remote reconstruction failed: {resp.text}")

  mesh_format = resp.headers.get("X-Mesh-Format", "obj").lower()
  mesh_name = f"{uuid.uuid4().hex}.{mesh_format}"
  mesh_path = os.path.join(STATIC_DIR, mesh_name)
  with open(mesh_path, "wb") as out_f:
      out_f.write(resp.content)
  return mesh_path


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


@router.post("/reconstruct_image_fastavatar")
async def reconstruct_image_fastavatar(
    file: UploadFile = File(...),
    target_splats: int = Query(150000, ge=1000, le=800000),
    edge_angle: float = Query(35.0, ge=5.0, le=80.0),
    edge_oversample: float = Query(1.5, ge=0.5, le=4.0),
):
    """
    Single-image photorealistic / simple-face reconstruction using FastAvatar.
    The heavy model runs in a remote Colab API; this endpoint just:
      - saves the uploaded image
      - calls the FastAvatar API to get a mesh
      - converts that mesh to a GSP using the existing pipeline
    """
    upload_id = uuid.uuid4().hex
    original_name = file.filename or "image"
    ext = os.path.splitext(original_name)[1].lower()

    if ext not in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}:
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    work_dir = os.path.join(UPLOAD_DIR, f"img-{upload_id}")
    os.makedirs(work_dir, exist_ok=True)
    img_path = os.path.join(work_dir, original_name)
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    mesh_path = await _fetch_remote_mesh(img_path, mode="fastavatar")
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


@router.post("/reconstruct_image_pixel3dmm")
async def reconstruct_image_pixel3dmm(
    file: UploadFile = File(...),
    target_splats: int = Query(150000, ge=1000, le=800000),
    edge_angle: float = Query(35.0, ge=5.0, le=80.0),
    edge_oversample: float = Query(1.5, ge=0.5, le=4.0),
):
    """
    Single-image medical/scientific reconstruction using Pixel3DMM.
    Same flow as reconstruct_image_fastavatar but hits a different remote API.
    """
    upload_id = uuid.uuid4().hex
    original_name = file.filename or "image"
    ext = os.path.splitext(original_name)[1].lower()

    if ext not in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}:
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    work_dir = os.path.join(UPLOAD_DIR, f"img-{upload_id}")
    os.makedirs(work_dir, exist_ok=True)
    img_path = os.path.join(work_dir, original_name)
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    mesh_path = await _fetch_remote_mesh(img_path, mode="pixel3dmm")
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
