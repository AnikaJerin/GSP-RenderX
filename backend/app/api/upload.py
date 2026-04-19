from fastapi import APIRouter, UploadFile, File, Query, HTTPException
import shutil
import os
import uuid
import zipfile
import httpx
import numpy as np
from ..services.pipeline import convert_mesh_to_gaussians, convert_vessel_volume_to_gaussians
from ..services.generator_registry import list_generators
from ..services.mesh_loader import load_mesh
from ..services.model_state import (
    get_learned_mesh_manifest_template,
    get_learned_mesh_model_state,
    get_learned_vessel_manifest_template,
    get_learned_vessel_model_state,
)
from ..services.structural_features import extract_mesh_structural_features
from ..services.vessel_loader import load_vessel_volume
from ..services.vessel_centerline import extract_vessel_structural_features as extract_loaded_vessel_features

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

SUPPORTED_EXTS = {".stl", ".obj", ".ply", ".glb", ".gltf"}
SUPPORTED_VESSEL_EXTS = {".npy", ".npz"}

def _get_remote_api_url(mode: str) -> str:
    """
    Resolve remote reconstruction endpoint from the current process environment.
    Reading this at request time lets you restart less often while debugging config.
    """
    if mode == "fastavatar":
        base_url = os.getenv("FASTAVATAR_API", "").strip()
        if not base_url:
            raise HTTPException(
                status_code=503,
                detail="FASTAVATAR_API is not configured. Set it to your remote FastAvatar base URL.",
            )
        return f"{base_url.rstrip('/')}/api/reconstruct/face"

    if mode == "pixel3dmm":
        base_url = os.getenv("PIXEL3DMM_API", "").strip()
        if not base_url:
            raise HTTPException(
                status_code=503,
                detail="PIXEL3DMM_API is not configured. Set it to your remote Pixel3DMM base URL.",
            )
        return f"{base_url.rstrip('/')}/api/reconstruct/medical"

    raise HTTPException(status_code=400, detail="Unknown reconstruction mode")


def _find_mesh_file(root_dir: str):
    for root, _, files in os.walk(root_dir):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in SUPPORTED_EXTS:
                return os.path.join(root, name)
    return None


def _persist_upload_to_mesh(file: UploadFile, upload_id_prefix: str = ""):
    upload_id = f"{upload_id_prefix}{uuid.uuid4().hex}"
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

    return {
        "upload_id": upload_id,
        "original_name": original_name,
        "mesh_path": mesh_path,
    }


def _persist_upload_to_vessel_volume(file: UploadFile, upload_id_prefix: str = ""):
    upload_id = f"{upload_id_prefix}{uuid.uuid4().hex}"
    original_name = file.filename or "volume"
    ext = os.path.splitext(original_name)[1].lower()

    if ext not in SUPPORTED_VESSEL_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported vessel volume type: {ext}. Use .npy or .npz exported from Colab.",
        )

    work_dir = os.path.join(UPLOAD_DIR, upload_id)
    os.makedirs(work_dir, exist_ok=True)

    file_path = os.path.join(work_dir, original_name)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "upload_id": upload_id,
        "original_name": original_name,
        "volume_path": file_path,
    }


async def _fetch_remote_mesh(image_path: str, mode: str) -> str:
  """
  Send the image to the appropriate remote reconstruction API and save
  the returned mesh under STATIC_DIR. Returns the local mesh file path.
  This assumes the remote API returns raw mesh bytes with an optional
  X-Mesh-Format header (obj/ply/glb).
  """
  url = _get_remote_api_url(mode)

  with open(image_path, "rb") as f:
      files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
      try:
          async with httpx.AsyncClient(timeout=120.0) as client:
              resp = await client.post(url, files=files)
      except httpx.TimeoutException as exc:
          raise HTTPException(
              status_code=504,
              detail=f"Remote reconstruction timed out while calling {url}",
          ) from exc
      except httpx.RequestError as exc:
          raise HTTPException(
              status_code=502,
              detail=f"Remote reconstruction request failed for {url}: {exc}",
          ) from exc

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
    generator: str = Query("heuristic_mesh"),
):
    persisted = _persist_upload_to_mesh(file)
    mesh_path = persisted["mesh_path"]

    mesh_ext = os.path.splitext(mesh_path)[1].lower()
    mesh_out = f"{uuid.uuid4().hex}{mesh_ext}"
    mesh_out_path = os.path.join(STATIC_DIR, mesh_out)
    shutil.copyfile(mesh_path, mesh_out_path)

    try:
        gaussian_data = convert_mesh_to_gaussians(
            mesh_path,
            samples=target_splats,
            edge_angle=edge_angle,
            edge_oversample=edge_oversample,
            generator_name=generator,
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        **gaussian_data,
        "mesh_url": f"/static/{mesh_out}",
        "mesh_type": mesh_ext.replace(".", ""),
    }


@router.post("/analyze_mesh")
async def analyze_mesh(
    file: UploadFile = File(...),
    edge_angle: float = Query(35.0, ge=5.0, le=80.0),
):
    persisted = _persist_upload_to_mesh(file, upload_id_prefix="analysis-")
    mesh = load_mesh(persisted["mesh_path"])
    features = extract_mesh_structural_features(mesh, edge_angle_threshold=edge_angle)

    return {
        "source_name": persisted["original_name"],
        "problem_focus": "geometry",
        "research_target": "edge_and_thin_structure_fidelity_under_budget",
        "structural_features": features.as_dict(),
        "feature_vector_names": features.feature_names(),
        "feature_vector": features.feature_vector(),
        "learned_mesh_model_state": get_learned_mesh_model_state(),
    }


@router.post("/export_mesh_training_record")
async def export_mesh_training_record(
    file: UploadFile = File(...),
    edge_angle: float = Query(35.0, ge=5.0, le=80.0),
    target_splats: int = Query(500000, ge=50000, le=2000000),
):
    persisted = _persist_upload_to_mesh(file, upload_id_prefix="trainrec-")
    mesh = load_mesh(persisted["mesh_path"])
    features = extract_mesh_structural_features(mesh, edge_angle_threshold=edge_angle)

    return {
        "record_version": 1,
        "source_name": persisted["original_name"],
        "problem_focus": "geometry",
        "research_target": "edge_and_thin_structure_fidelity_under_budget",
        "feature_vector_names": features.feature_names(),
        "feature_vector": features.feature_vector(),
        "structural_features": features.as_dict(),
        "baseline_policy": {
            "generator": "heuristic_mesh",
            "target_splats": target_splats,
            "edge_angle": edge_angle,
            "edge_oversample": 1.5,
            "surface_size": 0.014,
            "edge_size": 0.0045,
        },
        "training_contract": get_learned_mesh_manifest_template(),
        "publishable_data_notes": {
            "recommended_mesh_domains": [
                "CAD parts with sharp edges",
                "thin-structure meshes",
                "public synthetic geometric meshes",
            ],
            "recommended_labels": [
                "quality under fixed splat budget",
                "edge fidelity score",
                "thin-structure preservation score",
            ],
        },
    }


@router.post("/analyze_vessel_volume")
async def analyze_vessel_volume(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
):
    persisted = _persist_upload_to_vessel_volume(file, upload_id_prefix="vessel-analysis-")
    vessel = load_vessel_volume(persisted["volume_path"])
    features, vessel_mask, centerline, _ = extract_loaded_vessel_features(
        vessel, threshold=threshold
    )

    return {
        "source_name": persisted["original_name"],
        "problem_focus": "topology_preserving_centerline_reconstruction",
        "research_target": "centerline_conditioned_gaussian_representation",
        "structural_features": features.as_dict(),
        "feature_vector_names": features.feature_names(),
        "feature_vector": features.feature_vector(),
        "volume_shape": list(vessel_mask.shape),
        "centerline_ratio": float(
            np.count_nonzero(centerline) / max(np.count_nonzero(vessel_mask), 1)
        ),
        "learned_vessel_model_state": get_learned_vessel_model_state(),
    }


@router.post("/convert_vessel_volume")
async def convert_vessel_volume(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    max_points: int = Query(120000, ge=1000, le=500000),
):
    persisted = _persist_upload_to_vessel_volume(file, upload_id_prefix="vessel-convert-")

    try:
        gaussian_data = convert_vessel_volume_to_gaussians(
            persisted["volume_path"],
            threshold=threshold,
            max_points=max_points,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        **gaussian_data,
        "metadata": {
            **(gaussian_data.get("metadata") or {}),
            "learned_vessel_model_state": get_learned_vessel_model_state(),
            "source_type": "vessel_volume",
        },
        "mesh_url": None,
        "mesh_type": None,
        "source_type": "vessel_volume",
    }


@router.get("/vessel_model_contract")
async def vessel_model_contract():
    return {
        "training_contract": get_learned_vessel_manifest_template(),
        "learned_vessel_model_state": get_learned_vessel_model_state(),
        "local_usage": {
            "supported_uploads": sorted(SUPPORTED_VESSEL_EXTS),
            "workflow": [
                "Train the vessel model in Colab on TOF-MRA or DSA data.",
                "Export .npz predictions containing vessel_mask or vessel_prob.",
                "Optionally include centerline_mask, radius_map, uncertainty_map, and spacing.",
                "Upload the exported file here to analyze topology and render centerline-aware GSP output.",
            ],
        },
    }


@router.get("/pipeline_capabilities")
async def pipeline_capabilities():
    return {
        "generators": list_generators(),
        "remote_apis": {
            "fastavatar_configured": bool(os.getenv("FASTAVATAR_API", "").strip()),
            "pixel3dmm_configured": bool(os.getenv("PIXEL3DMM_API", "").strip()),
        },
        "learned_mesh_model_state": get_learned_mesh_model_state(),
        "learned_vessel_model_state": get_learned_vessel_model_state(),
    }


@router.get("/research_status")
async def research_status():
    return {
        "focus": "structural_gaussian_learning",
        "problems": [
            "Replace heuristic Gaussian generation with a learned structural Gaussian representation that adapts to geometry and topology.",
            "Improve edge/thin-structure fidelity under fixed storage and rendering budgets.",
            "Recover distal vessels with explicit uncertainty and centerline-aware structure instead of hallucinated detail.",
        ],
        "implemented_now": {
            "heuristic_mesh_pipeline": True,
            "learned_mesh_fallback": True,
            "mesh_structural_feature_extraction": True,
            "vessel_centerline_generator": True,
            "uncertainty_aware_vessel_inference": False,
        },
        "next_targets": [
            "Train a lightweight mesh feature-to-Gaussian allocator.",
            "Train a Colab-based topology-aware vessel model that exports centerline-aware .npz bundles.",
            "Extend GSP metadata and viewer controls for uncertainty visualization.",
        ],
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
