from fastapi import APIRouter, UploadFile, File
import shutil
import os
from ..services.pipeline import convert_mesh_to_gaussians

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/convert")
async def convert_model(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    gaussian_data = convert_mesh_to_gaussians(file_path, samples=60000)

    return gaussian_data
