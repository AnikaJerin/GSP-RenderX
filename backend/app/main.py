from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # 1. Import the middleware
from .api.upload import router as upload_router
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI(title="GSP Render Engine")

# make sure directory exists
os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")


# 2. Add the CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # Allows your React dev server
    allow_credentials=True,
    allow_methods=["*"], # Allows POST, GET, etc.
    allow_headers=["*"], # Allows all headers
)

app.include_router(upload_router)