from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # 1. Import the middleware
from .api.upload import router as upload_router

app = FastAPI(title="GSP Render Engine")

# 2. Add the CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"], # Allows your React dev server
    allow_credentials=True,
    allow_methods=["*"], # Allows POST, GET, etc.
    allow_headers=["*"], # Allows all headers
)

app.include_router(upload_router)