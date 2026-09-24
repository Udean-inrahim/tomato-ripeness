import io
import os

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from detector import TomatoDetector

app = FastAPI(title="Deteksi Kematangan Tomat API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.pt")

detector = TomatoDetector(MODEL_PATH)
detector.load()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "detector": detector.mode,
        "model": detector.model_name,
    }


@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    raw = await image.read()
    pil_image = Image.open(io.BytesIO(raw)).convert("RGB")

    result = detector.detect(pil_image)

    return {
        "image_width": result["image_width"],
        "image_height": result["image_height"],
        "detections": result["detections"],
    }