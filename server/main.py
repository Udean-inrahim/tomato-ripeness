import io
import os

from fastapi import FastAPI, Request
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

SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(SERVER_DIR, "models/best.pt"))

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
async def detect(request: Request):
    raw = await request.body()
    pil_image = Image.open(io.BytesIO(raw)).convert("RGB")

    result = detector.detect(pil_image)

    return {
        "image_width": result["image_width"],
        "image_height": result["image_height"],
        "detections": result["detections"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
