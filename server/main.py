import asyncio
import io
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError

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
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", 10 * 1024 * 1024))

detector = TomatoDetector(MODEL_PATH)
detector.load()
_detect_lock = asyncio.Lock()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "detector": detector.mode,
        "model": detector.model_name,
        "model_loaded": detector.model is not None,
        "max_image_bytes": MAX_IMAGE_BYTES,
    }


async def _read_image_bytes(request: Request) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_IMAGE_BYTES:
                raise HTTPException(status_code=413, detail="Ukuran gambar melebihi batas 10 MB")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Header Content-Length tidak valid") from exc

    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Ukuran gambar melebihi batas 10 MB")

    if not body:
        raise HTTPException(status_code=400, detail="Body gambar kosong")
    return bytes(body)


@app.post("/detect")
async def detect(request: Request):
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type and content_type != "application/octet-stream" and not content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Gunakan Content-Type image/* atau application/octet-stream")

    raw = await _read_image_bytes(request)
    try:
        with Image.open(io.BytesIO(raw)) as source:
            pil_image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Format gambar tidak valid atau rusak") from exc

    try:
        async with _detect_lock:
            result = await asyncio.to_thread(detector.detect, pil_image)
    finally:
        pil_image.close()

    return {
        "image_width": result["image_width"],
        "image_height": result["image_height"],
        "detections": result["detections"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
