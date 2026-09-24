"""Labeling tool untuk tomat (FastAPI).

Run:
  cd server/labeling
  venv python server.py          # port 8088
Lalu buka http://localhost:8088

Fitur:
  - Browser gambar dalam folder images/
  - Klik-drag untuk menggambar kotak, pilih kelas mentah/setengah_matang/matang
  - Simpan label format YOLO ke labels/<nama>.txt
Load model detector untuk bantuan? Gunakan tools/auto_label.py terpisah.
"""

import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).resolve().parent
IMAGES = BASE / "images"
LABELS = BASE / "labels"
IMAGES.mkdir(exist_ok=True)
LABELS.mkdir(exist_ok=True)

CLASSES = ["mentah", "setengah_matang", "matang"]

app = FastAPI(title="Tomato Labeler")
app.mount("/images", StaticFiles(directory=str(IMAGES)), name="images")
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    html = (BASE / "static" / "index.html").read_text("utf-8")
    return html.replace("__CLASSES__", json.dumps(CLASSES))


@app.get("/api/images")
def api_images():
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    files = sorted(
        p.name for p in IMAGES.iterdir() if p.suffix.lower() in exts
    )
    return {"images": files, "classes": CLASSES}


@app.get("/api/labels")
def api_labels(name: str):
    """Muat label (indeks kelas + koordinat normalized x1y1x2y2)."""
    txt = LABELS / (Path(name).stem + ".txt")
    boxes = []
    if txt.exists():
        for line in txt.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                cls = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])
                boxes.append({
                    "cls": cls,
                    "x1": round(xc - w / 2, 6),
                    "y1": round(yc - h / 2, 6),
                    "x2": round(xc + w / 2, 6),
                    "y2": round(yc + h / 2, 6),
                })
    return {"boxes": boxes}


@app.post("/api/labels")
async def api_save(request: Request):
    body = await request.json()
    name = body.get("name", "")
    boxes = body.get("boxes", [])
    txt = LABELS / (Path(name).stem + ".txt")
    lines = []
    for b in boxes:
        x1, y1 = min(b["x1"], b["x2"]), min(b["y1"], b["y2"])
        x2, y2 = max(b["x1"], b["x2"]), max(b["y1"], b["y2"])
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue
        lines.append(f"{int(b['cls'])} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    txt.write_text("\n".join(lines) + ("\n" if lines else ""))
    return {"ok": True, "count": len(lines)}


@app.post("/api/status")
async def api_status():
    n = sum(1 for p in LABELS.glob("*.txt") if p.stat().st_size > 0)
    total = sum(1 for p in IMAGES.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    return {"labeled": n, "images": total}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8088)