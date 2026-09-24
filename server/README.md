# Server Deteksi Tomat — FastAPI + YOLOv8

Backend inference untuk aplikasi Flutter `tomato_ripeness`. Menerima gambar via `multipart/form-data`, menjalankan model **YOLOv8** (ultralytics), lalu mengembalikan bounding box + label dalam format JSON yang sudah disepakati.

## Instalasi

Disarankan pakai virtual environment:

```bash
cd server
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## Menjalankan

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Server siap di `http://localhost:8000`.

### Model

1. Siapkan file model YOLOv8 hasil training (format `.pt`, misal `best.pt`) dan letakkan di `server/models/`.
2. Tanpa file model, server otomatis memakai **fallback segmentasi warna** (sama dengan demo di Flutter) supaya alur integrasi tetap bisa diuji.

Ganti model:

```bash
set MODEL_PATH=models/best.pt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Label yang dikenali: `mentah`, `setengah_matang`, `matang`.

## Endpoint

`POST /detect`

- `multipart/form-data`, field `image` = file gambar (jpeg/png).
- `GET /health` — cek status server.

### Contoh respons

```json
{
  "image_width": 640,
  "image_height": 640,
  "detections": [
    { "label": "mentah", "confidence": 0.82, "x": 0.1, "y": 0.2, "w": 0.25, "h": 0.3 }
  ]
}
```

`x, y, w, h` dinormalisasi 0–1 terhadap ukuran gambar asli.

## Melatih model sendiri (ringkas)

1. Kumpulkan dataset foto tomat dengan label YOLO (kotak + `mentah`/`setengah_matang`/`matang`).
2. Di Google Colab: `pip install ultralytics`, taruh dataset dalam format YOLO lalu:

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="dataset.yaml", epochs=100, imgsz=640)
```

3. Ambil `runs/detect/train/weights/best.pt` ke `server/models/`.