# Server Deteksi Tomat — FastAPI + YOLOv8

Backend inference untuk aplikasi Flutter `tomato_ripeness`. Menerima gambar sebagai raw body `application/octet-stream`, menjalankan model **YOLOv8** (ultralytics), lalu mengembalikan bounding box + label dalam format JSON yang sudah disepakati.

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

Model YOLOv8 siap pakai disimpan di `server/models/best.pt` (6.2MB, YOLOv8n, 3 kelas: `mentah`, `setengah_matang`, `matang`). Model saat ini dilatih pada dataset sintetis (lingkaran tomat di atas background daun) — menggantikan fallback warna.

Tanpa file model, server otomatis memakai **fallback segmentasi warna** supaya alur integrasi selalu bisa diuji.

Ganti model:

```bash
set MODEL_PATH=models/best.pt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Label yang dikenali: `mentah`, `setengah_matang`, `matang`.

## Endpoint

`POST /detect`

- Body request: raw bytes dengan `Content-Type: application/octet-stream` (maksimal 10 MB).
- Gambar otomatis dirotasi mengikuti EXIF orientation.
- `GET /health` — cek status server, model, dan batas ukuran upload.

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

### Latihan dari dataset sintetis (termasuk di repo)

```bash
cd server
.venv\Scripts\python.exe tools\generate_dataset.py   # buat dataset sintetis
.venv\Scripts\python.exe tools\train.py 40 320       # train YOLOv8n (CPU ~37 menit)
Copy-Item dataset\runs\yolov8n\weights\best.pt models\best.pt
```

Untuk foto asli, letakkan foto dan label YOLO di `labeling/images` dan
`labeling/labels`, lalu jalankan:

```bash
.venv\Scripts\python.exe tools\finetune.py 24 320
Copy-Item dataset\runs\finetune\weights\best.pt models\best.pt
```

Label kosong pada foto non-tomat diperlakukan sebagai negative sample.