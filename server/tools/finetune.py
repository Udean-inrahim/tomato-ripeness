"""Fine-tune YOLOv8n pada data asli yang dilabeli dari labeling tool.

Struktur:
  server/labeling/images/*.jpg        (foto asli)
  server/labeling/labels/*.txt        (label YOLO dari tool labeling)

Dataset digabung: foto asli (train + val) + dataset sintetis yang sudah ada
di server/dataset (sebagai augmentasi dasar). data.yaml menunjuk ke dataset
yang terisi keduanya.

Run:
  venv python tools/finetune.py [epochs] [imgsz]
Contoh:
  venv python tools/finetune.py 60 320
Hasil:
  server/dataset/runs/finetune/weights/best.pt -> server/models/best.pt
"""

import shutil
import sys
from pathlib import Path

import numpy as np

if not hasattr(np, "trapz"):  # NumPy 2.x removed np.trapz
    np.trapz = np.trapezoid

from ultralytics import YOLO

SERVER = Path(__file__).resolve().parent.parent
LABELING = SERVER / "labeling"
IMAGES_DIR = LABELING / "images"
LABELS_DIR = LABELING / "labels"
DATASET = SERVER / "dataset"
REAL_IMG = DATASET / "images" / "real"
REAL_LBL = DATASET / "labels" / "real"
SYNTH_IMG = DATASET / "images"
SYNTH_LBL = DATASET / "labels"


def copy_real_splits():
    """Salin foto asli ke train+val untuk augmentasi (flip/mirror)."""
    imgs = sorted(p for p in IMAGES_DIR.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    print(f"Foto asli ditemukan: {len(imgs)}")
    if not imgs:
        print("TIDAK ADA foto asli di labeling/images. Berhenti.")
        sys.exit(1)

    REAL_IMG.mkdir(parents=True, exist_ok=True)
    REAL_LBL.mkdir(parents=True, exist_ok=True)

    # gunakan semua foto untuk train, foto pertama juga untuk val
    n_train = 0
    for img in imgs:
        stem = img.stem
        lbl = LABELS_DIR / (stem + ".txt")
        if not lbl.exists() or lbl.stat().st_size == 0:
            print(f"  SKIP (belum dilabeli): {img.name}")
            continue
        shutil.copy(img, REAL_IMG / f"{n_train:04d}{img.suffix.lower()}")
        shutil.copy(lbl, REAL_LBL / f"{n_train:04d}.txt")
        # mirror untuk variasi
        shutil.copy(img, REAL_IMG / f"{n_train:04d}_mirror{img.suffix.lower()}")
        shutil.copy(lbl, REAL_LBL / f"{n_train:04d}_mirror.txt")
        n_train += 1

    # val: satu foto pertama (tanpa mirror) sebagai evaluasi
    VAL_IMG = DATASET / "images" / "val"
    VAL_LBL = DATASET / "labels" / "val"
    VAL_IMG.mkdir(parents=True, exist_ok=True)
    VAL_LBL.mkdir(parents=True, exist_ok=True)
    if n_train >= 2:
        for idx in (0, 1):
            src_img = REAL_IMG / f"{idx:04d}.jpg"
            if src_img.exists():
                shutil.copy(src_img, VAL_IMG / f"real_val_{idx}.jpg")
                shutil.copy(REAL_LBL / f"{idx:04d}.txt", VAL_LBL / f"real_val_{idx}.txt")
    return n_train


def make_data_yaml(n_real):
    """Gabungkan foto asli + sintetis yang sudah ada di dataset/images."""
    data_yaml = DATASET / "data.yaml"
    data_yaml.write_text(
        f"""
path: {DATASET.as_posix()}
train: images
val: images/val
nc: 3
names: [mentah, setengah_matang, matang]
""".strip()
        + "\n"
    )
    print(f"data.yaml menunjuk train = seluruh images/ ({n_real} foto asli + sintetis)")


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 320

    n_real = copy_real_splits()
    make_data_yaml(n_real)

    # fine-tune dari model sintetis terbaik (cepat) atau pretrained COCO
    start = DATASET / "runs" / "yolov8n" / "weights" / "best.pt"
    if not start.exists():
        start = SERVER / "models" / "best.pt"
    weight = str(start) if start.exists() else "yolov8n.pt"
    print(f"Start weight: {weight}")

    model = YOLO(weight)
    model.train(
        data=str(DATASET / "data.yaml"),
        epochs=epochs,
        imgsz=imgsz,
        batch=8,
        device="cpu",
        workers=0,
        project=str(DATASET / "runs"),
        name="finetune",
        exist_ok=True,
        patience=20,
        verbose=True,
    )
    print("FINETUNE DONE")


if __name__ == "__main__":
    main()