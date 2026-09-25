"""Fine-tune YOLOv8n pada data asli yang dilabeli dari labeling tool.

Struktur:
  server/labeling/images/*.jpg        (foto asli)
  server/labeling/labels/*.txt        (label YOLO dari tool labeling)

Foto asli disalin ke dataset/images/train + labels/train, digabung dengan
dataset sintetis yang sudah ada. data.yaml menunjuk path absolut.

Run:
  venv python tools/finetune.py [epochs] [imgsz]
Contoh:
  venv python tools/finetune.py 60 320
Hasil:
  server/dataset/runs/finetune/weights/best.pt -> server/models/best.pt
"""

import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

if not hasattr(np, "trapz"):  # NumPy 2.x removed np.trapz
    np.trapz = np.trapezoid

from ultralytics import YOLO

SERVER = Path(__file__).resolve().parent.parent
LABELING = SERVER / "labeling"
IMAGES_DIR = LABELING / "images"
LABELS_DIR = LABELING / "labels"
DATASET = SERVER / "dataset"
TRAIN_IMG = DATASET / "images" / "train"
TRAIN_LBL = DATASET / "labels" / "train"
VAL_IMG = DATASET / "images" / "val"
VAL_LBL = DATASET / "labels" / "val"


def _flip_label_hx(path: Path, out: Path):
    """Tulis label mirror: koordinat x basis jadi 1-x."""
    lines = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            cls = parts[0]
            xc = float(parts[1])
            parts[1] = f"{1.0 - xc:.6f}"
            lines.append(" ".join([cls] + [f"{float(p):.6f}" for p in parts[1:]]))
    out.write_text("\n".join(lines) + ("\n" if lines else ""))


def copy_real_splits():
    """Salin foto asli + label (dan versi mirror) ke train; 1 foto ke val."""
    imgs = sorted(p for p in IMAGES_DIR.glob("*")
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    print(f"Foto asli ditemukan: {len(imgs)}")
    if not imgs:
        print("TIDAK ADA foto asli di labeling/images. Berhenti.")
        sys.exit(1)

    TRAIN_IMG.mkdir(parents=True, exist_ok=True)
    TRAIN_LBL.mkdir(parents=True, exist_ok=True)
    VAL_IMG.mkdir(parents=True, exist_ok=True)
    VAL_LBL.mkdir(parents=True, exist_ok=True)

    # bersihkan salinan real sebelumnya agar tidak menumpuk bolak-balik
    for f in TRAIN_IMG.glob("real_*"):
        f.unlink()
    for f in TRAIN_LBL.glob("real_*"):
        f.unlink()
    for f in VAL_IMG.glob("real_val_*"):
        f.unlink()
    for f in VAL_LBL.glob("real_val_*"):
        f.unlink()

    rng = random.Random(42)
    labeled = []
    for img in imgs:
        lbl = LABELS_DIR / (img.stem + ".txt")
        if not lbl.exists():
            print(f"  SKIP (belum dilabeli): {img.name}")
            continue
        labeled.append((img, lbl))

    if not labeled:
        print("TIDAK ADA foto yang dilabeli. Berhenti.")
        sys.exit(1)

    # Semua foto asli masuk train (val tetap sintetis) agar tiap foto
    # benar-benar dipelajari model.
    n_train = 0
    for img, lbl in labeled:
        ext = img.suffix.lower()

        shutil.copy(img, TRAIN_IMG / f"real_{n_train:01d}{ext}")
        shutil.copy(lbl, TRAIN_LBL / f"real_{n_train:01d}.txt")

        # mirror sungguhan: flip horizontal gambar + label x
        flip = PILImage.open(img).transpose(PILImage.FLIP_LEFT_RIGHT)
        flip.save(TRAIN_IMG / f"real_{n_train:01d}_mirror{ext}")
        _flip_label_hx(lbl, TRAIN_LBL / f"real_{n_train:01d}_mirror.txt")

        # oversample real 8x (kopi + mirror sudah 2, tambah 7 duplikat) agar
        # tidak tenggelam di antara data sintetis
        for k in range(1, 8):
            shutil.copy(img, TRAIN_IMG / f"real_{n_train:01d}_dup{k}{ext}")
            shutil.copy(lbl, TRAIN_LBL / f"real_{n_train:01d}_dup{k}.txt")
            shutil.copy(TRAIN_IMG / f"real_{n_train:01d}_mirror{ext}",
                        TRAIN_IMG / f"real_{n_train:01d}_dup{k}_mirror{ext}")
            _flip_label_hx(lbl, TRAIN_LBL / f"real_{n_train:01d}_dup{k}_mirror.txt")
        n_train += 1

    print(f"Train (asli): {n_train} foto x8 = {n_train*8}, Val: sintetis")
    return n_train


def make_data_yaml(n_real):
    data_yaml = DATASET / "data.yaml"
    data_yaml.write_text(
        f"""
path: {DATASET.as_posix()}
train: images/train
val: images/val
nc: 3
names: [mentah, setengah_matang, matang]
""".strip()
        + "\n"
    )
    print(f"data.yaml -> train: images/train ({n_real} pasang real), val: images/val")


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 320

    n_real = copy_real_splits()
    make_data_yaml(n_real)

    # gunakan model terbaru (fine-tune sebelumnya) bila tersedia
    start = DATASET / "runs" / "finetune" / "weights" / "best.pt"
    if not start.exists():
        start = SERVER / "models" / "best.pt"
    if not start.exists():
        start = DATASET / "runs" / "yolov8n" / "weights" / "best.pt"
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