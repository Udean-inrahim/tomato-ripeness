"""Generate synthetic tomato dataset (YOLO format) for quick training.

Produces: dataset/images/{train,val}, dataset/labels/{train,val},
dataset/data.yaml with classes 0=mentah 1=setengah_matang 2=matang.
"""

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent / "dataset"
IMG_SIZE = 320
N_TRAIN = 400
N_VAL = 80

CLASSES = ["mentah", "setengah_matang", "matang"]

# Warna dasar (RGB) tomato: (dasar, highlight offset)
COLORS = {  # class index -> (center color, edge color)
    0: ((96, 168, 70), (70, 120, 46)),      # mentah: hijau
    1: ((235, 150, 45), (160, 100, 30)),    # setengah_matng: oranye
    2: ((225, 45, 38), (150, 25, 20)),      # matang: merah
}


def random_leaf_background(size):
    """Ciptakan background seperti semak daun hijau (blob hijau)."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    rng = np.random.default_rng()
    base = rng.integers(55, 150)
    arr[:] = (base, base + rng.integers(10, 60), rng.integers(20, 60))
    n = rng.integers(180, 360)
    for _ in range(n):
        x = rng.uniform(0, size)
        y = rng.uniform(0, size)
        r = rng.uniform(6, 42)
        g = int(np.clip(base + rng.uniform(0, 45), 0, 255))
        rr = int(np.clip(base - rng.uniform(0, 30), 0, 255))
        img = Image.fromarray(arr)
        img = img.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        d.ellipse((x - r, y - r, x + r, y + r), fill=(g, g, rr, 140))
        img.alpha_composite(overlay)
        arr = np.asarray(img.convert("RGB"))
    return Image.fromarray(arr)


def draw_tomato(img, cx, cy, radius, class_idx, rx=None, ry=None):
    """Gambar tomat bulat (kadang elips) dengan gradien radial + kilau."""
    d = ImageDraw.Draw(img, "RGBA")
    center, edge = COLORS[class_idx]
    rx = rx or radius
    ry = ry or radius
    steps = 10
    for i in range(steps):
        t = i / steps
        r = radius * (1 - t)
        color = tuple(
            int(center[c] + (edge[c] - center[c]) * t) for c in range(3)
        )
        d.ellipse(
            (cx - rx * (1 - t * 0.9), cy - ry * (1 - t * 0.9),
             cx + rx * (1 - t * 0.9), cy + ry * (1 - t * 0.9)),
            fill=(*color, 255),
        )
    # kilau kecil
    shine = int(radius * 0.32)
    sx, sy = cx - rx * 0.35, cy - ry * 0.35
    d.ellipse(
        (sx - shine, sy - shine, sx + shine, sy + shine),
        fill=(255, 255, 255, 70),
    )
    return img


def generate_one(rng, size, allow_negative=True):
    img = random_leaf_background(size)
    d_extra = ImageDraw.Draw(img)
    # ranting/garis hijau
    for _ in range(rng.integers(2, 6)):
        x = rng.uniform(0, size)
        d_extra.line(
            (x, 0, x + rng.uniform(-40, 40), size),
            fill=(70, 100, 40),
            width=rng.integers(2, 6),
        )

    # distractor: bentuk daun memanjang (bukan tomat)
    for _ in range(rng.integers(1, 5)):
        cx = rng.uniform(0, size)
        cy = rng.uniform(0, size)
        rx = rng.uniform(8, 18)
        ry = rng.uniform(28, 70)
        angle = rng.uniform(0, 180)
        if rng.random() < 0.5:
            rx, ry = ry, rx
        if angle > 45 and angle < 135:
            rx, ry = ry, rx
        d_extra.ellipse(
            (cx - rx, cy - ry, cx + rx, cy + ry),
            fill=(rng.integers(55, 125), rng.integers(130, 180),
                  rng.integers(40, 70)),
        )

    # 20% gambar tanpa tomat sama sekali (negatif murni)
    if allow_negative and rng.random() < 0.20:
        return img, []

    n_tomatoes = rng.integers(1, 5)
    boxes = []
    for _ in range(n_tomatoes):
        radius = rng.uniform(16, 62)
        stretch = rng.uniform(0.8, 1.25)
        cx = rng.uniform(radius * stretch + 8, size - radius * stretch - 8)
        cy = rng.uniform(radius + 8, size - radius - 8)
        class_idx = rng.integers(0, 3)
        draw_tomato(img, cx, cy, radius, class_idx,
                    rx=radius * stretch, ry=radius * (1 / stretch))

        x_center = cx / size
        y_center = cy / size
        w = (radius * stretch * 1.9) / size
        h = (radius * (1 / stretch) * 1.9) / size
        boxes.append((class_idx, x_center, y_center, w, h))
    return img, boxes


def make_split(kind, n, size):
    rng = np.random.default_rng(seed=42 if kind == "train" else 7)
    img_dir = ROOT / "images" / kind
    lbl_dir = ROOT / "labels" / kind
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        img, boxes = generate_one(rng, size)
        img.save(img_dir / f"{kind}_{i:04d}.jpg", quality=92)
        with open(lbl_dir / f"{kind}_{i:04d}.txt", "w") as f:
            for class_idx, xc, yc, w, h in boxes:
                f.write(f"{class_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    print(f"[{kind}] {n} gambar -> {img_dir}")


def main():
    make_split("train", N_TRAIN, IMG_SIZE)
    make_split("val", N_VAL, IMG_SIZE)
    data_yaml = ROOT / "data.yaml"
    data_yaml.write_text(
        f"""
path: {ROOT.as_posix()}
train: images/train
val: images/val
nc: 3
names: [mentah, setengah_matang, matang]
""".strip()
        + "\n"
    )
    print(f"data.yaml -> {data_yaml}")


if __name__ == "__main__":
    main()