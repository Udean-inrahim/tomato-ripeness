import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image

LABEL_MAP = {0: "mentah", 1: "setengah_matang", 2: "matang"}

_MENTAH = 1
_SETENGAH = 2
_MATANG = 3


@dataclass
class Detection:
    label: str
    confidence: float
    x: float
    y: float
    w: float
    h: float


class TomatoDetector:
    """YOLOv8 detector dengan fallback segmentasi warna.

    Memakai model YOLOv8 bila model-file tersedia. Jika tidak, memakai
    segmentasi warna sederhana agar alur integrasi tetap bisa diuji.
    """

    def __init__(self, model_path: str = "models/best.pt"):
        self.model_path = model_path
        self.model = None
        self.model_name = "none"
        self.mode = "fallback_warna"

    def load(self):
        if os.path.exists(self.model_path):
            try:
                from ultralytics import YOLO

                self.model = YOLO(self.model_path)
                self.model_name = os.path.basename(self.model_path)
                self.mode = "yolo"
            except Exception as exc:  # pragma: no cover
                print(f"Gagal memuat model YOLO ({exc}); memakai fallback warna.")
                self.model = None
                self.model_name = "none"
                self.mode = "fallback_warna"
        else:
            print(f"Model {self.model_path} belum ada; memakai fallback warna.")

    def detect(self, pil_image: Image.Image):
        width, height = pil_image.size
        if self.model is not None:
            dets = self._detect_yolo(pil_image)
        else:
            dets = self._detect_fallback(pil_image)

        return {
            "image_width": width,
            "image_height": height,
            "detections": [d.__dict__ for d in dets],
        }

    # ---- YOLO ----
    def _detect_yolo(self, pil_image: Image.Image) -> List[Detection]:
        results = self.model(pil_image, conf=0.25, verbose=False)[0]
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        boxes = results.boxes.xyxyn.cpu().numpy()

        dets: List[Detection] = []
        for box, cls, score in zip(boxes, classes, scores):
            label = LABEL_MAP.get(int(cls))
            if label is None:
                continue
            x1, y1, x2, y2 = box
            dets.append(
                Detection(
                    label=label,
                    confidence=round(float(score), 4),
                    x=round(float(x1), 4),
                    y=round(float(y1), 4),
                    w=round(float(x2 - x1), 4),
                    h=round(float(y2 - y1), 4),
                )
            )
        # sort: atas-ke-bawah
        dets.sort(key=lambda d: (d.y, d.x))
        return dets

    # ---- Fallback warna ----
    def _detect_fallback(self, pil_image: Image.Image) -> List[Detection]:
        rgb = np.asarray(pil_image.resize((256, 256))).astype(np.float32) / 255.0
        hsv = _rgb_to_hsv(rgb)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        h2 = h * 2
        labels = np.zeros((256, 256), dtype=np.int8)

        valid = (v >= 0.15) & (v <= 0.95) & (s >= 0.20)
        ripe = valid & ((h2 <= 44) | (h2 >= 680))
        half = valid & (h2 >= 36) & (h2 <= 120) & (s >= 0.30)
        raw = valid & (h2 >= 130) & (h2 <= 320) & (s >= 0.25)
        labels[ripe] = _MATANG
        labels[half] = _SETENGAH
        labels[raw] = _MENTAH

        dets: List[Detection] = []
        visited = np.zeros((256, 256), dtype=bool)
        for y in range(256):
            for x in range(256):
                if visited[y, x] or labels[y, x] == 0:
                    continue
                cells = _flood_fill(labels, visited, x, y)
                if len(cells) < 3:
                    continue

                ys, xs = zip(*cells)
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)

                bw = max_x - min_x + 1
                bh = max_y - min_y + 1

                # Tolak benda memanjang (daun/batang): bbox hampir persegi.
                elongation = max(bw, bh) / max(bh, bw)
                if elongation > 3.0:
                    continue

                # Tolak blob tidak solid (fragmen/tekstur berantakan).
                fill = len(cells) / (bw * bh)
                if fill < 0.30:
                    continue

                area_w = bw / 256
                area_h = bh / 256
                covered = area_w * area_h
                if covered < 0.0008:
                    continue
                if covered > 0.5:
                    continue

                cell_labels = labels[ys, xs]
                counts = np.bincount(cell_labels, minlength=4)
                dominant = int(np.argmax(counts))
                if dominant == _MENTAH:
                    label = "mentah"
                elif dominant == _SETENGAH:
                    label = "setengah_matang"
                else:
                    label = "matang"

                confidence = round(min(0.95, 0.5 + fill * 0.3 + covered * 3), 4)
                dets.append(
                    Detection(
                        label=label,
                        confidence=confidence,
                        x=round(min_x / 256, 4),
                        y=round(min_y / 256, 4),
                        w=round(area_w, 4),
                        h=round(area_h, 4),
                    )
                )

        dets.sort(key=lambda d: (d.y, d.x))
        return dets[:18]


def _circularity(labels: np.ndarray, cells, min_x, min_y, max_x, max_y) -> float:
    in_cluster = np.zeros_like(labels, dtype=bool)
    for y, x in cells:
        in_cluster[y, x] = True

    area = len(cells)
    perimeter = 0
    height, width = labels.shape
    for y, x in cells:
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                perimeter += 1
            elif not in_cluster[ny, nx]:
                perimeter += 1

    if perimeter == 0:
        return 0.0
    return 4 * np.pi * area / (perimeter * perimeter)


def _flood_fill(labels: np.ndarray, visited: np.ndarray, start_x: int, start_y: int):
    label = labels[start_y, start_x]
    stack = [(start_x, start_y)]
    cells = []
    while stack:
        x, y = stack.pop()
        if x < 0 or y < 0 or x >= 256 or y >= 256:
            continue
        if visited[y, x] or labels[y, x] != label:
            continue
        visited[y, x] = True
        cells.append((y, x))
        stack.append((x + 1, y))
        stack.append((x - 1, y))
        stack.append((x, y + 1))
        stack.append((x, y - 1))
    return cells


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """rgb (...,3) float 0-1 -> hsv (deg, sat, val)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    delta = mx - mn

    h = np.zeros_like(r)
    mask_r = mx == r
    mask_g = mx == g
    mask_b = mx == b
    h[mask_r] = (60 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)) % 360
    h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / delta[mask_g] + 2)) % 360
    h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / delta[mask_b] + 4)) % 360
    h[delta == 0] = 0

    s = np.zeros_like(r)
    np.divide(delta, mx, out=s, where=mx != 0)

    hsv = np.stack([h, s, mx], axis=-1)
    return hsv