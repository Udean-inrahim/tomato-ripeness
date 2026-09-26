import os
from dataclasses import dataclass
from typing import List

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
    """YOLOv8 detector dengan fallback segmentasi warna."""

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
            except Exception as exc:
                print(f"Gagal memuat model YOLO ({exc}); memakai fallback warna.")
                self.model = None
                self.model_name = "none"
                self.mode = "fallback_warna"
        else:
            print(f"Model {self.model_path} belum ada; memakai fallback warna.")

    def detect(self, pil_image: Image.Image):
        width, height = pil_image.size
        if self.model is not None:
            yolo_dets = self._detect_yolo(pil_image)
            color_dets = self._detect_ripe_color(pil_image)
            dets = _merge_detections(yolo_dets, color_dets)
        else:
            dets = self._detect_fallback(pil_image)

        return {
            "image_width": width,
            "image_height": height,
            "detections": [d.__dict__ for d in dets],
        }

    # ---- YOLO ----
    def _detect_yolo(self, pil_image: Image.Image) -> List[Detection]:
        results = self.model(pil_image, conf=0.1, verbose=False)[0]
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        boxes = results.boxes.xyxyn.cpu().numpy()

        dets: List[Detection] = []
        for box, cls, score in zip(boxes, classes, scores):
            label = LABEL_MAP.get(int(cls))
            if label is None:
                continue
            x1, y1, x2, y2 = box

            # Filter bentuk: tomat bulat (aspect ratio 0.4-2.5)
            ar = (x2 - x1) / (y2 - y1)
            if ar < 0.4 or ar > 2.5:
                continue

            # Skip jika box terlalu kecil (< 1% area gambar)
            box_area = (x2 - x1) * (y2 - y1)
            if box_area < 0.0001:
                continue

            # Class-specific confidence threshold:
            #   mentah    : 0.45 (daun rentan FP)
            #   setengah  : 0.20 (jarang, accept lower conf)
            #   matang    : 0.30 (butuh cukup confident)
            class_conf = {0: 0.45, 1: 0.20, 2: 0.30}
            if float(score) < class_conf[int(cls)]:
                continue

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
        dets.sort(key=lambda d: (d.y, d.x))
        return dets

    def _detect_ripe_color(self, pil_image: Image.Image) -> List[Detection]:
        """Deteksi tomat matang/setengah (merah/oranye) via segmentasi warna."""
        rgb = np.asarray(pil_image.resize((256, 256))).astype(np.float32) / 255.0
        hsv = _rgb_to_hsv(rgb)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        labels = np.zeros((256, 256), dtype=np.int8)
        ripe = (s >= 0.40) & (v >= 0.25) & (v <= 0.97) & ((h <= 25) | (h >= 335))
        half = (h >= 15) & (h <= 65) & (s >= 0.40) & (v >= 0.35) & (v <= 0.97)
        labels[ripe] = _MATANG
        labels[half] = _SETENGAH

        dets: List[Detection] = []
        visited = np.zeros((256, 256), dtype=bool)
        for y in range(256):
            for x in range(256):
                if visited[y, x] or labels[y, x] == 0:
                    continue
                cells = _flood_fill(labels, visited, x, y)
                if len(cells) < 6:
                    continue

                ys, xs = zip(*cells)
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                bw = max_x - min_x + 1
                bh = max_y - min_y + 1

                elongation = max(bw, bh) / max(bh, bw)
                if elongation > 2.4:
                    continue
                if bw < 4 or bh < 4:
                    continue
                circ = _circularity(labels, cells, min_x, min_y, max_x, max_y)
                if circ < 0.45:
                    continue
                fill = len(cells) / (bw * bh)
                if fill < 0.35:
                    continue

                cell_idx = np.asarray(ys), np.asarray(xs)
                mr = float(np.mean(rgb[cell_idx][..., 0])) * 255
                mg = float(np.mean(rgb[cell_idx][..., 1])) * 255
                mb = float(np.mean(rgb[cell_idx][..., 2])) * 255

                ripe_dominant = mr - mg >= 25 and mr - mb >= 40
                half_dominant = mr - mg >= 18 and mr - mb >= 35
                if not (ripe_dominant or half_dominant):
                    continue

                ripe_cells = np.mean((h[cell_idx] <= 25) | (h[cell_idx] >= 335))
                label = "matang" if ripe_dominant or ripe_cells >= 0.5 else "setengah_matang"

                covered = (bw / 256) * (bh / 256)
                if covered < 0.001 or covered > 0.5:
                    continue

                dets.append(
                    Detection(
                        label=label,
                        confidence=0.8,
                        x=round(min_x / 256, 4),
                        y=round(min_y / 256, 4),
                        w=round(bw / 256, 4),
                        h=round(bh / 256, 4),
                    )
                )

        dets.sort(key=lambda d: (d.y, d.x))
        return dets[:18]

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
        mentah = valid & (h2 >= 140) & (h2 <= 300) & (s >= 0.25)
        labels[ripe] = _MATANG
        labels[half] = _SETENGAH
        labels[mentah] = _MENTAH

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

                if bw < 3 or bh < 3:
                    continue
                ar = bw / bh
                if ar < 0.5 or ar > 2.0:
                    continue
                circ = _circularity(labels, cells, min_x, min_y, max_x, max_y)
                if circ < 0.40:
                    continue

                elongation = max(bw, bh) / max(bh, bw)
                if elongation > 3.0:
                    continue

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


def _iou(a: Detection, b: Detection) -> float:
    ax1, ay1 = a.x, a.y
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx1, by1 = b.x, b.y
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = a.w * a.h
    area_b = b.w * b.h
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _merge_detections(yolo_dets, color_dets):
    """Gabung hasil YOLO + warna merah/oranye, buang duplikat (IoU)."""
    merged = list(yolo_dets)
    for cd in color_dets:
        dup = False
        for d in merged:
            if _iou(d, cd) > 0.35:
                dup = True
                break
        if not dup:
            merged.append(cd)
    merged.sort(key=lambda d: (d.y, d.x))
    return merged[:18]


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """rgb (...,3) float 0-1 -> hsv (deg, sat, val)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    delta = mx - mn
    eps = 1e-10

    h = np.full_like(r, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        valid = (delta > eps) & (mx > eps)
        h = np.where(valid & (mx == r), (60 * (((g - b) / np.where(valid, delta, 1.0)) % 6)) % 360, h)
        h = np.where(valid & (mx == g), (60 * ((b - r) / np.where(valid, delta, 1.0) + 2)) % 360, h)
        h = np.where(valid & (mx == b), (60 * ((r - g) / np.where(valid, delta, 1.0) + 4)) % 360, h)

    s = np.zeros_like(r)
    np.divide(delta, mx, out=s, where=mx > eps)

    hsv = np.stack([h, s, mx], axis=-1)
    return hsv
