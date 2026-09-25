import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detector import TomatoDetector

img_path = Path(__file__).resolve().parent / "images 6.jpg"
pil = Image.open(img_path).convert("RGB")
print(f"ukuran: {pil.width}x{pil.height}")

det = TomatoDetector(str(Path(__file__).resolve().parent.parent / "models" / "best.pt"))
det.load()

# raw YOLO tanpa filter
res = det.model(pil, conf=0.1, verbose=False)[0]
boxes = res.boxes.xyxyn.cpu().numpy()
scores = res.boxes.conf.cpu().numpy()
classes = res.boxes.cls.cpu().numpy().astype(int)
names = ["mentah", "setengah_matang", "matang"]
print(f"\n-- Raw YOLO (conf>=0.1) total={len(boxes)} --")
for box, sc, cl in zip(boxes, scores, classes):
    print(f"  {names[cl]:<16} conf={sc:.3f} box=({box[0]:.3f},{box[1]:.3f},{box[2]:.3f},{box[3]:.3f})")

print(f"\n-- Deteksi hasil TomatoDetector --")
res2 = det.detect(pil)
for d in res2["detections"]:
    print(f"  {d['label']:<16} conf={d['confidence']:.3f} x={d['x']:.3f} y={d['y']:.3f} w={d['w']:.3f} h={d['h']:.3f}")

print(f"\n  total: {len(res2['detections'])}")

# label manual untuk perbandingan
lbl = Path(__file__).resolve().parent.parent / "labeling" / "labels" / "images 6.txt"
print("\n-- Label manual --")
for line in lbl.read_text().splitlines():
    p = line.split()
    if len(p) == 5:
        cls, xc, yc, w, h = int(p[0]), *map(float, p[1:])
        print(f"  {names[cls]:<16} cx={xc:.3f} cy={yc:.3f} w={w:.3f} h={h:.3f}")