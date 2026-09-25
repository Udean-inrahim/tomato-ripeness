import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detector import TomatoDetector, _rgb_to_hsv

img_path = Path(__file__).resolve().parent / "dsc02094.jpg"
pil = Image.open(img_path).convert("RGB")
arr = np.asarray(pil).astype(np.float32) / 255.0
hsv = _rgb_to_hsv(arr)
h = hsv[..., 0].reshape(-1)
s = hsv[..., 1].reshape(-1)
v = hsv[..., 2].reshape(-1)

print(f"ukuran: {pil.width}x{pil.height}")

# dominasi warna
red = ((h <= 30) | (h >= 330)) & (s >= 0.38) & (v >= 0.3)
orange = (h >= 10) & (h <= 60) & (s >= 0.3) & (v >= 0.4)
green = (h >= 60) & (h <= 160) & (s >= 0.15) & (v >= 0.3) & (v <= 0.85)
print(f"piksel merah: {red.mean()*100:.2f}%, oranye: {orange.mean()*100:.2f}%, hijau(sdg): {green.mean()*100:.2f}%")

# YOLO mentah (tanpa post-filter) dengan conf rendah
det = TomatoDetector(str(Path(__file__).resolve().parent.parent / "models" / "best.pt"))
det.load()
model = det.model
res = model(pil, conf=0.1, verbose=False)[0]
boxes = res.boxes.xyxyn.cpu().numpy()
scores = res.boxes.conf.cpu().numpy()
classes = res.boxes.cls.cpu().numpy().astype(int)

names = ["mentah", "setengah_matang", "matang"]
print("\n-- Raw YOLO (conf>=0.1) --")
for box, sc, cl in zip(boxes, scores, classes):
    x1, y1, x2, y2 = box
    bw = int(x2 * pil.width) - int(x1 * pil.width)
    bh = int(y2 * pil.height) - int(y1 * pil.height)
    crop = arr[int(y1*pil.height):int(y2*pil.height), int(x1*pil.width):int(x2*pil.width)]
    m = crop.reshape(-1, 3).mean(axis=0) * 255
    mh = float(np.mean(_rgb_to_hsv(crop)[..., 0]))
    print(f"  {names[cl]:<16} conf={sc:.2f} box=({x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}) "
          f"{bw}x{bh}px RGB~({m[0]:.0f},{m[1]:.0f},{m[2]:.0f}) hue={mh:.0f}")
print(f"  total raw: {len(boxes)}")