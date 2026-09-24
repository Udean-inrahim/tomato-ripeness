import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detector import TomatoDetector

d = TomatoDetector(str(Path(__file__).resolve().parent.parent / "models" / "best.pt"))
d.load()

input_dir = Path(__file__).resolve().parent
for img_file in sorted(input_dir.glob("*.jp*g")):
    pil = Image.open(img_file).convert("RGB")
    res = d.detect(pil)
    print(f"=== {img_file.name} ({pil.width}x{pil.height}) ===")
    if not res["detections"]:
        print("  (tidak ada deteksi)")
    for det in res["detections"]:
        print(f"  {det['label']:<16} conf={det['confidence']:.2f} "
              f"x={det['x']:.2f} y={det['y']:.2f} w={det['w']:.2f} h={det['h']:.2f}")
        # tampilkan warna rata-rata di area box
        x1 = int(det["x"] * pil.width); y1 = int(det["y"] * pil.height)
        x2 = int((det["x"] + det["w"]) * pil.width); y2 = int((det["y"] + det["h"]) * pil.height)
        crop = np.asarray(pil.crop((x1, y1, x2, y2))).reshape(-1, 3)
        mean = crop.mean(axis=0).astype(int)
        print(f"    avg RGB ~ ({mean[0]}, {mean[1]}, {mean[2]})")
    print()