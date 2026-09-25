import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detector import TomatoDetector

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "annotated"
OUT.mkdir(exist_ok=True)

COLORS = {"mentah": (50, 160, 50), "setengah_matang": (240, 150, 30), "matang": (230, 40, 40)}

det = TomatoDetector(str(ROOT.parent / "models" / "best.pt"))
det.load()

for path in sorted(ROOT.glob("*.jpg")):
    pil = Image.open(path).convert("RGB")
    res = det.detect(pil)
    draw = ImageDraw.Draw(pil)
    for d in res["detections"]:
        x1 = d["x"] * pil.width
        y1 = d["y"] * pil.height
        x2 = (d["x"] + d["w"]) * pil.width
        y2 = (d["y"] + d["h"]) * pil.height
        draw.rectangle([x1, y1, x2, y2], outline=COLORS[d["label"]], width=4)
        draw.text((x1 + 3, y1 + 3), f"{d['label']} {d['confidence']:.2f}", fill=COLORS[d["label"]])
    out = OUT / (path.stem + "_annotated.png")
    pil.save(out)
    print(f"{path.name}: {len(res['detections'])} deteksi -> {out.name}")