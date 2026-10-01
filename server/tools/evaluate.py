import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "labeling" / "images"
LABEL_DIR = ROOT / "labeling" / "labels"
NAMES = ["mentah", "setengah_matang", "matang"]
sys.path.insert(0, str(ROOT))
from detector import TomatoDetector


def read_labels(path: Path):
    if not path.exists():
        return []
    labels = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, xc, yc, w, h = parts
        labels.append(
            {
                "cls": int(cls),
                "box": [
                    float(xc) - float(w) / 2,
                    float(yc) - float(h) / 2,
                    float(xc) + float(w) / 2,
                    float(yc) + float(h) / 2,
                ],
            }
        )
    return labels


def iou(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def evaluate(detector, image, ground_truth, iou_threshold=0.5):
    result = detector.detect(image)
    predictions = []
    for detection in result["detections"]:
        predictions.append(
            {
                "cls": NAMES.index(detection["label"]),
                "score": detection["confidence"],
                "box": [
                    detection["x"],
                    detection["y"],
                    detection["x"] + detection["w"],
                    detection["y"] + detection["h"],
                ],
            }
        )

    matched_pred = set()
    tp = 0
    for gt in ground_truth:
        best_iou = iou_threshold
        best_index = -1
        for index, pred in enumerate(predictions):
            if index in matched_pred or pred["cls"] != gt["cls"]:
                continue
            score = iou(pred["box"], gt["box"])
            if score >= best_iou:
                best_iou = score
                best_index = index
        if best_index >= 0:
            matched_pred.add(best_index)
            tp += 1

    fp = len(predictions) - len(matched_pred)
    fn = len(ground_truth) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return tp, fp, fn, precision, recall, f1, len(predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models" / "best.pt"))
    args = parser.parse_args()

    detector = TomatoDetector(args.model)
    detector.load()
    if detector.model is None:
        raise RuntimeError(detector.load_error or "Model YOLO tidak berhasil dimuat")

    images = []
    for image_path in sorted(IMAGE_DIR.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        label_path = LABEL_DIR / (image_path.stem + ".txt")
        images.append((image_path, read_labels(label_path)))

    total_tp = total_fp = total_fn = 0
    for image_path, ground_truth in images:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        tp, fp, fn, precision, recall, f1, predicted = evaluate(
            detector, image, ground_truth
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn
        print(
            f"{image_path.name}: gt={len(ground_truth):2d} pred={predicted:2d} "
            f"tp={tp:2d} fp={fp:2d} fn={fn:2d} P={precision:.2f} R={recall:.2f} F1={f1:.2f}"
        )
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(
        f"TOTAL: tp={total_tp} fp={total_fp} fn={total_fn} "
        f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}"
    )


if __name__ == "__main__":
    main()
