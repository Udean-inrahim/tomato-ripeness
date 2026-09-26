import argparse
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "labeling" / "images"
LABEL_DIR = ROOT / "labeling" / "labels"
NAMES = ["mentah", "setengah_matang", "matang"]


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


def evaluate(model, image, ground_truth, imgsz, conf, iou_threshold=0.5):
    result = model.predict(
        image,
        imgsz=imgsz,
        conf=conf,
        iou=0.5,
        device="cpu",
        verbose=False,
    )[0]
    predictions = []
    if result.boxes is not None:
        boxes = result.boxes.xyxyn.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        for box, score, cls in zip(boxes, scores, classes):
            predictions.append(
                {"cls": int(cls), "score": float(score), "box": box.tolist()}
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
    parser.add_argument("--sizes", default="320,480,640")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    model = YOLO(args.model)
    images = []
    for image_path in sorted(IMAGE_DIR.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        label_path = LABEL_DIR / (image_path.stem + ".txt")
        images.append((image_path, read_labels(label_path)))

    for imgsz in [int(value) for value in args.sizes.split(",")]:
        total_tp = total_fp = total_fn = 0
        print(f"\nimgsz={imgsz} conf={args.conf:.2f}")
        for image_path, ground_truth in images:
            image = Image.open(image_path).convert("RGB")
            tp, fp, fn, precision, recall, f1, predicted = evaluate(
                model, image, ground_truth, imgsz, args.conf
            )
            total_tp += tp
            total_fp += fp
            total_fn += fn
            print(
                f"  {image_path.name}: gt={len(ground_truth):2d} pred={predicted:2d} "
                f"tp={tp:2d} fp={fp:2d} fn={fn:2d} P={precision:.2f} R={recall:.2f} F1={f1:.2f}"
            )
        precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
        recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(
            f"  TOTAL: tp={total_tp} fp={total_fp} fn={total_fn} "
            f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}"
        )


if __name__ == "__main__":
    main()