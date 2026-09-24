"""Train YOLOv8n on synthetic tomato dataset (CPU friendly)."""

import sys
from pathlib import Path

import numpy as np

if not hasattr(np, "trapz"):  # NumPy 2.x removed np.trapz
    np.trapz = np.trapezoid

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent / "dataset"


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 320

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(ROOT / "data.yaml"),
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        device="cpu",
        workers=0,
        project=str(ROOT / "runs"),
        name="yolov8n",
        exist_ok=True,
        verbose=True,
    )
    print("TRAIN DONE")


if __name__ == "__main__":
    main()