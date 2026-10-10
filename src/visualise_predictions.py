"""
visualise_predictions.py — Visualise model predictions on val set
Zaalima Development · Project 2 · Computer Vision

Usage:
    python src/visualise_predictions.py --weights runs/train/neu_defect_yolov8/weights/best.pt
"""

import argparse
import random
from pathlib import Path
import cv2
import numpy as np

DEFECT_CLASSES = ["crazing","inclusion","patches","pitted_surface","rolled-in_scale","scratches"]
COLORS = [(255,80,80),(80,255,80),(80,80,255),(255,255,80),(255,80,255),(80,255,255)]


def visualise(weights: str, n_samples: int = 16, conf: float = 0.3):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("pip install ultralytics")
        return

    model   = YOLO(weights)
    val_dir = Path("data/processed/val/images")
    images  = list(val_dir.glob("*.jpg"))
    samples = random.sample(images, min(n_samples, len(images)))

    grid_size = int(n_samples ** 0.5)
    cell_size = 200
    grid = np.zeros((grid_size * cell_size, grid_size * cell_size, 3), dtype=np.uint8)

    for idx, img_path in enumerate(samples):
        img     = cv2.imread(str(img_path))
        results = model.predict(img, conf=conf, verbose=False)[0]
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf_  = float(box.conf[0])
            x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
            color  = COLORS[cls_id % len(COLORS)]
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
            cv2.putText(img, f"{DEFECT_CLASSES[cls_id]} {conf_:.2f}",
                        (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        row, col = idx // grid_size, idx % grid_size
        img_resized = cv2.resize(img, (cell_size, cell_size))
        grid[row*cell_size:(row+1)*cell_size, col*cell_size:(col+1)*cell_size] = img_resized

    out_path = "data/processed/prediction_grid.jpg"
    cv2.imwrite(out_path, grid)
    print(f"✅ Prediction grid saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/train/neu_defect_yolov8/weights/best.pt")
    parser.add_argument("--samples", default=16, type=int)
    parser.add_argument("--conf",    default=0.3, type=float)
    args = parser.parse_args()
    visualise(args.weights, args.samples, args.conf)
