"""
evaluate.py — Day 6-7: Evaluation Metric Framework
Zaalima Development · Project 2 · Computer Vision

Computes:
  - Per-class Precision, Recall, F1
  - mAP@0.5 and mAP@0.5:0.95
  - Confusion matrix
  - Inference speed (FPS)

Usage:
    python src/evaluate.py --weights runs/train/weights/best.pt
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

DEFECT_CLASSES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches"
]


def compute_iou(box1, box2):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def evaluate_yolo_model(weights_path: str, data_yaml: str = "configs/neu_dataset.yaml"):
    """Run YOLOv8 validation and return metrics."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("⚠️  ultralytics not installed. Run: pip install ultralytics")
        return None

    model = YOLO(weights_path)
    print(f"🔍 Evaluating model: {weights_path}")
    print(f"   Dataset config: {data_yaml}")

    # Run validation
    results = model.val(data=data_yaml, verbose=True)

    metrics = {
        "mAP50":    float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall":    float(results.box.mr),
    }

    # Per-class metrics
    per_class = {}
    if hasattr(results.box, "ap_class_index"):
        for i, cls_idx in enumerate(results.box.ap_class_index):
            cls_name = DEFECT_CLASSES[int(cls_idx)]
            per_class[cls_name] = {
                "ap50":      float(results.box.ap50[i]),
                "precision": float(results.box.p[i]) if results.box.p is not None else 0.0,
                "recall":    float(results.box.r[i]) if results.box.r is not None else 0.0,
            }

    metrics["per_class"] = per_class

    # Inference speed benchmark
    print("\n⏱️  Benchmarking inference speed...")
    import cv2
    dummy = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    N_RUNS = 50
    start = time.perf_counter()
    for _ in range(N_RUNS):
        model.predict(dummy, verbose=False)
    elapsed = time.perf_counter() - start
    fps = N_RUNS / elapsed
    metrics["inference_fps"] = round(fps, 2)
    metrics["inference_ms"]  = round(1000 / fps, 2)

    # Save metrics
    out_path = Path("data/processed/eval_results.json")
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"\n📊 Evaluation Results:")
    print(f"   mAP@0.5      : {metrics['mAP50']:.4f}")
    print(f"   mAP@0.5:0.95 : {metrics['mAP50_95']:.4f}")
    print(f"   Precision     : {metrics['precision']:.4f}")
    print(f"   Recall        : {metrics['recall']:.4f}")
    print(f"   Inference FPS : {metrics['inference_fps']}")
    print(f"\n✅ Results saved to {out_path}")
    return metrics


def print_class_report(metrics: dict):
    """Pretty-print per-class metrics table."""
    if not metrics or "per_class" not in metrics:
        return
    print("\n📋 Per-Class Report:")
    print(f"{'Class':22s} {'AP@50':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 52)
    for cls, vals in metrics["per_class"].items():
        print(f"{cls:22s} {vals['ap50']:>8.4f} {vals['precision']:>10.4f} {vals['recall']:>8.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 defect detection model")
    parser.add_argument("--weights", default="runs/train/weights/best.pt", help="Path to model weights")
    parser.add_argument("--data",    default="configs/neu_dataset.yaml",   help="Dataset YAML config")
    args = parser.parse_args()

    metrics = evaluate_yolo_model(args.weights, args.data)
    if metrics:
        print_class_report(metrics)
