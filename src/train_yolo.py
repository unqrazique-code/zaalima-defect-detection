"""
train_yolo.py — Week 2: YOLOv8 Model Training
Zaalima Development · Project 2 · Computer Vision

Trains YOLOv8n (nano) on the augmented NEU dataset.
Recommended: run on Kaggle GPU (free T4).

Usage:
    python src/train_yolo.py
    python src/train_yolo.py --model yolov8s --epochs 100 --imgsz 416
"""

import argparse
import json
from pathlib import Path


def train(model_size="yolov8n", epochs=50, imgsz=200, batch=16, data_yaml="configs/neu_augmented.yaml"):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("⚠️  ultralytics not installed. Run: pip install ultralytics")
        return

    print("=" * 55)
    print("YOLOv8 Training — Week 2")
    print("=" * 55)
    print(f"  Model   : {model_size}.pt")
    print(f"  Epochs  : {epochs}")
    print(f"  Img size: {imgsz}")
    print(f"  Batch   : {batch}")
    print(f"  Data    : {data_yaml}")
    print()

    model = YOLO(f"{model_size}.pt")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="neu_defect_yolov8",
        project="runs/train",
        patience=15,           # early stopping
        save=True,
        save_period=10,
        val=True,
        plots=True,
        cache=True,
        workers=2,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        augment=True,          # YOLOv8 built-in mosaic augment
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.3,
        mosaic=1.0,
        mixup=0.1,
    )

    # Save training summary
    summary = {
        "model": model_size,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "best_weights": str(Path("runs/train/neu_defect_yolov8/weights/best.pt").resolve()),
        "final_mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
        "final_mAP50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
    }
    Path("data/processed/training_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n✅ Training complete!")
    print(f"   Best weights: {summary['best_weights']}")
    print(f"   mAP@0.5     : {summary['final_mAP50']:.4f}")
    print(f"   mAP@0.5:0.95: {summary['final_mAP50_95']:.4f}")
    print("\n   Next: python src/tune_hyperparams.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="yolov8n", help="YOLOv8 model size (yolov8n/s/m)")
    parser.add_argument("--epochs", default=50,  type=int)
    parser.add_argument("--imgsz",  default=200, type=int)
    parser.add_argument("--batch",  default=16,  type=int)
    parser.add_argument("--data",   default="configs/neu_augmented.yaml")
    args = parser.parse_args()
    train(args.model, args.epochs, args.imgsz, args.batch, args.data)
