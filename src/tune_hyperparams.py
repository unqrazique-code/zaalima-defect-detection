"""
tune_hyperparams.py — Week 2 Day 5-6: Hyperparameter Tuning
Zaalima Development · Project 2 · Computer Vision

Uses YOLOv8's built-in ray-tune integration to find
optimal hyperparameters via random search.

Usage:
    python src/tune_hyperparams.py
"""

import json
from pathlib import Path


def tune(data_yaml="configs/neu_augmented.yaml", iterations=30):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("⚠️  ultralytics not installed.")
        return

    print("=" * 55)
    print("Hyperparameter Tuning — Week 2 Day 5-6")
    print("=" * 55)

    model = YOLO("yolov8n.pt")

    result_grid = model.tune(
        data=data_yaml,
        epochs=20,
        iterations=iterations,
        optimizer="AdamW",
        plots=True,
        save=True,
        val=True,
    )

    # Extract best hyperparams
    best = result_grid.get_best_result()
    best_config = best.config
    best_map    = best.metrics.get("metrics/mAP50(B)", 0)

    summary = {
        "best_mAP50": float(best_map),
        "best_config": {k: v for k, v in best_config.items()
                        if k in ["lr0", "lrf", "momentum", "weight_decay",
                                 "warmup_epochs", "box", "cls", "dfl",
                                 "hsv_h", "hsv_s", "hsv_v", "degrees",
                                 "translate", "scale", "fliplr", "flipud"]},
    }

    out = Path("data/processed/tuning_results.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n✅ Best mAP@0.5: {best_map:.4f}")
    print(f"   Best config saved to {out}")


def error_analysis(weights="runs/train/neu_defect_yolov8/weights/best.pt",
                   data_yaml="configs/neu_dataset.yaml"):
    """Week 2 Day 7: Analyse false positives and false negatives."""
    try:
        from ultralytics import YOLO
        import cv2, numpy as np
    except ImportError:
        return

    print("\n🔍 Error Analysis — Week 2 Day 7")
    model  = YOLO(weights)
    val_dir = Path("data/processed/val/images")
    lbl_dir = Path("data/processed/val/labels")

    fp_count = fn_count = tp_count = 0
    class_errors = {cls: {"fp": 0, "fn": 0} for cls in [
        "crazing","inclusion","patches","pitted_surface","rolled-in_scale","scratches"]}

    for img_path in sorted(val_dir.glob("*.jpg"))[:200]:
        preds  = model.predict(str(img_path), verbose=False, conf=0.25)[0]
        lbl_path = lbl_dir / img_path.with_suffix(".txt").name
        gt_classes = set()
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                cls_id = int(line.split()[0])
                gt_classes.add(cls_id)

        pred_classes = set(int(c) for c in preds.boxes.cls.cpu().numpy()) if preds.boxes else set()
        CLASSES = ["crazing","inclusion","patches","pitted_surface","rolled-in_scale","scratches"]

        for cls_id in gt_classes:
            if cls_id in pred_classes:
                tp_count += 1
            else:
                fn_count += 1
                class_errors[CLASSES[cls_id]]["fn"] += 1

        for cls_id in pred_classes:
            if cls_id not in gt_classes:
                fp_count += 1
                class_errors[CLASSES[cls_id]]["fp"] += 1

    total = tp_count + fp_count + fn_count
    report = {
        "true_positives":  tp_count,
        "false_positives": fp_count,
        "false_negatives": fn_count,
        "precision": tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0,
        "recall":    tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0,
        "class_errors": class_errors,
    }
    out = Path("data/processed/error_analysis.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"   TP: {tp_count}  FP: {fp_count}  FN: {fn_count}")
    print(f"   Precision: {report['precision']:.4f}  Recall: {report['recall']:.4f}")
    print(f"✅ Error analysis saved to {out}")


if __name__ == "__main__":
    tune()
    error_analysis()
