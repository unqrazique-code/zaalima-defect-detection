"""
utils.py — Shared utilities
Zaalima Development · Project 2 · Computer Vision
"""

import cv2
import numpy as np
from pathlib import Path

DEFECT_CLASSES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches"
]

CLASS_COLORS = [
    (255, 80,  80),
    (80,  255, 80),
    (80,  80,  255),
    (255, 255, 80),
    (255, 80,  255),
    (80,  255, 255),
]


def preprocess_image(img: np.ndarray, size: int = 200) -> np.ndarray:
    """Resize and normalise image for model input."""
    img = cv2.resize(img, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img.transpose(2, 0, 1), axis=0)


def draw_boxes(frame: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes and labels on frame."""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_id = det["class_id"]
        color  = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        label  = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return frame


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    """Convert YOLO normalised coords to pixel xyxy."""
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2


def compute_class_distribution(label_dir: Path) -> dict:
    """Count instances per class in a label directory."""
    counts = {cls: 0 for cls in DEFECT_CLASSES}
    for lbl_file in label_dir.glob("*.txt"):
        for line in lbl_file.read_text().strip().splitlines():
            cls_id = int(line.split()[0])
            if cls_id < len(DEFECT_CLASSES):
                counts[DEFECT_CLASSES[cls_id]] += 1
    return counts
