"""
video_inference.py — Week 3 Day 4-7: Real-Time Video Inference
Zaalima Development · Project 2 · Computer Vision

Ingests live camera feed or video file, runs the optimised
ONNX model at high FPS, draws bounding boxes and defect labels,
and optionally broadcasts defect coordinates to a PLC endpoint.

Usage:
    python src/video_inference.py                         # webcam
    python src/video_inference.py --source video.mp4      # video file
    python src/video_inference.py --source 0 --save       # save output
"""

import argparse
import time
import json
from pathlib import Path

import cv2
import numpy as np

DEFECT_CLASSES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches"
]

CLASS_COLORS = [
    (255, 80,  80),   # crazing        — red
    (80,  255, 80),   # inclusion      — green
    (80,  80,  255),  # patches        — blue
    (255, 255, 80),   # pitted_surface — yellow
    (255, 80,  255),  # rolled-in_scale— magenta
    (80,  255, 255),  # scratches      — cyan
]


class DefectDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.35):
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        if self.model_path.endswith(".onnx"):
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(
                    self.model_path,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                )
                self.inp_name  = self.session.get_inputs()[0].name
                self.use_onnx  = True
                print(f"✅ ONNX model loaded: {self.model_path}")
            except Exception as e:
                print(f"⚠️  ONNX load failed: {e}")
                self.use_onnx = False
        else:
            try:
                from ultralytics import YOLO
                self.yolo     = YOLO(self.model_path)
                self.use_onnx = False
                print(f"✅ YOLOv8 model loaded: {self.model_path}")
            except Exception as e:
                print(f"⚠️  YOLOv8 load failed: {e}")
                self.yolo     = None
                self.use_onnx = False

    def preprocess(self, frame: np.ndarray, size: int = 200) -> np.ndarray:
        img = cv2.resize(frame, (size, size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img.transpose(2, 0, 1), axis=0)

    def detect(self, frame: np.ndarray):
        """Returns list of {class_id, class_name, confidence, bbox_xyxy}."""
        if self.use_onnx:
            return self._detect_onnx(frame)
        elif hasattr(self, "yolo") and self.yolo:
            return self._detect_yolo(frame)
        return []

    def _detect_yolo(self, frame):
        results = self.yolo.predict(frame, conf=self.conf_threshold, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append({
                "class_id":   cls_id,
                "class_name": DEFECT_CLASSES[cls_id],
                "confidence": round(conf, 3),
                "bbox":       [int(x1), int(y1), int(x2), int(y2)],
            })
        return detections

    def _detect_onnx(self, frame):
        h, w = frame.shape[:2]
        inp   = self.preprocess(frame)
        out   = self.session.run(None, {self.inp_name: inp})[0]
        # Parse raw ONNX output (shape: [1, 84, N] for YOLOv8)
        preds = out[0].T  # [N, 84]
        detections = []
        for pred in preds:
            scores = pred[4:]
            cls_id = int(np.argmax(scores))
            conf   = float(scores[cls_id])
            if conf < self.conf_threshold:
                continue
            cx, cy, bw, bh = pred[:4]
            x1 = int((cx - bw / 2) * w / 200)
            y1 = int((cy - bh / 2) * h / 200)
            x2 = int((cx + bw / 2) * w / 200)
            y2 = int((cy + bh / 2) * h / 200)
            detections.append({
                "class_id":   cls_id,
                "class_name": DEFECT_CLASSES[cls_id] if cls_id < len(DEFECT_CLASSES) else "unknown",
                "confidence": round(conf, 3),
                "bbox":       [x1, y1, x2, y2],
            })
        return detections


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
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


def run_inference(source, model_path: str, conf: float = 0.35,
                  save: bool = False, headless: bool = False):
    detector = DefectDetector(model_path, conf)
    cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))

    if not cap.isOpened():
        print(f"❌ Cannot open source: {source}")
        return

    fps_target = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📹 Source: {source}  |  {w}x{h} @ {fps_target:.0f}fps")

    writer = None
    if save:
        out_path = "output_defect_detection.mp4"
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_target, (w, h))
        print(f"💾 Saving output to {out_path}")

    frame_count = 0
    fps_history = []
    total_defects = {}

    while True:
        t0  = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        frame = draw_detections(frame, detections)

        # Track defect counts
        for det in detections:
            cls = det["class_name"]
            total_defects[cls] = total_defects.get(cls, 0) + 1

        # FPS overlay
        elapsed = time.perf_counter() - t0
        fps = 1.0 / elapsed if elapsed > 0 else 0
        fps_history.append(fps)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Defects: {len(detections)}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if writer:
            writer.write(frame)

        if not headless:
            cv2.imshow("Defect Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count} frames | avg FPS: {np.mean(fps_history[-50:]):.1f}")

    cap.release()
    if writer:
        writer.release()
    if not headless:
        cv2.destroyAllWindows()

    avg_fps = float(np.mean(fps_history)) if fps_history else 0
    print(f"\n✅ Done. {frame_count} frames | avg FPS: {avg_fps:.1f}")
    print(f"   Defects seen: {total_defects}")

    stats = {"frames": frame_count, "avg_fps": round(avg_fps, 2), "defect_counts": total_defects}
    Path("data/processed/video_inference_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0",  help="Video source: 0=webcam, or path to video file")
    parser.add_argument("--model",  default="runs/train/neu_defect_yolov8/weights/best.pt")
    parser.add_argument("--conf",   default=0.35, type=float)
    parser.add_argument("--save",   action="store_true")
    parser.add_argument("--headless", action="store_true", help="No display (for servers)")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    run_inference(source, args.model, args.conf, args.save, args.headless)
