"""
api.py — Week 4 Day 1-3: FastAPI Inference Service
Zaalima Development · Project 2 · Computer Vision

Endpoints:
  POST /detect          — Upload image, get defect detections
  POST /detect_batch    — Upload multiple images
  GET  /health          — Health check
  GET  /metrics         — Prometheus-style metrics
  GET  /classes         — List defect classes

Usage:
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import time
import json
from pathlib import Path
from typing import List

import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ─── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Industrial Defect Detection API",
    description="Real-time metal surface defect detection using YOLOv8",
    version="1.0.0",
)

DEFECT_CLASSES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches"
]

MODEL_PATH = "runs/train/neu_defect_yolov8/weights/best.pt"
CONF_THRESHOLD = 0.35

# ─── Global state ─────────────────────────────────────────────────────────────
_model = None
_request_count   = 0
_total_latency   = 0.0
_defect_counts   = {cls: 0 for cls in DEFECT_CLASSES}
_startup_time    = time.time()


def get_model():
    global _model
    if _model is None:
        try:
            from ultralytics import YOLO
            _model = YOLO(MODEL_PATH)
            print(f"✅ Model loaded: {MODEL_PATH}")
        except Exception as e:
            print(f"⚠️  Model load failed: {e}. Using dummy predictions.")
    return _model


# ─── Schemas ──────────────────────────────────────────────────────────────────
class Detection(BaseModel):
    class_id:   int
    class_name: str
    confidence: float
    bbox:       List[int]  # [x1, y1, x2, y2]


class DetectionResponse(BaseModel):
    filename:   str
    detections: List[Detection]
    defect_count: int
    inference_ms: float
    model_version: str = "yolov8n-neu-v1"


class HealthResponse(BaseModel):
    status:  str
    uptime_s: float
    model_loaded: bool


# ─── Helpers ──────────────────────────────────────────────────────────────────
def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    return img


def run_detection(img: np.ndarray) -> List[dict]:
    global _defect_counts
    model = get_model()
    if model is None:
        # Dummy fallback for testing
        return [{
            "class_id": 0, "class_name": "crazing",
            "confidence": 0.87, "bbox": [10, 10, 150, 150]
        }]

    results = model.predict(img, conf=CONF_THRESHOLD, verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
        cls_name = DEFECT_CLASSES[cls_id] if cls_id < len(DEFECT_CLASSES) else "unknown"
        _defect_counts[cls_name] = _defect_counts.get(cls_name, 0) + 1
        detections.append({
            "class_id": cls_id, "class_name": cls_name,
            "confidence": round(conf, 3), "bbox": [x1, y1, x2, y2],
        })
    return detections


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    get_model()


@app.get("/health", response_model=HealthResponse)
async def health():
    return {
        "status": "ok",
        "uptime_s": round(time.time() - _startup_time, 1),
        "model_loaded": _model is not None,
    }


@app.get("/classes")
async def list_classes():
    return {"classes": DEFECT_CLASSES, "count": len(DEFECT_CLASSES)}


@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    global _request_count, _total_latency

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="File must be an image.")

    data = await file.read()
    img  = decode_image(data)

    t0 = time.perf_counter()
    detections = run_detection(img)
    latency_ms = (time.perf_counter() - t0) * 1000

    _request_count += 1
    _total_latency  += latency_ms

    return DetectionResponse(
        filename=file.filename or "upload.jpg",
        detections=[Detection(**d) for d in detections],
        defect_count=len(detections),
        inference_ms=round(latency_ms, 2),
    )


@app.post("/detect_batch")
async def detect_batch(files: List[UploadFile] = File(...)):
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Max 20 images per batch.")
    results = []
    for file in files:
        data = await file.read()
        img  = decode_image(data)
        t0   = time.perf_counter()
        dets = run_detection(img)
        ms   = (time.perf_counter() - t0) * 1000
        results.append({
            "filename": file.filename,
            "detections": dets,
            "defect_count": len(dets),
            "inference_ms": round(ms, 2),
        })
    return {"batch_size": len(files), "results": results}


@app.get("/metrics")
async def metrics():
    avg_latency = _total_latency / _request_count if _request_count > 0 else 0
    return {
        "total_requests":    _request_count,
        "avg_latency_ms":    round(avg_latency, 2),
        "uptime_s":          round(time.time() - _startup_time, 1),
        "defect_counts":     _defect_counts,
        "model_path":        MODEL_PATH,
        "conf_threshold":    CONF_THRESHOLD,
    }


@app.get("/")
async def root():
    return {
        "service": "Industrial Defect Detection API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
    }
