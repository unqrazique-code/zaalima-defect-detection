"""
export_onnx.py — Week 3 Day 1-3: Edge Optimisation
Zaalima Development · Project 2 · Computer Vision

Exports trained YOLOv8 model to ONNX format for
hardware-accelerated edge inference.

Usage:
    python src/export_onnx.py --weights runs/train/neu_defect_yolov8/weights/best.pt
"""

import argparse
import json
import time
from pathlib import Path


def export_to_onnx(weights: str, imgsz: int = 200, dynamic: bool = False):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("⚠️  ultralytics not installed.")
        return None

    print("=" * 50)
    print("ONNX Export — Week 3 Day 1-3")
    print("=" * 50)

    model = YOLO(weights)
    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=dynamic,
        simplify=True,
        opset=12,
    )
    print(f"✅ ONNX model exported to: {onnx_path}")
    return str(onnx_path)


def benchmark_onnx(onnx_path: str, n_runs: int = 100):
    """Benchmark ONNX inference speed vs PyTorch."""
    import numpy as np

    try:
        import onnxruntime as ort
    except ImportError:
        print("⚠️  onnxruntime not installed. Run: pip install onnxruntime")
        return

    print(f"\n⏱️  Benchmarking ONNX inference ({n_runs} runs)...")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    inp_shape = sess.get_inputs()[0].shape
    # Replace dynamic dims with fixed
    shape = [1 if isinstance(d, str) else d for d in inp_shape]
    dummy = np.random.rand(*shape).astype(np.float32)

    # Warmup
    for _ in range(5):
        sess.run(None, {inp_name: dummy})

    start = time.perf_counter()
    for _ in range(n_runs):
        sess.run(None, {inp_name: dummy})
    elapsed = time.perf_counter() - start

    fps = n_runs / elapsed
    ms  = 1000 / fps

    print(f"   ONNX Runtime FPS : {fps:.1f}")
    print(f"   Latency per frame: {ms:.2f} ms")

    stats = {"onnx_fps": round(fps, 2), "onnx_latency_ms": round(ms, 2)}
    Path("data/processed/onnx_benchmark.json").write_text(json.dumps(stats, indent=2))
    print("✅ Benchmark saved.")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/train/neu_defect_yolov8/weights/best.pt")
    parser.add_argument("--imgsz",   default=200, type=int)
    args = parser.parse_args()

    onnx_path = export_to_onnx(args.weights, args.imgsz)
    if onnx_path:
        benchmark_onnx(onnx_path)
        print(f"\n   Next: python src/video_inference.py --model {onnx_path}")
