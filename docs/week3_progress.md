# Week 3 Progress — Edge Optimisation

## Day 1-3: ONNX Export
- Exported best.pt → best.onnx (opset 12, graph simplified)
- ONNX Runtime benchmark: 120 FPS vs 45 FPS PyTorch (2.7× speedup)
- Latency: 8.3ms per frame on CPU

## Day 4-7: Real-Time Video Inference
- Built OpenCV inference pipeline (`src/video_inference.py`)
- Supports webcam (source=0) and video file input
- Per-class colour-coded bounding boxes with confidence scores
- FPS overlay and defect counter displayed in real-time
- Tested on sample industrial inspection video at 60+ FPS
