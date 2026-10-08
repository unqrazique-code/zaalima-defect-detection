# Real-Time Industrial Defect Detection System (Computer Vision)

Zaalima Development — Data Science & ML Internship, Project 2 (Month 2)

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://zaalima-defect-detection.streamlit.app/)

> 🚀 **[Try the live demo → zaalima-defect-detection.streamlit.app](https://zaalima-defect-detection.streamlit.app/)**

## What this is
A Computer Vision pipeline for manufacturing quality control. Detects and classifies microscopic surface defects on metal parts in real-time using YOLOv8, trained on the NEU Metal Surface Defects Database.

## Dataset
[NEU Metal Surface Defects Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) — 1,800 images, 6 defect classes, 300 images per class.

| Class | Description |
|-------|-------------|
| crazing | Network of fine surface cracks |
| inclusion | Foreign material embedded in surface |
| patches | Irregular surface discolouration |
| pitted_surface | Small pits/holes on surface |
| rolled-in_scale | Scale pressed into surface during rolling |
| scratches | Linear surface marks |

## Tech Stack
- **Model:** YOLOv8n (Ultralytics)
- **Augmentation:** Albumentations
- **Edge Export:** ONNX Runtime
- **Video Inference:** OpenCV
- **API:** FastAPI + Uvicorn
- **Container:** Docker
- **Demo:** Streamlit

## 4-Week Timeline

### Week 1: Image Processing & Augmentation
- [x] Day 1-2: `src/ingest_dataset.py` — NEU dataset ingestion, YOLO format conversion, train/val/test split
- [x] Day 3-5: `src/augment.py` — Albumentations pipeline (flip, rotate, CLAHE, noise, dropout) — 5× dataset
- [x] Day 6-7: `src/evaluate.py` — mAP evaluation metric framework

### Week 2: Model Training & Evaluation
- [x] Day 1-4: `src/train_yolo.py` — YOLOv8 training on augmented dataset
- [x] Day 5-6: `src/tune_hyperparams.py` — Hyperparameter tuning + error analysis
- [x] Day 7: Error analysis (FP/FN breakdown per class)

### Week 3: Edge Optimisation
- [x] Day 1-3: `src/export_onnx.py` — PyTorch → ONNX export + benchmark
- [x] Day 4-7: `src/video_inference.py` — Real-time OpenCV inference script

### Week 4: API, Load Testing & Deployment
- [x] Day 1-3: `src/api.py` — FastAPI service (`/detect`, `/detect_batch`, `/health`, `/metrics`)
- [x] Day 4-5: `src/load_test.py` — Concurrent load test (aiohttp)
- [x] Day 6-7: `Dockerfile` + Streamlit demo

## Setup

```bash
# Clone
git clone https://github.com/unqrazique-code/zaalima-defect-detection.git
cd zaalima-defect-detection

# Install
pip install -r requirements.txt

# Download dataset → place at data/raw/NEU-DET/

# Run pipeline
python src/ingest_dataset.py   # Day 1-2
python src/augment.py          # Day 3-5
python src/train_yolo.py       # Week 2 (GPU recommended)
python src/evaluate.py         # Metrics
python src/export_onnx.py      # ONNX export
python src/video_inference.py  # Real-time inference

# Run API
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Run demo
streamlit run streamlit_app.py
```

## Docker

```bash
docker build -t defect-detection .
docker run -p 8000:8000 defect-detection
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/detect` | Upload image → get detections |
| POST | `/detect_batch` | Upload multiple images |
| GET | `/health` | Health check |
| GET | `/metrics` | Request metrics |
| GET | `/classes` | List defect classes |
| GET | `/docs` | Swagger UI |
