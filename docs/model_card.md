# Model Card — NEU Defect Detection YOLOv8n

## Model Details
- **Architecture:** YOLOv8n (Ultralytics)
- **Task:** Object detection — metal surface defect classification
- **Input:** RGB image, 200×200px
- **Output:** Bounding boxes with class labels and confidence scores
- **Training dataset:** NEU Metal Surface Defects Database (augmented, 7,200 images)

## Performance
| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.847 |
| mAP@0.5:0.95 | 0.621 |
| Precision | 0.883 |
| Recall | 0.812 |
| PyTorch FPS | ~45 |
| ONNX FPS | ~120 |

## Per-Class Performance
| Class | AP@0.5 | Notes |
|-------|--------|-------|
| crazing | 0.821 | Fine crack network |
| inclusion | 0.856 | Distinct texture |
| patches | 0.798 | Sometimes confused with pitted_surface |
| pitted_surface | 0.812 | Sometimes confused with patches |
| rolled-in_scale | 0.891 | Distinctive shape |
| scratches | 0.903 | Linear pattern, easiest to detect |

## Intended Use
- Real-time quality control on steel manufacturing assembly lines
- Edge deployment via ONNX Runtime on CPU/GPU
- Not intended for safety-critical applications without human oversight

## Limitations
- Trained on 200×200px greyscale images; performance may degrade on higher-res colour images
- patches ↔ pitted_surface confusion (similar texture)
- Not validated on datasets outside NEU

## Training Details
- Platform: Kaggle GPU (NVIDIA T4)
- Epochs: 80 (early stopped at 67)
- Optimiser: AdamW, lr=0.001, cosine decay
- Augmentation: Albumentations (5× expansion) + YOLOv8 mosaic/mixup
