# Week 1 Progress — Image Processing & Augmentation

## Day 1-2: Dataset Ingestion
- Downloaded NEU Metal Surface Defects Database (1,800 images, 6 classes)
- Implemented XML → YOLO format converter (`src/ingest_dataset.py`)
- Train/val/test split: 80/15/5
- Verified bounding box normalisation across all 1,800 images

## Day 3-5: Augmentation Pipeline
- Built Albumentations pipeline (`src/augment.py`)
- Transforms: HorizontalFlip, VerticalFlip, Rotate±15°, CLAHE, GaussNoise, CoarseDropout
- 4× augmented copies per training image → 7,200 total training images
- Verified no label corruption after augmentation

## Day 6-7: Evaluation Framework
- Implemented mAP@0.5 and mAP@0.5:0.95 computation (`src/evaluate.py`)
- Added per-class precision/recall reporting
- Added FPS benchmark to measure inference speed
