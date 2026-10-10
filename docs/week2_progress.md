# Week 2 Progress — Model Training & Evaluation

## Day 1-4: YOLOv8 Training
- Trained YOLOv8n on augmented NEU dataset (Kaggle T4 GPU)
- Hyperparams: AdamW lr=0.001, 80 epochs, batch=32, imgsz=200
- Early stopping triggered at epoch 67 (patience=20)
- Best mAP@0.5: 0.847 | mAP@0.5:0.95: 0.621

## Day 5-6: Hyperparameter Tuning
- Ray-tune search over lr0, weight_decay, mosaic, mixup params
- Best config saved to data/processed/tuning_results.json
- 3% mAP improvement over baseline config

## Day 7: Error Analysis
- FP/FN breakdown per class
- Highest confusion: patches ↔ pitted_surface (similar texture)
- Lowest error rate: scratches (linear pattern, distinct)
