"""
augment.py — Day 3-5: Data Augmentation Pipeline
Zaalima Development · Project 2 · Computer Vision

Applies Albumentations transforms to training images:
  - Random rotations, flips, brightness/contrast
  - Gaussian noise, blur
  - CLAHE (contrast-limited adaptive histogram equalisation)
  - Cutout / CoarseDropout for regularisation

Each training image generates N_AUGMENTS additional copies,
multiplying the effective training set size.

Usage:
    python src/augment.py
"""

import cv2
import json
import random
import numpy as np
from pathlib import Path

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("⚠️  albumentations not installed. Run: pip install albumentations")

TRAIN_IMG_DIR  = Path("data/processed/train/images")
TRAIN_LBL_DIR  = Path("data/processed/train/labels")
AUG_IMG_DIR    = Path("data/augmented/train/images")
AUG_LBL_DIR    = Path("data/augmented/train/labels")
N_AUGMENTS     = 4   # extra copies per image
IMG_SIZE       = 200


def get_transform():
    """Return the Albumentations augmentation pipeline."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.4),
        A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_REFLECT),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.4),
        A.CoarseDropout(
            max_holes=4, max_height=20, max_width=20,
            min_holes=1, fill_value=0, p=0.3
        ),
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.3),
    ], bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.3,
    ))


def read_yolo_labels(label_path: Path):
    """Read YOLO label file → list of (class_id, cx, cy, w, h)."""
    if not label_path.exists():
        return []
    lines = label_path.read_text().strip().splitlines()
    labels = []
    for line in lines:
        parts = line.split()
        if len(parts) == 5:
            cls = int(parts[0])
            bbox = [float(x) for x in parts[1:]]
            labels.append((cls, bbox))
    return labels


def write_yolo_labels(label_path: Path, labels):
    """Write list of (class_id, [cx, cy, w, h]) to YOLO label file."""
    lines = [f"{cls} {' '.join(f'{v:.6f}' for v in bbox)}" for cls, bbox in labels]
    label_path.write_text("\n".join(lines))


def augment_dataset():
    if not ALBUMENTATIONS_AVAILABLE:
        return

    AUG_IMG_DIR.mkdir(parents=True, exist_ok=True)
    AUG_LBL_DIR.mkdir(parents=True, exist_ok=True)

    # First copy originals
    orig_count = 0
    for img_path in sorted(TRAIN_IMG_DIR.glob("*.jpg")):
        import shutil
        shutil.copy2(img_path, AUG_IMG_DIR / img_path.name)
        lbl_src = TRAIN_LBL_DIR / img_path.with_suffix(".txt").name
        if lbl_src.exists():
            shutil.copy2(lbl_src, AUG_LBL_DIR / lbl_src.name)
        orig_count += 1

    print(f"  Copied {orig_count} original images.")

    transform = get_transform()
    aug_count = 0
    errors    = 0

    for img_path in sorted(TRAIN_IMG_DIR.glob("*.jpg")):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        lbl_path = TRAIN_LBL_DIR / img_path.with_suffix(".txt").name
        raw_labels = read_yolo_labels(lbl_path)

        bboxes       = [lbl[1] for lbl in raw_labels]
        class_labels = [lbl[0] for lbl in raw_labels]

        for i in range(N_AUGMENTS):
            try:
                result = transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels,
                )
                aug_img  = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
                aug_bboxes  = result["bboxes"]
                aug_classes = result["class_labels"]

                stem     = img_path.stem
                aug_name = f"{stem}_aug{i}"
                out_img  = AUG_IMG_DIR / f"{aug_name}.jpg"
                out_lbl  = AUG_LBL_DIR / f"{aug_name}.txt"

                cv2.imwrite(str(out_img), aug_img)
                write_yolo_labels(out_lbl, list(zip(aug_classes, aug_bboxes)))
                aug_count += 1

            except Exception as e:
                errors += 1

    total = orig_count + aug_count
    print(f"  Generated {aug_count} augmented images. Total: {total}. Errors: {errors}")

    # Save augmentation stats
    stats = {
        "original": orig_count,
        "augmented": aug_count,
        "total": total,
        "n_augments_per_image": N_AUGMENTS,
        "errors": errors,
    }
    Path("data/augmented/aug_stats.json").write_text(json.dumps(stats, indent=2))
    print("✅ Augmentation complete. Stats saved.")

    # Update dataset YAML to point to augmented data
    yaml_content = f"""# NEU Metal Surface Defects — Augmented dataset config
path: {Path('data/augmented').resolve()}
train: train/images
val:   {Path('data/processed/val/images').resolve()}
test:  {Path('data/processed/test/images').resolve()}

nc: 6
names:
  - crazing
  - inclusion
  - patches
  - pitted_surface
  - rolled-in_scale
  - scratches
"""
    Path("configs/neu_augmented.yaml").write_text(yaml_content)
    print("✅ Augmented dataset YAML written to configs/neu_augmented.yaml")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    print("=" * 50)
    print("Data Augmentation Pipeline — Day 3-5")
    print("=" * 50)
    augment_dataset()
