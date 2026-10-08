"""
ingest_dataset.py — Day 1-2: NEU Metal Surface Defects Dataset Ingestion
Zaalima Development · Project 2 · Computer Vision

Downloads and organizes the NEU dataset into YOLO annotation format.
NEU dataset: 6 defect classes, 1800 images (300 per class), 200x200px

Classes:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled-in_scale
  5: scratches

Usage:
    python src/ingest_dataset.py
"""

import os
import shutil
import random
import json
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────
DEFECT_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

CLASS_TO_ID = {cls: i for i, cls in enumerate(DEFECT_CLASSES)}

RAW_DIR       = Path("data/raw/NEU-DET")
PROCESSED_DIR = Path("data/processed")
TRAIN_RATIO   = 0.8
VAL_RATIO     = 0.15
TEST_RATIO    = 0.05

SPLITS = ["train", "val", "test"]


def create_yolo_dirs():
    """Create YOLO-format directory structure."""
    for split in SPLITS:
        (PROCESSED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (PROCESSED_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
    print("✅ Directory structure created.")


def convert_annotation_to_yolo(xml_path: Path, class_id: int, img_w=200, img_h=200):
    """
    Convert a simple bounding box annotation to YOLO format.
    YOLO format: <class_id> <cx> <cy> <w> <h>  (all normalised 0-1)

    For NEU dataset the bounding box covers the full defect region.
    If XML parsing is unavailable we fall back to a centred full-image box.
    """
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_path)
        root = tree.getroot()
        lines = []
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            cx = ((xmin + xmax) / 2) / img_w
            cy = ((ymin + ymax) / 2) / img_h
            bw = (xmax - xmin) / img_w
            bh = (ymax - ymin) / img_h
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        return "\n".join(lines)
    except Exception:
        # Fallback: centred full-image box
        return f"{class_id} 0.500000 0.500000 0.900000 0.900000"


def organise_dataset():
    """Split images into train/val/test and write YOLO labels."""
    if not RAW_DIR.exists():
        print(f"⚠️  Raw dataset not found at {RAW_DIR}")
        print("   Download NEU-DET from: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database")
        print("   Place it at: data/raw/NEU-DET/")
        print("   Expected structure:")
        print("     data/raw/NEU-DET/IMAGES/  (*.jpg)")
        print("     data/raw/NEU-DET/ANNOTATIONS/  (*.xml)")
        return False

    image_dir = RAW_DIR / "IMAGES"
    annot_dir = RAW_DIR / "ANNOTATIONS"

    stats = {}
    for cls in DEFECT_CLASSES:
        images = sorted(image_dir.glob(f"{cls}_*.jpg"))
        if not images:
            images = sorted(image_dir.glob(f"{cls}*.jpg"))

        random.shuffle(images)
        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        split_map = (
            [(img, "train") for img in images[:n_train]] +
            [(img, "val")   for img in images[n_train:n_train + n_val]] +
            [(img, "test")  for img in images[n_train + n_val:]]
        )

        class_id = CLASS_TO_ID[cls]
        for img_path, split in split_map:
            # Copy image
            dst_img = PROCESSED_DIR / split / "images" / img_path.name
            shutil.copy2(img_path, dst_img)

            # Convert annotation
            xml_path = annot_dir / img_path.with_suffix(".xml").name
            label_str = convert_annotation_to_yolo(xml_path, class_id)
            label_path = PROCESSED_DIR / split / "labels" / img_path.with_suffix(".txt").name
            label_path.write_text(label_str)

        stats[cls] = {"total": n, "train": n_train, "val": n_val, "test": n - n_train - n_val}
        print(f"  {cls:20s} → train:{n_train}  val:{n_val}  test:{n - n_train - n_val}")

    # Save split stats
    (PROCESSED_DIR / "split_stats.json").write_text(json.dumps(stats, indent=2))
    print("✅ Dataset organised into YOLO format.")
    return True


def write_yaml():
    """Write dataset YAML for YOLOv8 training."""
    yaml_content = f"""# NEU Metal Surface Defects — YOLOv8 dataset config
path: {PROCESSED_DIR.resolve()}
train: train/images
val:   val/images
test:  test/images

nc: {len(DEFECT_CLASSES)}
names: {DEFECT_CLASSES}
"""
    yaml_path = Path("configs/neu_dataset.yaml")
    yaml_path.parent.mkdir(exist_ok=True)
    yaml_path.write_text(yaml_content)
    print(f"✅ Dataset YAML written to {yaml_path}")


if __name__ == "__main__":
    random.seed(42)
    print("=" * 50)
    print("NEU Dataset Ingestion — Day 1-2")
    print("=" * 50)
    create_yolo_dirs()
    ok = organise_dataset()
    if ok:
        write_yaml()
        print("\n🎉 Done! Run augmentation next: python src/augment.py")
    else:
        print("\n📥 Please download the dataset first, then re-run this script.")
