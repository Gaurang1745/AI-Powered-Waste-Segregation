"""
Download TrashNet dataset and reorganize into 3-class structure for YOLOv8.

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --config config.yaml
"""

import argparse
import os
import shutil
import random
import zipfile
import urllib.request
from pathlib import Path
from collections import Counter, defaultdict

from utils import load_config, CLASS_MAP, TARGET_CLASSES

TRASHNET_URL = (
    "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
)


def download_dataset(raw_dir):
    """Download TrashNet dataset if not already present."""
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    # Check if data already exists
    existing_classes = [
        d.name for d in raw_path.iterdir()
        if d.is_dir() and d.name in CLASS_MAP
    ]
    if len(existing_classes) >= 4:
        print(f"Raw dataset already exists at {raw_path} with classes: {existing_classes}")
        return True

    zip_path = raw_path / "dataset-resized.zip"

    print("Downloading TrashNet dataset...")
    print(f"URL: {TRASHNET_URL}")

    try:
        urllib.request.urlretrieve(TRASHNET_URL, str(zip_path))
        print(f"Downloaded to {zip_path}")
    except Exception as e:
        print(f"\nAutomatic download failed: {e}")
        print("\nPlease download the dataset manually:")
        print(f"  1. Go to https://github.com/garythung/trashnet")
        print(f"  2. Download data/dataset-resized.zip")
        print(f"  3. Extract into {raw_path}/ so the structure is:")
        print(f"     {raw_path}/glass/")
        print(f"     {raw_path}/paper/")
        print(f"     {raw_path}/cardboard/")
        print(f"     {raw_path}/plastic/")
        print(f"     {raw_path}/metal/")
        print(f"     {raw_path}/trash/")
        print(f"\n  OR use Kaggle CLI:")
        print(f"  kaggle datasets download -d feyzazkefe/trashnet -p {raw_path} --unzip")
        return False

    print("Extracting...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(raw_path))

    # The zip extracts into a subdirectory - move contents up if needed
    extracted_dir = raw_path / "dataset-resized"
    if extracted_dir.exists():
        for item in extracted_dir.iterdir():
            dest = raw_path / item.name
            if not dest.exists():
                shutil.move(str(item), str(dest))
        extracted_dir.rmdir()

    # Clean up zip
    zip_path.unlink()
    print("Extraction complete.")
    return True


def validate_raw_data(raw_dir):
    """Validate raw dataset and print statistics."""
    raw_path = Path(raw_dir)
    print(f"\n{'='*50}")
    print("Raw Dataset Summary")
    print(f"{'='*50}")

    total = 0
    for class_name in sorted(CLASS_MAP.keys()):
        class_dir = raw_path / class_name
        if class_dir.exists():
            count = len([
                f for f in class_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ])
            print(f"  {class_name:12s} -> {CLASS_MAP[class_name]:18s} : {count:4d} images")
            total += count
        else:
            print(f"  {class_name:12s} : MISSING")

    print(f"  {'Total':12s} : {' '*20} {total:4d} images")
    return total > 0


def collect_images(raw_dir):
    """Collect all images grouped by target class."""
    raw_path = Path(raw_dir)
    grouped = defaultdict(list)

    for original_class, target_class in CLASS_MAP.items():
        class_dir = raw_path / original_class
        if not class_dir.exists():
            continue

        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                grouped[target_class].append(img_file)

    return grouped


def split_data(images, train_ratio, val_ratio, seed):
    """Split a list of images into train/val/test sets."""
    rng = random.Random(seed)
    shuffled = images.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def create_dataset(config):
    """Create the processed 3-class dataset."""
    ds_cfg = config["dataset"]
    processed_dir = Path(ds_cfg["processed_dir"])

    # Clean previous processed data
    if processed_dir.exists():
        shutil.rmtree(str(processed_dir))

    # Create directory structure
    for split in ["train", "val", "test"]:
        for cls in TARGET_CLASSES:
            (processed_dir / split / cls).mkdir(parents=True, exist_ok=True)

    # Collect and group images
    grouped = collect_images(ds_cfg["raw_dir"])

    # Split and copy
    summary = defaultdict(Counter)

    for target_class, images in grouped.items():
        splits = split_data(
            images,
            ds_cfg["train_split"],
            ds_cfg["val_split"],
            ds_cfg["random_seed"],
        )

        for split_name, split_images in splits.items():
            dest_dir = processed_dir / split_name / target_class

            for img_path in split_images:
                dest_path = dest_dir / img_path.name

                # Handle name collisions (e.g., paper1.jpg and cardboard1.jpg both go to paper_cardboard)
                if dest_path.exists():
                    stem = img_path.stem
                    suffix = img_path.suffix
                    counter = 1
                    while dest_path.exists():
                        dest_path = dest_dir / f"{stem}_{counter}{suffix}"
                        counter += 1

                shutil.copy2(str(img_path), str(dest_path))
                summary[split_name][target_class] += 1

    # Print summary
    print(f"\n{'='*60}")
    print("Processed Dataset Summary")
    print(f"{'='*60}")
    print(f"  {'Class':<20s} {'Train':>6s} {'Val':>6s} {'Test':>6s} {'Total':>6s}")
    print(f"  {'-'*44}")

    totals = Counter()
    for cls in TARGET_CLASSES:
        train_n = summary["train"][cls]
        val_n = summary["val"][cls]
        test_n = summary["test"][cls]
        total_n = train_n + val_n + test_n
        print(f"  {cls:<20s} {train_n:>6d} {val_n:>6d} {test_n:>6d} {total_n:>6d}")
        totals["train"] += train_n
        totals["val"] += val_n
        totals["test"] += test_n

    grand_total = totals["train"] + totals["val"] + totals["test"]
    print(f"  {'-'*44}")
    print(f"  {'Total':<20s} {totals['train']:>6d} {totals['val']:>6d} {totals['test']:>6d} {grand_total:>6d}")
    print(f"\nDataset saved to: {processed_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Prepare waste classification dataset")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    # Download
    if not download_dataset(config["dataset"]["raw_dir"]):
        return

    # Validate
    if not validate_raw_data(config["dataset"]["raw_dir"]):
        print("No valid images found in raw dataset.")
        return

    # Process
    create_dataset(config)
    print("\nDataset preparation complete! You can now run: python train.py")


if __name__ == "__main__":
    main()
