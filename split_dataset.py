"""
Maize Leaf Nutrient-Deficiency Dataset -- Stratified Split
==========================================================
Creates a reproducible 70 / 15 / 15  train / validation / test split.

Key design decisions
--------------------
* **Metadata-only**: The split is recorded in a CSV file
  (`dataset_split.csv`) rather than copying images into new folders.
  The original Dataset/ directory is never modified.
* **Two-stage stratified split**: We first separate 70 % train vs
  30 % temp, then split temp 50/50 into validation and test.  Both
  stages use `StratifiedShuffleSplit` so class proportions are
  preserved in every split.
* **Fixed random seed** (`SEED = 42`) ensures identical results on
  every run.

Output CSV columns
------------------
    image_path   -- path relative to the project root (e.g. Dataset/Healthy/img001.jpeg)
    class_name   -- folder name the image belongs to
    split        -- one of: train, val, test

Requirements
------------
    pip install scikit-learn

Usage
-----
    python split_dataset.py
"""

import os
import csv
from pathlib import Path
from collections import Counter

from sklearn.model_selection import StratifiedShuffleSplit

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path(__file__).resolve().parent / "Dataset"
OUTPUT_CSV   = Path(__file__).resolve().parent / "dataset_split.csv"
SEED         = 42

# Recognised image extensions (same list used in inspect_dataset.py).
VALID_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif",
}


# ──────────────────────────────────────────────────────────────────────
# STEP 1 — Collect all image paths and their class labels
# ──────────────────────────────────────────────────────────────────────
# We walk each class sub-folder and record every valid image file.
# Paths are stored *relative to the project root* so the CSV remains
# portable (works on any machine where the repo is cloned).

def collect_image_paths(root: Path):
    """
    Return two parallel lists:
        paths  -- list[str]  relative paths like 'Dataset/Healthy/img.jpeg'
        labels -- list[str]  class names like 'Healthy'
    """
    paths = []
    labels = []
    project_root = root.parent  # one level above Dataset/

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for dirpath, _, filenames in os.walk(class_dir):
            for fname in sorted(filenames):
                fpath = Path(dirpath) / fname
                if fpath.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                    # Store a relative path for portability.
                    rel = fpath.relative_to(project_root)
                    paths.append(str(rel))
                    labels.append(class_name)

    return paths, labels


# ──────────────────────────────────────────────────────────────────────
# STEP 2 — Two-stage stratified split  (70 / 15 / 15)
# ──────────────────────────────────────────────────────────────────────
# Stage A:  70 % train  |  30 % temp
# Stage B:  temp split 50/50  ->  15 % val  |  15 % test
#
# StratifiedShuffleSplit guarantees that *each split* mirrors the
# overall class distribution as closely as possible.

def stratified_split(paths, labels, seed=42):
    """
    Return a dict  {image_path: split_name}  where split_name is one
    of 'train', 'val', or 'test'.
    """
    # --- Stage A: train vs temp (val + test) ---
    splitter_a = StratifiedShuffleSplit(
        n_splits=1, test_size=0.30, random_state=seed
    )
    train_idx, temp_idx = next(splitter_a.split(paths, labels))

    # Gather the temp subset for the second split.
    temp_paths  = [paths[i] for i in temp_idx]
    temp_labels = [labels[i] for i in temp_idx]

    # --- Stage B: val vs test (50/50 of the 30 % temp) ---
    splitter_b = StratifiedShuffleSplit(
        n_splits=1, test_size=0.50, random_state=seed
    )
    val_idx, test_idx = next(splitter_b.split(temp_paths, temp_labels))

    # Build the assignment dictionary.
    assignments = {}
    for i in train_idx:
        assignments[paths[i]] = "train"
    for i in val_idx:
        assignments[temp_paths[i]] = "val"
    for i in test_idx:
        assignments[temp_paths[i]] = "test"

    return assignments


# ──────────────────────────────────────────────────────────────────────
# STEP 3 — Write the CSV
# ──────────────────────────────────────────────────────────────────────

def save_csv(assignments, output_path: Path):
    """
    Write a CSV with columns: image_path, class_name, split.
    Rows are sorted by (split, class_name, image_path) for readability.
    """
    rows = []
    for img_path, split in assignments.items():
        # Derive the class name from the path  (Dataset/<class>/file.jpeg).
        class_name = Path(img_path).parts[1]
        rows.append((img_path, class_name, split))

    # Sort for deterministic, readable output.
    rows.sort(key=lambda r: (r[2], r[1], r[0]))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "class_name", "split"])
        writer.writerows(rows)

    print(f"Saved split metadata to: {output_path}")
    print(f"Total rows written: {len(rows)}")


# ──────────────────────────────────────────────────────────────────────
# STEP 4 — Verification & summary table
# ──────────────────────────────────────────────────────────────────────

def verify_and_summarise(assignments, total_expected=463):
    """
    Print a per-class x per-split table and run sanity checks.
    """
    # Count images per (class, split).
    class_split_counts = Counter()
    split_counts = Counter()
    class_names = set()

    for img_path, split in assignments.items():
        cls = Path(img_path).parts[1]
        class_split_counts[(cls, split)] += 1
        split_counts[split] += 1
        class_names.add(cls)

    class_names = sorted(class_names)
    splits = ["train", "val", "test"]

    # ── Pretty table ──
    sep = "-" * 62
    header = f"  {'Class':<20s} {'Train':>7s} {'Val':>7s} {'Test':>7s} {'Total':>7s}"
    print()
    print(sep)
    print("  SPLIT SUMMARY  (target ratio 70 / 15 / 15)")
    print(sep)
    print(header)
    print(sep)

    for cls in class_names:
        tr = class_split_counts[(cls, "train")]
        va = class_split_counts[(cls, "val")]
        te = class_split_counts[(cls, "test")]
        tot = tr + va + te
        print(f"  {cls:<20s} {tr:>7d} {va:>7d} {te:>7d} {tot:>7d}")

    # Totals row.
    tr_t = split_counts["train"]
    va_t = split_counts["val"]
    te_t = split_counts["test"]
    grand = tr_t + va_t + te_t
    print(sep)
    print(f"  {'TOTAL':<20s} {tr_t:>7d} {va_t:>7d} {te_t:>7d} {grand:>7d}")
    print(f"  {'PERCENTAGE':<20s} {tr_t/grand*100:>6.1f}% {va_t/grand*100:>6.1f}% "
          f"{te_t/grand*100:>6.1f}%")
    print(sep)
    print()

    # ── Sanity checks ──
    errors = []

    # Check 1: total count.
    if grand != total_expected:
        errors.append(
            f"Total count mismatch: expected {total_expected}, got {grand}"
        )

    # Check 2: no image in more than one split.
    path_splits = {}
    for img_path, split in assignments.items():
        if img_path in path_splits:
            errors.append(f"DUPLICATE: {img_path} in both "
                          f"'{path_splits[img_path]}' and '{split}'")
        path_splits[img_path] = split

    # Check 3: every split is non-empty.
    for s in splits:
        if split_counts[s] == 0:
            errors.append(f"Split '{s}' is empty!")

    if errors:
        print("  *** VERIFICATION FAILED ***")
        for e in errors:
            print(f"  [ERROR] {e}")
    else:
        print("  All checks passed:")
        print(f"    - train + val + test = {grand} (expected {total_expected})")
        print(f"    - No image appears in more than one split")
        print(f"    - All three splits are non-empty")
    print()


# ──────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Collecting image paths ...")
    paths, labels = collect_image_paths(DATASET_ROOT)
    print(f"Found {len(paths)} images across {len(set(labels))} classes.\n")

    print("Performing stratified 70/15/15 split (seed={}) ...".format(SEED))
    assignments = stratified_split(paths, labels, seed=SEED)

    save_csv(assignments, OUTPUT_CSV)
    verify_and_summarise(assignments, total_expected=len(paths))
