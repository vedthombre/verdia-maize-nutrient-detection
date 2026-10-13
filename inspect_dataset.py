"""
Maize Leaf Nutrient-Deficiency Dataset — Inspection Script
===========================================================
This script performs a **read-only** inspection of the dataset.
It will NOT modify, rename, move, or delete any files.

What it reports
---------------
1. Valid image count per class
2. File extension breakdown
3. Image dimension statistics (min / max / mean width & height)
4. Unreadable / corrupted images
5. Exact duplicate images (via MD5 hash)
6. A clear, consolidated summary table

Requirements
------------
    pip install Pillow
    (Python ≥ 3.8 recommended)

Usage
-----
    python inspect_dataset.py
"""

import os
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

# Pillow is used only for opening images — nothing is written back.
from PIL import Image

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
# Path to the root dataset folder that contains the five class sub-folders.
DATASET_ROOT = Path(__file__).resolve().parent / "Dataset"

# Extensions we treat as "valid image files".
VALID_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif",
}


# ──────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────

def is_image_file(filepath: Path) -> bool:
    """Return True if the file extension looks like an image."""
    return filepath.suffix.lower() in VALID_IMAGE_EXTENSIONS


def compute_md5(filepath: Path, chunk_size: int = 8192) -> str:
    """
    Compute the MD5 hash of a file.

    We read the file in chunks so that even very large images do not
    require loading the entire file into memory at once.
    """
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def try_open_image(filepath: Path):
    """
    Attempt to fully decode an image.

    Pillow is lazy — calling Image.open() only reads the header.
    We call img.verify() which forces a full decode check, reliably
    catching truncated or corrupted files.

    Returns
    -------
    (width, height) on success, or None if the image is unreadable.
    """
    try:
        with Image.open(filepath) as img:
            # .verify() does a full integrity check but invalidates the
            # image object, so we grab the size first.
            size = img.size          # (width, height)
            img.verify()             # raises if corrupted
        return size
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────
# MAIN INSPECTION LOGIC
# ──────────────────────────────────────────────────────────────────────

def inspect_dataset(root: Path):
    """
    Walk every class sub-folder under `root` and collect statistics.

    Data structures populated
    -------------------------
    class_counts   : {class_name: int}         — valid images per class
    ext_counter    : Counter                   — extension frequencies
    widths, heights: [int]                     — dimension lists
    corrupted      : [(class_name, filepath)]  — unreadable images
    hash_map       : {md5: [filepath]}         — for duplicate detection
    """

    # --- accumulators ---
    class_counts: dict[str, int] = {}
    ext_counter: Counter = Counter()
    widths: list[int] = []
    heights: list[int] = []
    corrupted: list[tuple[str, Path]] = []
    hash_map: defaultdict[str, list[Path]] = defaultdict(list)
    non_image_files: list[tuple[str, Path]] = []

    # Discover class folders (sorted for deterministic output).
    class_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    if not class_dirs:
        print(f"[ERROR] No sub-folders found in {root}")
        return

    print(f"Dataset root : {root}")
    print(f"Classes found: {[d.name for d in class_dirs]}\n")

    # --- iterate through each class folder ---
    for class_dir in class_dirs:
        class_name = class_dir.name
        valid_count = 0

        # os.walk gives us recursive traversal in case there are
        # unexpected sub-directories inside a class folder.
        for dirpath, _, filenames in os.walk(class_dir):
            for fname in filenames:
                fpath = Path(dirpath) / fname

                # ── Step 1: check extension ──
                if not is_image_file(fpath):
                    non_image_files.append((class_name, fpath))
                    continue

                ext_counter[fpath.suffix.lower()] += 1

                # ── Step 2: try to open & verify ──
                result = try_open_image(fpath)
                if result is None:
                    corrupted.append((class_name, fpath))
                    continue

                w, h = result
                widths.append(w)
                heights.append(h)
                valid_count += 1

                # ── Step 3: hash for duplicate detection ──
                file_hash = compute_md5(fpath)
                hash_map[file_hash].append(fpath)

        class_counts[class_name] = valid_count

    # ──────────────────────────────────────────────────────────────────
    # REPORT
    # ──────────────────────────────────────────────────────────────────

    separator = "=" * 62

    # --- Per-class counts ---
    print(separator)
    print("  PER-CLASS IMAGE COUNTS")
    print(separator)
    total = 0
    for cls, count in class_counts.items():
        print(f"  {cls:<20s} {count:>5d} images")
        total += count
    print(f"  {'TOTAL':<20s} {total:>5d} images")
    print()

    # --- Extension breakdown ---
    print(separator)
    print("  FILE EXTENSION BREAKDOWN")
    print(separator)
    for ext, count in ext_counter.most_common():
        print(f"  {ext:<10s} {count:>5d}")
    print()

    # --- Dimension statistics ---
    print(separator)
    print("  IMAGE DIMENSION STATISTICS")
    print(separator)
    if widths:
        avg_w = sum(widths) / len(widths)
        avg_h = sum(heights) / len(heights)
        print(f"  Width  -> min: {min(widths):>5d}  max: {max(widths):>5d}  "
              f"mean: {avg_w:>8.1f}")
        print(f"  Height -> min: {min(heights):>5d}  max: {max(heights):>5d}  "
              f"mean: {avg_h:>8.1f}")

        # Also report unique resolutions (useful to see if images are uniform).
        unique_sizes = set(zip(widths, heights))
        print(f"  Unique resolutions: {len(unique_sizes)}")
        if len(unique_sizes) <= 10:
            for w, h in sorted(unique_sizes):
                print(f"    {w} x {h}")
    else:
        print("  (no valid images found)")
    print()

    # --- Non-image files ---
    print(separator)
    print("  NON-IMAGE FILES")
    print(separator)
    if non_image_files:
        for cls, fpath in non_image_files:
            print(f"  [{cls}] {fpath}")
    else:
        print("  None -- all files have recognised image extensions.")
    print()

    # --- Corrupted / unreadable images ---
    print(separator)
    print("  CORRUPTED / UNREADABLE IMAGES")
    print(separator)
    if corrupted:
        for cls, fpath in corrupted:
            print(f"  [{cls}] {fpath}")
    else:
        print("  None -- all images opened and verified successfully.")
    print()

    # --- Duplicate detection ---
    print(separator)
    print("  EXACT DUPLICATE IMAGES (by MD5 hash)")
    print(separator)
    duplicates_found = False
    for file_hash, paths in hash_map.items():
        if len(paths) > 1:
            duplicates_found = True
            print(f"  Hash {file_hash}:")
            for p in paths:
                print(f"    -> {p}")
    if not duplicates_found:
        print("  None -- every image file is unique.")
    print()

    print(separator)
    print("  Inspection complete. No files were modified.")
    print(separator)


# ──────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    inspect_dataset(DATASET_ROOT)
