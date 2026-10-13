"""
Step 1 of 2: Pre-compute and SAVE GrabCut masks to disk.
Lightweight — no PyTorch, no training. Just OpenCV + numpy.
Saves each mask as a .npy file in masks_cache/ directory.
"""
import cv2
import csv
import time
import numpy as np
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH     = PROJECT_ROOT / "dataset_split.csv"
MASK_DIR     = PROJECT_ROOT / "masks_cache"
MASK_DIR.mkdir(exist_ok=True)

NEUTRAL_U8 = np.array([124, 116, 104], dtype=np.uint8)

def compute_and_save_mask(idx, img_path_str):
    try:
        img = np.array(Image.open(img_path_str).convert("RGB"))
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        bgd  = np.zeros((1, 65), np.float64)
        fgd  = np.zeros((1, 65), np.float64)
        rect = (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8))
        cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
        ratio = float(mask2.mean())
        if 0.05 < ratio < 0.95:
            np.save(MASK_DIR / f"mask_{idx:04d}.npy", mask2)
            return True
        return False
    except Exception as e:
        print(f"  Error on idx {idx}: {e}", flush=True)
        return False

def main():
    # Collect training image paths from CSV
    train_samples = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == "train":
                train_samples.append((PROJECT_ROOT / row["image_path"]).as_posix())

    n = len(train_samples)
    print(f"Pre-computing masks for {n} training images...", flush=True)
    t0 = time.time()
    accepted = rejected = 0

    for i, img_path in enumerate(train_samples):
        ok = compute_and_save_mask(i, img_path)
        if ok:
            accepted += 1
        else:
            rejected += 1
        if (i+1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed
            eta = (n - i - 1) / rate
            print(f"  {i+1}/{n} | accepted={accepted} rejected={rejected} | "
                  f"ETA={eta:.0f}s", flush=True)

    total = time.time() - t0
    print(f"\nDone! {accepted} masks saved to masks_cache/. "
          f"{rejected} rejected. Total time: {total:.1f}s", flush=True)

if __name__ == "__main__":
    main()
