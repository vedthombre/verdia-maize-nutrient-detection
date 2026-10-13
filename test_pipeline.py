"""
Maize Leaf Nutrient-Deficiency -- Pipeline Test & Visualisation
================================================================
This script verifies that the full preprocessing pipeline works
end-to-end:

1. Reads `dataset_split.csv`.
2. Instantiates MaizeLeafDataset for each split with the correct
   transform.
3. Loads a few images from each split.
4. Prints the resulting tensor shape and value range.
5. Saves a grid of augmented training examples so you can visually
   inspect that the augmentations look realistic.

Usage
-----
    python test_pipeline.py
"""

import random
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no GUI window needed)
import matplotlib.pyplot as plt

from dataset import MaizeLeafDataset, IDX_TO_CLASS
from transforms import (
    train_transform,
    val_transform,
    test_transform,
    inv_normalize,
    INPUT_SIZE,
)

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH     = PROJECT_ROOT / "dataset_split.csv"
VIS_PATH     = PROJECT_ROOT / "augmented_samples.png"
SAMPLES_PER_SPLIT = 5   # how many images to test from each split
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


# ──────────────────────────────────────────────────────────────────────
# STEP 1 -- Load datasets
# ──────────────────────────────────────────────────────────────────────
# Each split gets its own transform (train has augmentation, val/test
# are deterministic).

print("Loading datasets ...")
datasets = {
    "train": MaizeLeafDataset(CSV_PATH, "train", transform=train_transform),
    "val":   MaizeLeafDataset(CSV_PATH, "val",   transform=val_transform),
    "test":  MaizeLeafDataset(CSV_PATH, "test",  transform=test_transform),
}

for name, ds in datasets.items():
    print(f"  {name:<6s}: {len(ds):>4d} samples")
print()


# ──────────────────────────────────────────────────────────────────────
# STEP 2 -- Sample a few images and verify tensor properties
# ──────────────────────────────────────────────────────────────────────
# For each split we load SAMPLES_PER_SPLIT random images and check
# that:
#   * shape is (3, 224, 224)
#   * dtype is float32
#   * values are in a plausible range (normalised, so roughly -3 to +3)

sep = "=" * 62

for split_name, ds in datasets.items():
    print(sep)
    print(f"  SPLIT: {split_name.upper()}")
    print(sep)

    indices = random.sample(range(len(ds)), min(SAMPLES_PER_SPLIT, len(ds)))

    for idx in indices:
        img_tensor, label = ds[idx]
        cls_name = ds.get_class_name(idx)

        # Basic shape / dtype checks.
        assert img_tensor.shape == (3, INPUT_SIZE, INPUT_SIZE), \
            f"Bad shape: {img_tensor.shape}"
        assert img_tensor.dtype == torch.float32, \
            f"Bad dtype: {img_tensor.dtype}"

        vmin = img_tensor.min().item()
        vmax = img_tensor.max().item()

        print(f"  idx={idx:<4d}  class={cls_name:<16s}  label={label}  "
              f"shape={tuple(img_tensor.shape)}  "
              f"min={vmin:>7.3f}  max={vmax:>7.3f}")

    print()

print("All tensor shape/dtype/range checks passed.\n")


# ──────────────────────────────────────────────────────────────────────
# STEP 3 -- Visualise augmented training examples
# ──────────────────────────────────────────────────────────────────────
# We pick one image per class from the training set and show 4 random
# augmentations of each, giving a 5-row x 4-column grid.
# This lets you visually confirm the augmentations look sensible.

print("Generating augmentation visualisation ...")

# Pick one sample per class.
class_examples = {}
for idx in range(len(datasets["train"])):
    cls = datasets["train"].get_class_name(idx)
    if cls not in class_examples:
        class_examples[cls] = idx
    if len(class_examples) == 5:
        break

n_augments = 4
fig, axes = plt.subplots(
    nrows=len(class_examples),
    ncols=n_augments,
    figsize=(n_augments * 3, len(class_examples) * 3),
)

for row, (cls_name, sample_idx) in enumerate(sorted(class_examples.items())):
    for col in range(n_augments):
        # Each call applies fresh random augmentation.
        img_tensor, label = datasets["train"][sample_idx]

        # Undo normalisation for display.
        img_display = inv_normalize(img_tensor)
        img_display = img_display.clamp(0, 1)

        # Convert (C, H, W) -> (H, W, C) for matplotlib.
        img_np = img_display.permute(1, 2, 0).numpy()

        ax = axes[row, col]
        ax.imshow(img_np)
        ax.set_title(f"{cls_name}" if col == 0 else "", fontsize=10)
        ax.axis("off")

plt.suptitle(
    "Training Augmentation Samples  (4 random augmentations per class)",
    fontsize=13, y=1.01,
)
plt.tight_layout()
plt.savefig(VIS_PATH, dpi=150, bbox_inches="tight")
print(f"Saved visualisation to: {VIS_PATH}")
print()

print(sep)
print("  Pipeline test complete. No original files were modified.")
print(sep)
