"""
Maize Leaf Nutrient-Deficiency -- PyTorch Dataset
==================================================
A lightweight `torch.utils.data.Dataset` that reads the split CSV
produced by `split_dataset.py` and applies the appropriate transform
from `transforms.py`.

This class does NOT copy, move, or modify any original image files.
Images are loaded on-the-fly from their original locations.
"""

import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

# ──────────────────────────────────────────────────────────────────────
# CLASS LABEL MAPPING
# ──────────────────────────────────────────────────────────────────────
# Explicit class-to-index mapping.  Matches the corrected folder names
# on disk and the class_name column in dataset_split.csv.
CLASS_NAMES = [
    "Healthy",       # 0
    "Magnesium",     # 1
    "Nitrogen",      # 2
    "Phosphorus",    # 3
    "Potassium",     # 4
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

# Display names for user-facing reports (identical to CLASS_NAMES now
# that the folders have been renamed to the correct spellings).
DISPLAY_NAMES = list(CLASS_NAMES)
IDX_TO_DISPLAY = dict(enumerate(DISPLAY_NAMES))
FOLDER_TO_DISPLAY = {n: n for n in CLASS_NAMES}


class MaizeLeafDataset(Dataset):
    """
    PyTorch Dataset for the maize leaf nutrient-deficiency images.

    Parameters
    ----------
    csv_path   : Path to `dataset_split.csv`.
    split      : One of 'train', 'val', or 'test'.
    transform  : A torchvision transform to apply to each image.
    root       : Project root directory that contains the `Dataset/`
                 folder.  Defaults to the parent of this script.

    Each sample is returned as:
        (image_tensor, label_index)
    where `label_index` is an integer in [0, 4].
    """

    def __init__(self, csv_path, split, transform=None, root=None):
        self.transform = transform
        self.root = Path(root) if root else Path(__file__).resolve().parent

        # Read the CSV and keep only rows for the requested split.
        self.samples = []  # list of (image_path, class_name, label_idx)
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    img_path = self.root / row["image_path"]
                    cls_name = row["class_name"]
                    label = CLASS_TO_IDX[cls_name]
                    self.samples.append((img_path, cls_name, label))

        self.split = split

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, _, label = self.samples[idx]

        # Open image and ensure RGB (some JPEGs may be grayscale or RGBA).
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def get_class_name(self, idx):
        """Return the class name string for sample at position `idx`."""
        _, cls_name, _ = self.samples[idx]
        return cls_name
