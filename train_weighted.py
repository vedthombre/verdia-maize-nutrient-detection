"""
Maize Leaf Nutrient-Deficiency -- Experiment 3 (Class Weights)
===============================================================
Fine-tunes the EfficientNet-B0 model with class-weighted CrossEntropyLoss.

Key ML decisions:
  * Model Configuration : Unfreeze last 3 blocks (same as Exp 2)
  * Loss function       : CrossEntropyLoss with inverse-frequency weights
                          calculated strictly from the training set.
  * Optimiser           : Adam (Backbone: 1e-4, Classifier: 1e-3)
  * Batch size          : 32
  * Epochs              : 30 (early stopping, patience=7)
  * Checkpoint          : saved at lowest *validation loss*

Usage
-----
    python train_weighted.py
"""

import json
import time
import random
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MaizeLeafDataset, DISPLAY_NAMES
from transforms import train_transform, val_transform
from model import build_efficientnet_b0, count_parameters

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent
CSV_PATH       = PROJECT_ROOT / "dataset_split.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR    = PROJECT_ROOT / "results"

BATCH_SIZE     = 32
LR_BACKBONE    = 1e-4
LR_CLASSIFIER  = 1e-3
NUM_EPOCHS     = 30
PATIENCE       = 7
SEED           = 42
NUM_WORKERS    = 0
UNFREEZE_BLKS  = 3

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, loader, criterion, optimiser, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimiser.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)
    return running_loss / total, correct / total

def main():
    set_seed(SEED)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    train_ds = MaizeLeafDataset(CSV_PATH, "train", transform=train_transform)
    val_ds   = MaizeLeafDataset(CSV_PATH, "val",   transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # ── CALCULATE CLASS WEIGHTS ──────────────────────────────────────
    # Calculated EXCLUSIVELY from the training set to prevent leakage.
    # We use inverse-frequency weighting: weight_i = N / (C * n_i)
    # This formula naturally centres weights around 1.0.
    train_labels = [label for _, _, label in train_ds.samples]
    label_counts = Counter(train_labels)
    N = len(train_labels)
    C = len(DISPLAY_NAMES)
    
    class_weights = []
    print("Class Weights Calculated from Train Set:")
    for i in range(C):
        w = N / (C * label_counts[i])
        class_weights.append(w)
        print(f"  {DISPLAY_NAMES[i]:<15s}: count={label_counts[i]:<3d} -> weight={w:.4f}")
    print()
    
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Load model exactly as in Experiment 2
    model = build_efficientnet_b0(num_classes=5, freeze_backbone=False, unfreeze_blocks=UNFREEZE_BLKS)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Discriminative learning rates
    backbone_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "classifier" in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)

    optimiser = torch.optim.Adam([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": classifier_params, "lr": LR_CLASSIFIER}
    ])

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    checkpoint_path = CHECKPOINT_DIR / "efficientnet_b0_weighted_best.pth"

    print("=" * 70)
    print(f"  {'Epoch':>5s}  {'Train Loss':>10s}  {'Train Acc':>9s}  {'Val Loss':>10s}  {'Val Acc':>9s}  {'Status'}")
    print("=" * 70)

    start_time = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimiser, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            status = "<-- saved"
        else:
            patience_counter += 1
            status = ""

        print(f"  {epoch:>5d}  {train_loss:>10.4f}  {train_acc:>8.1%}  {val_loss:>10.4f}  {val_acc:>8.1%}  {status}")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).")
            break

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Best epoch        : {best_epoch}")
    print(f"Best val loss     : {best_val_loss:.4f}")
    print(f"Best val accuracy : {history['val_acc'][best_epoch - 1]:.1%}")
    print(f"Checkpoint saved  : {checkpoint_path}")

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss
    history_path = RESULTS_DIR / "weighted_training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved     : {history_path}")

if __name__ == "__main__":
    main()
