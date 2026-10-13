"""
Maize Leaf Nutrient-Deficiency -- ResNet18 Training (Algorithm Comparison)
===========================================================================
Trains a ResNet18 model for the 5-class nutrient deficiency classification.

Architecture
------------
* Base: torchvision ResNet18, pretrained on ImageNet (IMAGENET1K_V1 weights).
* The final fully-connected layer (Linear(512, 1000)) is replaced with
  Linear(512, 5) to match our 5 output classes.
* The entire network is fine-tuned (no layers are frozen).
* Discriminative learning rates are applied: the backbone uses a lower LR
  and the new classification head uses a higher LR.

Split
-----
Uses the existing dataset_split.csv unchanged (train=324, val=69, test=70).
The existing EfficientNet-B0 implementation (model.py) is NOT modified.

Transforms
----------
* Training : project's train_transform (augmentation + normalization)
* Validation: project's val_transform (deterministic, no augmentation)
* Test       : project's test_transform (deterministic, no augmentation)
  All transforms use ImageNet mean/std as defined in transforms.py.

Checkpoint selection
---------------------
The checkpoint with the LOWEST validation loss is saved as the best model.
This checkpoint is used for evaluation and comparison.

Reproducibility
---------------
SEED=42 is set for Python, NumPy, and PyTorch at the start.
torch.backends.cudnn.deterministic=True and benchmark=False.

Outputs
-------
checkpoints/resnet18_comparison_best.pth       -- best checkpoint
results/resnet18_comparison_training_history.json -- epoch history
results/resnet18_comparison_training_config.json  -- full training config

Usage
-----
    python train_resnet18.py
"""

import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights

from dataset import MaizeLeafDataset, CLASS_NAMES
from transforms import train_transform, val_transform

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent
CSV_PATH       = PROJECT_ROOT / "dataset_split.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR    = PROJECT_ROOT / "results"

SEED           = 42
BATCH_SIZE     = 32
NUM_EPOCHS     = 40
PATIENCE       = 8          # early-stopping patience (epochs)
LR_BACKBONE    = 5e-5       # lower LR for pretrained backbone layers
LR_HEAD        = 1e-3       # higher LR for the new classification head
WEIGHT_DECAY   = 1e-4
NUM_WORKERS    = 0
NUM_CLASSES    = 5

CHECKPOINT_PATH  = CHECKPOINT_DIR / "resnet18_comparison_best.pth"
HISTORY_PATH     = RESULTS_DIR    / "resnet18_comparison_training_history.json"
CONFIG_PATH      = RESULTS_DIR    / "resnet18_comparison_training_config.json"


# ──────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────────────
def build_resnet18(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Load ImageNet-pretrained ResNet18 and replace the final FC layer
    with a new Linear(512, num_classes) head.

    The entire network is trainable.  Discriminative learning rates are
    applied externally (backbone_params vs head_params).
    """
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features   # 512 for ResNet18
    model.fc = nn.Linear(in_features, num_classes)

    return model


def count_parameters(model: nn.Module):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ──────────────────────────────────────────────────────────────────────
# TRAINING HELPERS
# ──────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimiser, device):
    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimiser.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimiser.step()
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct  += preds.eq(labels).sum().item()
        total    += images.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct  += preds.eq(labels).sum().item()
        total    += images.size(0)
    return running_loss / total, correct / total


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────
def main():
    set_seed(SEED)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print("  ResNet18 Training — Algorithm Comparison (Verdia)")
    print(f"  Device : {device}")
    print(f"{'='*70}\n")

    # ── Datasets and loaders ──────────────────────────────────────────
    train_ds = MaizeLeafDataset(CSV_PATH, "train", transform=train_transform)
    val_ds   = MaizeLeafDataset(CSV_PATH, "val",   transform=val_transform)
    test_ds  = MaizeLeafDataset(CSV_PATH, "test",  transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    # ── Model ─────────────────────────────────────────────────────────
    model = build_resnet18(num_classes=NUM_CLASSES)
    model = model.to(device)
    total_params, trainable_params = count_parameters(model)

    print("Model: ResNet18 (ImageNet pretrained, full fine-tune)")
    print(f"  Total parameters     : {total_params:,}")
    print(f"  Trainable parameters : {trainable_params:,}")
    print(f"  New FC head          : Linear(512, {NUM_CLASSES})")

    # ── Dataset sizes ─────────────────────────────────────────────────
    print(f"\nDataset split:")
    print(f"  Train      : {len(train_ds)}")
    print(f"  Validation : {len(val_ds)}")
    print(f"  Test       : {len(test_ds)}")

    # ── Training config ───────────────────────────────────────────────
    print(f"\nTraining configuration:")
    print(f"  SEED           : {SEED}")
    print(f"  BATCH_SIZE     : {BATCH_SIZE}")
    print(f"  NUM_EPOCHS     : {NUM_EPOCHS}")
    print(f"  PATIENCE       : {PATIENCE}")
    print(f"  LR_BACKBONE    : {LR_BACKBONE}")
    print(f"  LR_HEAD        : {LR_HEAD}")
    print(f"  WEIGHT_DECAY   : {WEIGHT_DECAY}")
    print(f"  Criterion      : CrossEntropyLoss (no class weights)")

    # ── Optimiser (discriminative LR) ─────────────────────────────────
    # Group backbone params (all layers except fc) and head params (fc only).
    head_params     = list(model.fc.parameters())
    head_param_ids  = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters()
                       if id(p) not in head_param_ids]

    optimiser = torch.optim.Adam([
        {"params": backbone_params, "lr": LR_BACKBONE,
         "weight_decay": WEIGHT_DECAY},
        {"params": head_params,     "lr": LR_HEAD,
         "weight_decay": WEIGHT_DECAY},
    ])

    criterion = nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────────────────────────────
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }
    best_val_loss    = float("inf")
    best_epoch       = -1
    patience_counter = 0

    print(f"\n{'='*70}")
    print(f"  {'Epoch':>5s}  {'TrainLoss':>9s}  {'TrainAcc':>8s}  "
          f"{'ValLoss':>9s}  {'ValAcc':>8s}  Status")
    print(f"{'='*70}")

    wall_start = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimiser, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_epoch       = epoch
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            status = "<-- saved"
        else:
            patience_counter += 1
            status = ""

        print(f"  {epoch:>5d}  {train_loss:>9.4f}  {train_acc:>7.1%}  "
              f"{val_loss:>9.4f}  {val_acc:>7.1%}  {status}")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs).")
            break

    elapsed = time.time() - wall_start
    print(f"{'='*70}")
    print(f"\nTraining complete in {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"Best epoch     : {best_epoch}")
    print(f"Best val loss  : {best_val_loss:.4f}")
    print(f"Best val acc   : {history['val_acc'][best_epoch - 1]:.1%}")
    print(f"Checkpoint     : {CHECKPOINT_PATH}")

    # ── Save history and config ────────────────────────────────────────
    history["best_epoch"]    = best_epoch
    history["best_val_loss"] = best_val_loss
    history["total_train_time_seconds"] = elapsed

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved  : {HISTORY_PATH}")

    config = {
        "model":         "ResNet18",
        "pretrained":    "IMAGENET1K_V1",
        "num_classes":   NUM_CLASSES,
        "class_names":   CLASS_NAMES,
        "freeze_backbone": False,
        "seed":          SEED,
        "batch_size":    BATCH_SIZE,
        "num_epochs_max": NUM_EPOCHS,
        "patience":      PATIENCE,
        "lr_backbone":   LR_BACKBONE,
        "lr_head":       LR_HEAD,
        "weight_decay":  WEIGHT_DECAY,
        "criterion":     "CrossEntropyLoss (no class weights)",
        "optimiser":     "Adam",
        "train_transform": (
            "RandomResizedCrop(224, scale=(0.8,1.0), BILINEAR), "
            "RandomHorizontalFlip(p=0.5), RandomRotation(15), "
            "ColorJitter(b=0.15, c=0.15, s=0.10, h=0.02), "
            "ToTensor, Normalize(ImageNet)"
        ),
        "val_transform":  (
            "Resize(256, BILINEAR), CenterCrop(224), "
            "ToTensor, Normalize(ImageNet)"
        ),
        "best_epoch":    best_epoch,
        "best_val_loss": best_val_loss,
        "total_train_time_seconds": elapsed,
        "device": str(device),
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved   : {CONFIG_PATH}\n")


if __name__ == "__main__":
    main()
