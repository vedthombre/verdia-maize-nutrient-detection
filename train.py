"""
Maize Leaf Nutrient-Deficiency -- EfficientNet-B0 Baseline Training
====================================================================
Trains the frozen-backbone EfficientNet-B0 on the maize leaf dataset.

Key ML decisions documented inline:
  * Loss function    : CrossEntropyLoss (no class weights -- clean baseline)
  * Optimiser        : Adam, lr=1e-3
  * Batch size       : 32
  * Epochs           : 30  (with early stopping, patience=7)
  * Checkpoint       : saved at lowest *validation loss*
  * Test-set leakage : test data is NEVER loaded here

Usage
-----
    python train.py
"""

import json
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MaizeLeafDataset, DISPLAY_NAMES
from transforms import train_transform, val_transform
from model import build_efficientnet_b0, count_parameters

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent
CSV_PATH      = PROJECT_ROOT / "dataset_split.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR    = PROJECT_ROOT / "results"

# Hyperparameters -- rationale:
#
# BATCH_SIZE = 32
#   324 train images / 32 = ~10 batches per epoch.  Small enough to
#   fit in memory (even on CPU), large enough for stable gradient
#   estimates.  Standard starting point for small-dataset transfer
#   learning.
#
# LEARNING_RATE = 1e-3
#   Standard Adam default.  Appropriate here because only the
#   classifier head (~6.4 K params) is being trained from random
#   initialisation.  The frozen backbone means there is no risk of
#   disrupting pretrained features with a "too-large" LR.
#
# NUM_EPOCHS = 30
#   Generous upper bound.  With a frozen backbone and small head,
#   convergence is fast.  Early stopping will terminate sooner if
#   validation loss plateaus.
#
# PATIENCE = 7
#   Number of epochs without validation-loss improvement before
#   stopping.  7 is a moderate value that allows recovery from
#   transient fluctuations without wasting compute.

BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS    = 30
PATIENCE      = 7
SEED          = 42
NUM_WORKERS   = 0   # safe default on Windows; avoids multiprocessing issues


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms can be slower but ensure reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────
# TRAINING & VALIDATION LOOPS
# ──────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimiser, device):
    """Run one training epoch.  Returns (avg_loss, accuracy)."""
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

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Run validation.  Returns (avg_loss, accuracy)."""
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

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ── Datasets ─────────────────────────────────────────────────────
    # NOTE: We deliberately do NOT load the test set here.
    # Test evaluation happens only in evaluate.py, after training is
    # complete and the best checkpoint has been selected by validation
    # performance.  This prevents test-set leakage.

    train_ds = MaizeLeafDataset(CSV_PATH, "train", transform=train_transform)
    val_ds   = MaizeLeafDataset(CSV_PATH, "val",   transform=val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )

    print(f"Train samples : {len(train_ds)}")
    print(f"Val samples   : {len(val_ds)}")
    print(f"Batch size    : {BATCH_SIZE}")
    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")
    print()

    # ── Model ────────────────────────────────────────────────────────
    model = build_efficientnet_b0(num_classes=5, freeze_backbone=True)
    model = model.to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters   : {total_params - trainable_params:,}")
    print()

    # ── Loss ─────────────────────────────────────────────────────────
    # CrossEntropyLoss is the standard loss for multi-class single-
    # label classification.  It combines LogSoftmax + NLLLoss and
    # expects raw logits.
    # No class weights for this baseline -- we want to see the
    # unweighted performance first.
    criterion = nn.CrossEntropyLoss()

    # ── Optimiser ────────────────────────────────────────────────────
    # Adam with lr=1e-3.  We only pass the classifier parameters
    # (the only trainable ones) to avoid unnecessary overhead.
    optimiser = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    # ── Training loop with early stopping ────────────────────────────
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [],   "val_acc": [],
    }

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    checkpoint_path = CHECKPOINT_DIR / "efficientnet_b0_best.pth"

    print("=" * 70)
    print(f"  {'Epoch':>5s}  {'Train Loss':>10s}  {'Train Acc':>9s}  "
          f"{'Val Loss':>10s}  {'Val Acc':>9s}  {'Status'}")
    print("=" * 70)

    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimiser, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # ── Checkpoint selection ─────────────────────────────────────
        # We save the model when validation loss improves.
        # Validation loss (rather than accuracy) is used because it is
        # a smoother signal, especially with only 69 val samples where
        # a single misclassification changes accuracy by ~1.4 %.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            status = "<-- saved"
        else:
            patience_counter += 1
            status = ""

        print(f"  {epoch:>5d}  {train_loss:>10.4f}  {train_acc:>8.1%}  "
              f"{val_loss:>10.4f}  {val_acc:>8.1%}  {status}")

        # ── Early stopping ───────────────────────────────────────────
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs).")
            break

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Best epoch        : {best_epoch}")
    print(f"Best val loss     : {best_val_loss:.4f}")
    print(f"Best val accuracy : {history['val_acc'][best_epoch - 1]:.1%}")
    print(f"Checkpoint saved  : {checkpoint_path}")

    # ── Save training history ────────────────────────────────────────
    history_path = RESULTS_DIR / "training_history.json"
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved     : {history_path}")
    print()

    # ── Diagnostic: overfitting check ────────────────────────────────
    final_train_acc = history["train_acc"][-1]
    final_val_acc = history["val_acc"][best_epoch - 1]
    gap = final_train_acc - final_val_acc

    if gap > 0.15:
        print("WARNING: Large train-val accuracy gap ({:.1%}).".format(gap))
        print("  This suggests overfitting.  Consider:")
        print("  - Adding regularisation (higher dropout, weight decay)")
        print("  - Using class weights or oversampling")
        print("  - Unfreezing some backbone layers with a small LR")
    elif gap > 0.05:
        print("NOTE: Moderate train-val gap ({:.1%}).".format(gap))
        print("  Monitor this if adding more epochs or unfreezing layers.")
    else:
        print("Train-val gap is small ({:.1%}) -- no obvious overfitting.".format(gap))

    print("\nNext step: run evaluate.py to compute test-set metrics.")


if __name__ == "__main__":
    main()
