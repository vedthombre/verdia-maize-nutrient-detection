"""
Maize Leaf Nutrient-Deficiency -- Test-Set Evaluation
======================================================
Loads the best EfficientNet-B0 checkpoint (selected by validation loss
during training) and evaluates it on the held-out TEST set.

This script is run ONLY after training is complete.  The test set was
never seen during training or checkpoint selection, so these metrics
are an unbiased estimate of generalisation performance.

Outputs
-------
* Classification report  (per-class precision / recall / F1, macro F1)
* Confusion matrix        (saved as an image)
* Training loss & accuracy curves (saved as images)
* All metrics saved to JSON

Usage
-----
    python evaluate.py
"""

import json
from pathlib import Path
from collections import Counter

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from dataset import MaizeLeafDataset, DISPLAY_NAMES
from transforms import test_transform
from model import build_efficientnet_b0

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parent
CSV_PATH        = PROJECT_ROOT / "dataset_split.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "efficientnet_b0_best.pth"
HISTORY_PATH    = PROJECT_ROOT / "results" / "training_history.json"
RESULTS_DIR     = PROJECT_ROOT / "results"
SEED            = 42
BATCH_SIZE      = 32
NUM_WORKERS     = 0


# ──────────────────────────────────────────────────────────────────────
# 1.  GENERATE TRAINING CURVES
# ──────────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, results_dir: Path):
    """
    Plot and save:
      * training vs validation loss
      * training vs validation accuracy
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    best_epoch = history.get("best_epoch", None)

    # ── Loss curve ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], "o-", label="Train loss", linewidth=2)
    ax.plot(epochs, history["val_loss"],   "s-", label="Val loss",   linewidth=2)
    if best_epoch:
        ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.6,
                   label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_path = results_dir / "loss_curve.png"
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"  Loss curve saved    : {loss_path}")

    # ── Accuracy curve ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [a * 100 for a in history["train_acc"]],
            "o-", label="Train accuracy", linewidth=2)
    ax.plot(epochs, [a * 100 for a in history["val_acc"]],
            "s-", label="Val accuracy", linewidth=2)
    if best_epoch:
        ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.6,
                   label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training vs Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    acc_path = results_dir / "accuracy_curve.png"
    fig.savefig(acc_path, dpi=150)
    plt.close(fig)
    print(f"  Accuracy curve saved: {acc_path}")


# ──────────────────────────────────────────────────────────────────────
# 2.  TEST-SET EVALUATION
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_test_set(model, loader, device):
    """
    Run the model on the test set and collect all predictions / labels.

    Returns
    -------
    all_labels : list[int]
    all_preds  : list[int]
    """
    model.eval()
    all_labels = []
    all_preds = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        all_labels.extend(labels.tolist())
        all_preds.extend(preds.cpu().tolist())

    return all_labels, all_preds


def print_and_save_metrics(all_labels, all_preds, results_dir: Path):
    """
    Compute and display:
      * per-class precision, recall, F1
      * macro F1
      * confusion matrix
      * per-class sample counts
    """
    num_classes = len(DISPLAY_NAMES)
    sep = "=" * 70

    # ── Per-class test sample counts ─────────────────────────────────
    label_counts = Counter(all_labels)
    print(f"\n{sep}")
    print("  TEST SET COMPOSITION")
    print(sep)
    for i in range(num_classes):
        print(f"  {DISPLAY_NAMES[i]:<16s}  {label_counts.get(i, 0):>3d} samples")
    print(f"  {'TOTAL':<16s}  {len(all_labels):>3d} samples")

    # ── Classification report ────────────────────────────────────────
    report_str = classification_report(
        all_labels, all_preds,
        target_names=DISPLAY_NAMES,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        all_labels, all_preds,
        target_names=DISPLAY_NAMES,
        output_dict=True,
        zero_division=0,
    )

    print(f"\n{sep}")
    print("  CLASSIFICATION REPORT")
    print(sep)
    print(report_str)

    # ── Highlight small-sample warning ───────────────────────────────
    potassium_count = label_counts.get(4, 0)
    if potassium_count <= 10:
        print(f"  NOTE: Potassium has only {potassium_count} test samples.")
        print("  Its per-class metrics may be unreliable due to small n.")
        print()

    # ── Overall test accuracy ────────────────────────────────────────
    correct = sum(1 for y, p in zip(all_labels, all_preds) if y == p)
    test_acc = correct / len(all_labels)
    macro_f1 = report_dict["macro avg"]["f1-score"]
    print(f"  Test Accuracy : {test_acc:.4f}  ({test_acc:.1%})")
    print(f"  Macro F1      : {macro_f1:.4f}")
    print()

    # ── Confusion matrix (plot) ──────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=DISPLAY_NAMES)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
    ax.set_title("Confusion Matrix -- EfficientNet-B0 Baseline (Test Set)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    cm_path = results_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved: {cm_path}")

    # ── Save metrics to JSON ─────────────────────────────────────────
    metrics = {
        "test_accuracy": test_acc,
        "macro_f1": macro_f1,
        "per_class": {},
        "confusion_matrix": cm.tolist(),
        "test_sample_counts": {DISPLAY_NAMES[i]: label_counts.get(i, 0)
                               for i in range(num_classes)},
    }
    for i, name in enumerate(DISPLAY_NAMES):
        metrics["per_class"][name] = {
            "precision": report_dict[name]["precision"],
            "recall":    report_dict[name]["recall"],
            "f1_score":  report_dict[name]["f1-score"],
            "support":   report_dict[name]["support"],
        }

    metrics_path = results_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics JSON saved  : {metrics_path}")

    return metrics


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Plot training curves ─────────────────────────────────────────
    print("Generating training curves ...")
    with open(HISTORY_PATH, "r") as f:
        history = json.load(f)
    plot_training_curves(history, RESULTS_DIR)
    print()

    # ── Load best checkpoint ─────────────────────────────────────────
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    model = build_efficientnet_b0(num_classes=5, freeze_backbone=True)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print("  Model loaded successfully.\n")

    # ── Load test set ────────────────────────────────────────────────
    # This is the ONLY place in the entire project where the test set
    # is loaded.  It was never seen during training or checkpoint
    # selection.
    test_ds = MaizeLeafDataset(CSV_PATH, "test", transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS,
    )
    print(f"Test samples: {len(test_ds)}")

    # ── Evaluate ─────────────────────────────────────────────────────
    all_labels, all_preds = evaluate_test_set(model, test_loader, device)
    metrics = print_and_save_metrics(all_labels, all_preds, RESULTS_DIR)

    print("\n" + "=" * 70)
    print("  Evaluation complete.")
    print("  REMINDER: This is the BASELINE result.")
    print("  Do not claim this is the best achievable performance.")
    print("=" * 70)


if __name__ == "__main__":
    main()
