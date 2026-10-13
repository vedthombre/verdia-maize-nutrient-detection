"""
Maize Leaf Nutrient-Deficiency -- Fine-Tuned Test-Set Evaluation
=================================================================
Evaluates the best fine-tuned EfficientNet-B0 checkpoint on the 
held-out test set. Saves output metrics with the `finetuned_` prefix
to avoid overwriting the baseline results.
"""

import json
from pathlib import Path
from collections import Counter

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from dataset import MaizeLeafDataset, DISPLAY_NAMES
from transforms import test_transform
from model import build_efficientnet_b0

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parent
CSV_PATH        = PROJECT_ROOT / "dataset_split.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "efficientnet_b0_finetuned_best.pth"
HISTORY_PATH    = PROJECT_ROOT / "results" / "finetuned_training_history.json"
RESULTS_DIR     = PROJECT_ROOT / "results"
SEED            = 42
BATCH_SIZE      = 32
NUM_WORKERS     = 0
UNFREEZE_BLKS   = 3

def plot_training_curves(history: dict, results_dir: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    best_epoch = history.get("best_epoch", None)

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], "o-", label="Train loss", linewidth=2)
    ax.plot(epochs, history["val_loss"],   "s-", label="Val loss",   linewidth=2)
    if best_epoch:
        ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.6, label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Fine-tuned: Training vs Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_path = results_dir / "finetuned_loss_curve.png"
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)

    # Accuracy curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Train accuracy", linewidth=2)
    ax.plot(epochs, [a * 100 for a in history["val_acc"]], "s-", label="Val accuracy", linewidth=2)
    if best_epoch:
        ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.6, label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Fine-tuned: Training vs Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    acc_path = results_dir / "finetuned_accuracy_curve.png"
    fig.savefig(acc_path, dpi=150)
    plt.close(fig)

@torch.no_grad()
def evaluate_test_set(model, loader, device):
    model.eval()
    all_labels, all_preds = [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        all_labels.extend(labels.tolist())
        all_preds.extend(preds.cpu().tolist())
    return all_labels, all_preds

def print_and_save_metrics(all_labels, all_preds, results_dir: Path):
    num_classes = len(DISPLAY_NAMES)
    
    report_str = classification_report(all_labels, all_preds, target_names=DISPLAY_NAMES, digits=4, zero_division=0)
    report_dict = classification_report(all_labels, all_preds, target_names=DISPLAY_NAMES, output_dict=True, zero_division=0)

    print("\n======================================================================")
    print("  CLASSIFICATION REPORT (FINE-TUNED)")
    print("======================================================================")
    print(report_str)

    correct = sum(1 for y, p in zip(all_labels, all_preds) if y == p)
    test_acc = correct / len(all_labels)
    macro_f1 = report_dict["macro avg"]["f1-score"]
    print(f"  Test Accuracy : {test_acc:.4f}  ({test_acc:.1%})")
    print(f"  Macro F1      : {macro_f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=DISPLAY_NAMES)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
    ax.set_title("Confusion Matrix -- Fine-Tuned (Test Set)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    cm_path = results_dir / "finetuned_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    metrics = {
        "test_accuracy": test_acc,
        "macro_f1": macro_f1,
        "per_class": {},
        "confusion_matrix": cm.tolist(),
        "test_sample_counts": {DISPLAY_NAMES[i]: Counter(all_labels).get(i, 0) for i in range(num_classes)},
    }
    for name in DISPLAY_NAMES:
        metrics["per_class"][name] = {
            "precision": report_dict[name]["precision"],
            "recall":    report_dict[name]["recall"],
            "f1_score":  report_dict[name]["f1-score"],
            "support":   report_dict[name]["support"],
        }
    metrics_path = results_dir / "finetuned_test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

def main():
    torch.manual_seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(HISTORY_PATH, "r") as f:
        history = json.load(f)
    plot_training_curves(history, RESULTS_DIR)

    model = build_efficientnet_b0(num_classes=5, freeze_backbone=False, unfreeze_blocks=UNFREEZE_BLKS)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    test_ds = MaizeLeafDataset(CSV_PATH, "test", transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_labels, all_preds = evaluate_test_set(model, test_loader, device)
    print_and_save_metrics(all_labels, all_preds, RESULTS_DIR)

if __name__ == "__main__":
    main()
