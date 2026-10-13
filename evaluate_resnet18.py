"""
Maize Leaf Nutrient-Deficiency -- ResNet18 Test-Set Evaluation
==============================================================
Evaluates the trained ResNet18 comparison checkpoint on the held-out test set.

Loads the checkpoint saved by train_resnet18.py and produces per-class
and aggregate metrics consistent with all other comparison evaluations.

Outputs
-------
results/resnet18_comparison_test_metrics.json
results/resnet18_confusion_matrix.png
results/resnet18_comparison_test_predictions.csv

Usage
-----
    python evaluate_resnet18.py
"""

import csv
import json
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from dataset import MaizeLeafDataset, CLASS_NAMES
from transforms import test_transform
from train_resnet18 import build_resnet18, NUM_CLASSES

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parent
CSV_PATH        = PROJECT_ROOT / "dataset_split.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "resnet18_comparison_best.pth"
RESULTS_DIR     = PROJECT_ROOT / "results"
BATCH_SIZE      = 32
NUM_WORKERS     = 0
SEED            = 42


# ──────────────────────────────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, loader, device):
    """Run forward pass on all test batches; return (labels, preds)."""
    model.eval()
    all_labels = []
    all_preds  = []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        all_labels.extend(labels.tolist())
        all_preds.extend(preds.cpu().tolist())
    return all_labels, all_preds


# ──────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────
def compute_and_save_metrics(all_labels, all_preds,
                             inference_time_s: float,
                             results_dir: Path):
    n = len(all_labels)
    correct   = sum(1 for y, p in zip(all_labels, all_preds) if y == p)
    test_acc  = correct / n

    report_str  = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES, digits=4, zero_division=0
    )
    report_dict = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )

    print("\n" + "="*70)
    print("  CLASSIFICATION REPORT — ResNet18 (Test Set)")
    print("="*70)
    print(report_str)

    macro_f1    = report_dict["macro avg"]["f1-score"]
    weighted_f1 = report_dict["weighted avg"]["f1-score"]
    macro_prec  = report_dict["macro avg"]["precision"]
    macro_rec   = report_dict["macro avg"]["recall"]

    print(f"  Test Accuracy   : {test_acc:.4f}  ({test_acc:.1%})")
    print(f"  Correct / Total : {correct} / {n}")
    print(f"  Macro F1        : {macro_f1:.4f}")
    print(f"  Weighted F1     : {weighted_f1:.4f}")
    print(f"  Macro Precision : {macro_prec:.4f}")
    print(f"  Macro Recall    : {macro_rec:.4f}")
    print(f"  Inference time  : {inference_time_s:.3f}s "
          f"({inference_time_s / n * 1000:.2f} ms/sample)")

    # Confusion matrix figure
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
    ax.set_title("Confusion Matrix — ResNet18 (Test Set)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    cm_path = results_dir / "resnet18_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"\n  Confusion matrix saved: {cm_path}")

    # JSON metrics
    metrics = {
        "model": "ResNet18",
        "checkpoint": "checkpoints/resnet18_comparison_best.pth",
        "test_accuracy":        test_acc,
        "correct_predictions":  correct,
        "total_predictions":    n,
        "macro_precision":      macro_prec,
        "macro_recall":         macro_rec,
        "macro_f1":             macro_f1,
        "weighted_f1":          weighted_f1,
        "inference_time_seconds": inference_time_s,
        "inference_ms_per_sample": inference_time_s / n * 1000,
        "per_class": {},
        "confusion_matrix": cm.tolist(),
        "test_sample_counts": {
            CLASS_NAMES[i]: int(Counter(all_labels).get(i, 0))
            for i in range(NUM_CLASSES)
        },
    }
    for name in CLASS_NAMES:
        metrics["per_class"][name] = {
            "precision": report_dict[name]["precision"],
            "recall":    report_dict[name]["recall"],
            "f1_score":  report_dict[name]["f1-score"],
            "support":   report_dict[name]["support"],
        }

    metrics_path = results_dir / "resnet18_comparison_test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics JSON saved  : {metrics_path}")

    # CSV predictions
    preds_path = results_dir / "resnet18_comparison_test_predictions.csv"
    with open(preds_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "true_label", "true_class",
                         "pred_label", "pred_class", "correct"])
        for i, (y, p) in enumerate(zip(all_labels, all_preds)):
            writer.writerow([
                i, y, CLASS_NAMES[y],
                p, CLASS_NAMES[p],
                int(y == p),
            ])
    print(f"  Predictions CSV     : {preds_path}")

    return metrics


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print("  ResNet18 Evaluation — Algorithm Comparison (Verdia)")
    print(f"  Checkpoint : {CHECKPOINT_PATH}")
    print(f"  Device     : {device}")
    print(f"{'='*70}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"ResNet18 checkpoint not found: {CHECKPOINT_PATH}\n"
            "Please run train_resnet18.py first."
        )

    # Load model
    model = build_resnet18(num_classes=NUM_CLASSES)
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()

    # Test dataset (deterministic transform, identical to EfficientNet eval)
    test_ds     = MaizeLeafDataset(CSV_PATH, "test", transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    print(f"\n  Test samples : {len(test_ds)}")

    # Inference timing
    t0 = time.time()
    all_labels, all_preds = run_inference(model, test_loader, device)
    inference_time = time.time() - t0

    compute_and_save_metrics(all_labels, all_preds,
                             inference_time, RESULTS_DIR)

    print(f"\n{'='*70}")
    print("  ResNet18 evaluation complete.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
