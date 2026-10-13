"""
Maize Leaf Nutrient-Deficiency -- Final Model Verification
=================================================================
Evaluates the best Experiment 5 checkpoint on the exact 70-image test set
to ensure reproducibility.
"""

import torch
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

from dataset import MaizeLeafDataset, DISPLAY_NAMES
from transforms import test_transform
from model import build_efficientnet_b0

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parent
CSV_PATH        = PROJECT_ROOT / "dataset_split.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "efficientnet_b0_deeper_finetuned_best.pth"
SEED            = 42
BATCH_SIZE      = 32
NUM_WORKERS     = 0
UNFREEZE_BLKS   = 5

def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_efficientnet_b0(num_classes=5, freeze_backbone=False, unfreeze_blocks=UNFREEZE_BLKS)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    test_ds = MaizeLeafDataset(CSV_PATH, "test", transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())

    num_test_images = len(all_labels)
    correct = sum(1 for y, p in zip(all_labels, all_preds) if y == p)
    incorrect = num_test_images - correct
    test_acc = correct / num_test_images

    report_str = classification_report(all_labels, all_preds, target_names=DISPLAY_NAMES, digits=4, zero_division=0)
    report_dict = classification_report(all_labels, all_preds, target_names=DISPLAY_NAMES, output_dict=True, zero_division=0)
    
    macro_f1 = report_dict["macro avg"]["f1-score"]
    weighted_f1 = report_dict["weighted avg"]["f1-score"]
    cm = confusion_matrix(all_labels, all_preds)

    print("==================================================")
    print("FINAL MODEL VERIFICATION (Experiment 5)")
    print("==================================================")
    print(f"Class Order verified : {DISPLAY_NAMES}")
    print(f"Number of test images: {num_test_images}")
    print(f"Correct predictions  : {correct}")
    print(f"Incorrect predictions: {incorrect}")
    print(f"Test Accuracy        : {test_acc:.4f} ({test_acc:.2%})")
    print(f"Macro F1             : {macro_f1:.4f}")
    print(f"Weighted F1          : {weighted_f1:.4f}\n")
    
    print(report_str)

    print("\nConfusion Matrix:")
    print(cm)

if __name__ == "__main__":
    main()
