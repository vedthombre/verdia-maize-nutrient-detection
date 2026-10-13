import os
import cv2
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from dataset import MaizeLeafDataset, CLASS_NAMES
from transforms import test_transform
from train_resnet18 import build_resnet18, NUM_CLASSES

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "dataset_split.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def create_leaf_mask(img_np):
    img_uint8 = np.uint8(img_np * 255)
    mask = np.zeros(img_uint8.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    h, w = img_uint8.shape[:2]
    rect = (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8))
    cv2.grabCut(img_uint8, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    return mask2

def run_evaluation(exp_name, checkpoint_path, device):
    print(f"\\nEvaluating {exp_name}...")
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    model = build_resnet18(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    test_ds = MaizeLeafDataset(CSV_PATH, "test", transform=test_transform)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    all_labels = []
    all_preds_orig = []
    
    # For background intervention
    flips_corr_incorr = 0
    
    for i in range(len(test_ds)):
        img_tensor, true_label = test_ds[i]
        
        # Original prediction
        with torch.no_grad():
            inp = img_tensor.unsqueeze(0).to(device)
            out_orig = model(inp)
            pred_orig = out_orig.argmax(dim=1).item()
            
        all_labels.append(true_label)
        all_preds_orig.append(pred_orig)
        
        # Intervention prediction
        img_np = img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        grabcut_mask = create_leaf_mask(img_np)
        masked_neutral_np = np.copy(img_np)
        masked_neutral_np[grabcut_mask == 0] = mean
        
        masked_neutral_tensor = torch.tensor(((masked_neutral_np - mean) / std).transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            out_neutral = model(masked_neutral_tensor)
            pred_neutral = out_neutral.argmax(dim=1).item()
            
        if pred_orig == true_label and pred_neutral != true_label:
            flips_corr_incorr += 1

    # Metrics
    n = len(all_labels)
    correct = sum(1 for y, p in zip(all_labels, all_preds_orig) if y == p)
    test_acc = correct / n
    
    report_dict = classification_report(all_labels, all_preds_orig, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    
    cm = confusion_matrix(all_labels, all_preds_orig)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
    ax.set_title(f"Confusion Matrix — {exp_name}")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    cm_path = RESULTS_DIR / f"{exp_name}_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    metrics = {
        "exp_name": exp_name,
        "test_accuracy": test_acc,
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "weighted_f1": report_dict["weighted avg"]["f1-score"],
        "per_class": {name: report_dict[name] for name in CLASS_NAMES},
        "background_flips": flips_corr_incorr
    }
    
    with open(RESULTS_DIR / f"{exp_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Results for {exp_name}:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Background flips (Correct->Incorrect): {flips_corr_incorr}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    run_evaluation("ExpA_Baseline", PROJECT_ROOT / "checkpoints" / "expA_baseline_best.pth", device)
    run_evaluation("ExpB_Augmented", PROJECT_ROOT / "checkpoints" / "expB_augmented_best.pth", device)

if __name__ == "__main__":
    main()
