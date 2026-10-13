import json
from pathlib import Path
from collections import Counter
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from dataset import MaizeLeafDataset, DISPLAY_NAMES
from transforms import test_transform
from model import build_efficientnet_b0

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "dataset_split.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "efficientnet_b0_best.pth"
SEED = 42

def main():
    # 1. Determinism
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Checkpoint Info
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    model = build_efficientnet_b0(num_classes=5, freeze_backbone=True)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # 3. Test Dataset
    test_ds = MaizeLeafDataset(CSV_PATH, "test", transform=test_transform)
    print(f"Test split uses CSV: {CSV_PATH}")
    print(f"Test transform: {test_transform}")
    
    loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)
    
    all_labels = []
    all_preds = []
    correct_count = 0
    
    print("\n==================================================")
    print("image_path | true_class | predicted_class | correct/incorrect")
    print("==================================================")
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            img_path = test_ds.samples[idx][0].relative_to(PROJECT_ROOT)
            true_label = labels.item()
            
            images = images.to(device)
            outputs = model(images)
            pred_label = outputs.max(1)[1].item()
            
            all_labels.append(true_label)
            all_preds.append(pred_label)
            
            true_name = DISPLAY_NAMES[true_label]
            pred_name = DISPLAY_NAMES[pred_label]
            is_correct = "correct" if true_label == pred_label else "incorrect"
            if is_correct == "correct":
                correct_count += 1
                
            print(f"{img_path} | {true_name} | {pred_name} | {is_correct}")

    total_samples = len(test_ds)
    manual_accuracy = correct_count / total_samples
    print(f"\nManual Accuracy: {correct_count} / {total_samples} = {manual_accuracy:.4f}")
    
    # 4. SKLearn Report
    report_dict = classification_report(all_labels, all_preds, target_names=DISPLAY_NAMES, output_dict=True, zero_division=0)
    report_str = classification_report(all_labels, all_preds, target_names=DISPLAY_NAMES, digits=4, zero_division=0)
    
    print("\nClassification Report (sklearn):")
    print(report_str)
    
    macro_f1 = report_dict["macro avg"]["f1-score"]
    sklearn_accuracy = report_dict["accuracy"]
    
    print(f"SKLearn Accuracy: {sklearn_accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

if __name__ == "__main__":
    main()
