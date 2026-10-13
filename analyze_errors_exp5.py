import json
import csv
from pathlib import Path
from collections import Counter
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import MaizeLeafDataset, DISPLAY_NAMES
from transforms import test_transform
from transforms import inv_normalize
from model import build_efficientnet_b0

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "dataset_split.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "efficientnet_b0_deeper_finetuned_best.pth"
RESULTS_DIR = PROJECT_ROOT / "results"
SEED = 42

def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Model (Experiment 5 configuration)
    model = build_efficientnet_b0(num_classes=5, freeze_backbone=False, unfreeze_blocks=5)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # Load Test Data
    test_ds = MaizeLeafDataset(CSV_PATH, "test", transform=test_transform)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)

    all_predictions = []
    errors = []
    potassium_predictions = []

    print("Running inference on test set...")
    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            img_path = str(test_ds.samples[idx][0].relative_to(PROJECT_ROOT))
            true_label = labels.item()
            true_class = DISPLAY_NAMES[true_label]

            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()

            # Sort probabilities
            sorted_indices = np.argsort(probs)[::-1]
            top_classes = [DISPLAY_NAMES[i] for i in sorted_indices]
            top_probs = [probs[i] for i in sorted_indices]

            predicted_class = top_classes[0]
            predicted_prob = top_probs[0]
            true_class_prob = probs[true_label]
            second_class = top_classes[1]
            second_prob = top_probs[1]
            is_correct = true_class == predicted_class

            pred_data = {
                "image_path": img_path,
                "true_class": true_class,
                "predicted_class": predicted_class,
                "predicted_probability": predicted_prob,
                "true_class_probability": true_class_prob,
                "second_predicted_class": second_class,
                "second_predicted_probability": second_prob,
                "correct": is_correct,
                "probs": probs, # Raw probabilities for internal use
                "top_classes": top_classes,
                "top_probs": top_probs,
                "image_tensor": images.squeeze().cpu() # for visualization
            }
            all_predictions.append(pred_data)

            if not is_correct:
                errors.append(pred_data)
            
            if true_class == "Potassium":
                potassium_predictions.append(pred_data)

    # 1. Save all predictions
    pred_csv_path = RESULTS_DIR / "deeper_finetuned_test_predictions.csv"
    with open(pred_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "true_class", "predicted_class", "predicted_probability", "true_class_probability", "second_predicted_class", "second_predicted_probability", "correct"])
        for p in all_predictions:
            writer.writerow([p["image_path"], p["true_class"], p["predicted_class"], f"{p['predicted_probability']:.4f}", f"{p['true_class_probability']:.4f}", p["second_predicted_class"], f"{p['second_predicted_probability']:.4f}", p["correct"]])

    # 4. Save error analysis CSV
    err_csv_path = RESULTS_DIR / "deeper_finetuned_error_analysis.csv"
    with open(err_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "true_class", "predicted_class", "predicted_probability", "true_class_probability", "second_class", "second_class_probability", "correct"])
        for p in errors:
            writer.writerow([p["image_path"], p["true_class"], p["predicted_class"], f"{p['predicted_probability']:.4f}", f"{p['true_class_probability']:.4f}", p["second_predicted_class"], f"{p['second_predicted_probability']:.4f}", p["correct"]])

    # Summary Stats
    print("\n==================================================")
    print("SUMMARY")
    print("==================================================")
    print(f"A. Total errors: {len(errors)} / {len(all_predictions)}")
    
    true_counts = Counter([e["true_class"] for e in errors])
    print("\nB. Error count by true class:")
    for k, v in true_counts.items():
        print(f"  {k}: {v}")
        
    pred_counts = Counter([e["predicted_class"] for e in errors])
    print("\nC. Error count by predicted class:")
    for k, v in pred_counts.items():
        print(f"  {k}: {v}")

    # E. Confusion Pairs
    print("\nE. Most common confusion pairs (True -> Predicted):")
    pairs = Counter([(e["true_class"], e["predicted_class"]) for e in errors])
    for (t, p), count in pairs.most_common():
        print(f"  {t} -> {p}: {count}")

    # 3. Specifically analyze all 8 Potassium images
    print("\n==================================================")
    print("F. POTASSIUM PREDICTIONS")
    print("==================================================")
    for p in potassium_predictions:
        print(f"Path: {p['image_path']}")
        print(f"True: {p['true_class']:<10} | Pred: {p['predicted_class']:<10} | Correct: {p['correct']}")
        print(f"P(Healthy): {p['probs'][0]:.4f} | P(Magnesium): {p['probs'][1]:.4f} | P(Nitrogen): {p['probs'][2]:.4f} | P(Phosphorus): {p['probs'][3]:.4f} | P(Potassium): {p['probs'][4]:.4f}")
        print("-" * 50)

    # 5. Create a visual contact sheet
    num_errors = len(errors)
    if num_errors > 0:
        cols = 4
        rows = (num_errors + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        if type(axes) is np.ndarray:
            axes = axes.flatten()
        else:
            axes = [axes]

        for i, ax in enumerate(axes):
            if i < num_errors:
                e = errors[i]
                img_tensor = e["image_tensor"]
                img_display = inv_normalize(img_tensor).clamp(0, 1).permute(1, 2, 0).numpy()
                ax.imshow(img_display)
                ax.set_title(f"True: {e['true_class']}\nPred: {e['predicted_class']}\nConf: {e['predicted_probability']:.2f}")
                ax.axis("off")
            else:
                ax.axis("off")

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "deeper_finetuned_misclassified_samples.png")
        print(f"\nVisual contact sheet saved to {RESULTS_DIR / 'deeper_finetuned_misclassified_samples.png'}")

if __name__ == "__main__":
    main()
