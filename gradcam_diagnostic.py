"""
Maize Leaf Nutrient-Deficiency -- Grad-CAM Diagnostic Script
=============================================================
This script performs an additional diagnostic audit on the Grad-CAM implementation
and investigates potential background dependence by masking out the background.
"""

import os
import cv2
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn.functional as F

from dataset import MaizeLeafDataset, CLASS_NAMES
from transforms import test_transform
from train_resnet18 import build_resnet18, NUM_CLASSES
from gradcam import GradCAM, overlay_cam

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "dataset_split.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "resnet18_comparison_best.pth"
RESULTS_DIR = PROJECT_ROOT / "gradcam_diagnostic_results"
RESULTS_DIR.mkdir(exist_ok=True)

def create_leaf_mask(img_np):
    """
    Creates a simple mask to isolate the leaf from the background.
    Uses GrabCut algorithm with a central rectangle assumption.
    Input img_np is [0, 1] RGB.
    """
    img_uint8 = np.uint8(img_np * 255)
    
    mask = np.zeros(img_uint8.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # Define a rectangle covering most of the image, assuming leaf is central
    h, w = img_uint8.shape[:2]
    rect = (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8))
    
    cv2.grabCut(img_uint8, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    
    # Modify mask: 0 and 2 are background, 1 and 3 are foreground
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    
    masked_img = img_uint8 * mask2[:, :, np.newaxis]
    return np.float32(masked_img) / 255.0, mask2

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_resnet18(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    target_layer = model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)

    test_ds = MaizeLeafDataset(CSV_PATH, "test", transform=test_transform)
    
    all_preds = []
    for i in range(len(test_ds)):
        img_tensor, true_label = test_ds[i]
        img_tensor = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(dim=1).item()
            all_preds.append((i, true_label, pred))
            
    def find_examples(true_class_name, correct=True, count=1):
        true_idx = CLASS_NAMES.index(true_class_name)
        found = []
        for i, t, p in all_preds:
            if t == true_idx and ((p == t) == correct):
                found.append(i)
                if len(found) == count:
                    break
        return found

    examples_to_plot = {}
    for cls in CLASS_NAMES:
        correct_idx = find_examples(cls, correct=True, count=1)
        if correct_idx:
            examples_to_plot[f"Correct_{cls}"] = correct_idx[0]
            
        incorrect_idx = find_examples(cls, correct=False, count=1)
        if incorrect_idx:
            examples_to_plot[f"Incorrect_{cls}"] = incorrect_idx[0]

    diagnostic_log = []

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for name, idx in examples_to_plot.items():
        img_tensor, true_label = test_ds[idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Original Inference
        cam, output, pred_class = grad_cam.generate_heatmap(img_tensor, target_class=None)
        probs = F.softmax(output, dim=1).squeeze().detach().cpu().numpy()
        confidence = probs[pred_class]
        
        img_np = img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        cam_overlay = overlay_cam(img_np, cam)
        
        # Masked Background Inference
        masked_img_np, mask = create_leaf_mask(img_np)
        
        # Re-normalize for the model
        masked_img_tensor = (masked_img_np - mean) / std
        masked_img_tensor = torch.tensor(masked_img_tensor.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)
        
        cam_masked, output_masked, pred_class_masked = grad_cam.generate_heatmap(masked_img_tensor, target_class=None)
        probs_masked = F.softmax(output_masked, dim=1).squeeze().detach().cpu().numpy()
        confidence_masked = probs_masked[pred_class_masked]
        
        cam_overlay_masked = overlay_cam(masked_img_np, cam_masked)
        
        diagnostic_log.append({
            "filename": test_ds.samples[idx][0].name,
            "condition": name,
            "true_class": CLASS_NAMES[true_label],
            "original_pred": CLASS_NAMES[pred_class],
            "original_conf": float(confidence),
            "masked_pred": CLASS_NAMES[pred_class_masked],
            "masked_conf": float(confidence_masked)
        })
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title(f"Original Image\nTrue: {CLASS_NAMES[true_label]}")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cam_overlay)
        axes[0, 1].set_title(f"Grad-CAM (Orig)\nPred: {CLASS_NAMES[pred_class]} ({confidence:.1%})")
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(masked_img_np)
        axes[1, 0].set_title(f"Masked Image (GrabCut)")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(cam_overlay_masked)
        axes[1, 1].set_title(f"Grad-CAM (Masked)\nPred: {CLASS_NAMES[pred_class_masked]} ({confidence_masked:.1%})")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        save_path = RESULTS_DIR / f"{name}_idx{idx}_diagnostic.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved diagnostic visualization to {save_path}")

    with open(RESULTS_DIR / "diagnostic_log.json", "w") as f:
        json.dump(diagnostic_log, f, indent=4)
        
    print("Diagnostic execution completed.")

if __name__ == "__main__":
    main()
