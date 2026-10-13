"""
Maize Leaf Nutrient-Deficiency -- Grad-CAM Implementation for ResNet18
========================================================================
Implements Grad-CAM to visualize regions influencing the CNN prediction.

The chosen model is ResNet18 based on its highest test accuracy (88.57%)
and highest macro F1 (0.8384). Although EfficientNet-B0 had a slightly
lower validation loss (0.2998 vs 0.3319), ResNet18 achieved better
generalization on the test set.

Usage
-----
    python gradcam.py
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

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "dataset_split.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "resnet18_comparison_best.pth"
RESULTS_DIR = PROJECT_ROOT / "gradcam_results"
RESULTS_DIR.mkdir(exist_ok=True)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        score = output[0, target_class]
        score.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam)
        
        cam = cam.detach().cpu().numpy()
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam, output, target_class


def overlay_cam(img, cam):
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam_img = heatmap + np.float32(img)
    cam_img = cam_img / np.max(cam_img)
    return np.uint8(255 * cam_img)


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
    
    # We want examples from multiple classes, correct and incorrect predictions,
    # specifically including Phosphorus and Potassium.
    # Let's run inference on the test set and pick examples.
    
    selected_indices = []
    
    # Let's collect true labels and predicted labels to select specific cases
    all_preds = []
    for i in range(len(test_ds)):
        img_tensor, true_label = test_ds[i]
        img_tensor = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(dim=1).item()
            all_preds.append((i, true_label, pred))
            
    # Select examples
    # 1. Correct Healthy
    # 2. Correct Magnesium
    # 3. Correct Nitrogen
    # 4. Correct Phosphorus
    # 5. Correct Potassium
    # 6. Incorrect Phosphorus
    # 7. Incorrect Potassium
    
    def find_example(true_class_name, correct=True):
        true_idx = CLASS_NAMES.index(true_class_name)
        for i, t, p in all_preds:
            if t == true_idx and ((p == t) == correct):
                return i
        return None

    examples_to_plot = {
        "Correct_Healthy": find_example("Healthy", True),
        "Correct_Magnesium": find_example("Magnesium", True),
        "Correct_Nitrogen": find_example("Nitrogen", True),
        "Correct_Phosphorus": find_example("Phosphorus", True),
        "Correct_Potassium": find_example("Potassium", True),
        "Incorrect_Phosphorus": find_example("Phosphorus", False),
        "Incorrect_Potassium": find_example("Potassium", False)
    }

    for name, idx in examples_to_plot.items():
        if idx is None:
            print(f"Skipping {name} (no example found)")
            continue
            
        img_tensor, true_label = test_ds[idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Get original image for visualization (un-normalize)
        # Assuming mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        img_np = img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        cam, output, pred_class = grad_cam.generate_heatmap(img_tensor, target_class=None)
        
        probs = F.softmax(output, dim=1).squeeze().detach().cpu().numpy()
        confidence = probs[pred_class]
        
        cam_overlay = overlay_cam(img_np, cam)
        
        # Plot side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img_np)
        axes[0].set_title(f"Original Image\nTrue: {CLASS_NAMES[true_label]}")
        axes[0].axis('off')
        
        axes[1].imshow(cam_overlay)
        axes[1].set_title(f"Grad-CAM\nPred: {CLASS_NAMES[pred_class]} ({confidence:.1%})")
        axes[1].axis('off')
        
        plt.tight_layout()
        save_path = RESULTS_DIR / f"{name}_idx{idx}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved {name} visualization to {save_path}")

if __name__ == "__main__":
    main()
