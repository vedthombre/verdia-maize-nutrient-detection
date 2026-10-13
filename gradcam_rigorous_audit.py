"""
Maize Leaf Nutrient-Deficiency -- Rigorous Grad-CAM Audit
===========================================================
Performs a rigorous background-masking experiment on the ENTIRE test set (70 images).
1. Original Image
2. Masked Background (Black)
3. Masked Background (Neutral Mean RGB)
4. Leaf Disruption (Mask a central square on the leaf)
"""

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
RESULTS_DIR = PROJECT_ROOT / "gradcam_rigorous_results"
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

def create_leaf_disruption_mask(img_np):
    # Mask out a central 25% area (e.g. 112x112 square in a 224x224 image)
    mask = np.ones(img_np.shape[:2], np.uint8)
    h, w = mask.shape
    start_y, start_x = h // 4, w // 4
    end_y, end_x = 3 * h // 4, 3 * w // 4
    mask[start_y:end_y, start_x:end_x] = 0
    return mask

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_resnet18(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    test_ds = MaizeLeafDataset(CSV_PATH, "test", transform=test_transform)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    diagnostic_log = []
    
    grid_images = [] # For saving the visual audit grid

    for i in range(len(test_ds)):
        img_tensor, true_label = test_ds[i]
        filename = test_ds.samples[i][0].name
        
        img_np = img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        # 1. Original Inference
        with torch.no_grad():
            inp = img_tensor.unsqueeze(0).to(device)
            out = model(inp)
            probs = F.softmax(out, dim=1).squeeze().cpu().numpy()
            orig_pred = out.argmax(dim=1).item()
            orig_conf = float(probs[orig_pred])
            
        # Masks
        grabcut_mask = create_leaf_mask(img_np)
        disruption_mask = create_leaf_disruption_mask(img_np)
        
        # 2. Masked Background (Black)
        masked_black_np = img_np * grabcut_mask[:, :, np.newaxis]
        masked_black_tensor = torch.tensor(((masked_black_np - mean) / std).transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            out_black = model(masked_black_tensor)
            probs_black = F.softmax(out_black, dim=1).squeeze().cpu().numpy()
            black_pred = out_black.argmax(dim=1).item()
            black_conf = float(probs_black[black_pred])
            
        # 3. Masked Background (Neutral Mean RGB)
        masked_neutral_np = np.copy(img_np)
        masked_neutral_np[grabcut_mask == 0] = mean # Apply ImageNet mean to background
        masked_neutral_tensor = torch.tensor(((masked_neutral_np - mean) / std).transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            out_neutral = model(masked_neutral_tensor)
            probs_neutral = F.softmax(out_neutral, dim=1).squeeze().cpu().numpy()
            neutral_pred = out_neutral.argmax(dim=1).item()
            neutral_conf = float(probs_neutral[neutral_pred])
            
        # 4. Leaf Disruption (Center Square Masked Black)
        disrupted_np = img_np * disruption_mask[:, :, np.newaxis]
        disrupted_tensor = torch.tensor(((disrupted_np - mean) / std).transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            out_disrupted = model(disrupted_tensor)
            probs_disrupted = F.softmax(out_disrupted, dim=1).squeeze().cpu().numpy()
            disrupted_pred = out_disrupted.argmax(dim=1).item()
            disrupted_conf = float(probs_disrupted[disrupted_pred])
            
        diagnostic_log.append({
            "index": i,
            "filename": filename,
            "true_class": CLASS_NAMES[true_label],
            "original": {"pred": CLASS_NAMES[orig_pred], "conf": orig_conf, "correct": orig_pred == true_label},
            "bg_black": {"pred": CLASS_NAMES[black_pred], "conf": black_conf, "correct": black_pred == true_label},
            "bg_neutral": {"pred": CLASS_NAMES[neutral_pred], "conf": neutral_conf, "correct": neutral_pred == true_label},
            "leaf_disrupted": {"pred": CLASS_NAMES[disrupted_pred], "conf": disrupted_conf, "correct": disrupted_pred == true_label}
        })
        
        # Save first image of each class for visual grid audit
        if len(grid_images) < 5:
            # check if we already have this class
            existing_classes = [g["class"] for g in grid_images]
            if CLASS_NAMES[true_label] not in existing_classes:
                grid_images.append({
                    "class": CLASS_NAMES[true_label],
                    "orig": img_np,
                    "mask": grabcut_mask,
                    "masked": masked_black_np
                })

    with open(RESULTS_DIR / "rigorous_log.json", "w") as f:
        json.dump(diagnostic_log, f, indent=4)
        
    # Plot Visual Grid
    fig, axes = plt.subplots(5, 3, figsize=(10, 15))
    for r, img_data in enumerate(grid_images):
        axes[r, 0].imshow(img_data["orig"])
        axes[r, 0].set_title(f"{img_data['class']} (Orig)")
        axes[r, 0].axis('off')
        
        axes[r, 1].imshow(img_data["mask"], cmap='gray', vmin=0, vmax=1)
        axes[r, 1].set_title("Binary Mask")
        axes[r, 1].axis('off')
        
        axes[r, 2].imshow(img_data["masked"])
        axes[r, 2].set_title("Masked Image")
        axes[r, 2].axis('off')
        
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "mask_visual_audit_grid.png", dpi=150)
    plt.close()
    
    print(f"Rigorous execution completed on {len(test_ds)} images.")

if __name__ == "__main__":
    main()
