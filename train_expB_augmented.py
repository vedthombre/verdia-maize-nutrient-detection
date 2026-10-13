import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights
from torchvision import transforms

from dataset import MaizeLeafDataset, CLASS_NAMES
from transforms import val_transform

PROJECT_ROOT   = Path(__file__).resolve().parent
CSV_PATH       = PROJECT_ROOT / "dataset_split.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR    = PROJECT_ROOT / "results"

SEED           = 42
BATCH_SIZE     = 32
NUM_EPOCHS     = 40
PATIENCE       = 8
LR_BACKBONE    = 5e-5
LR_HEAD        = 1e-3
WEIGHT_DECAY   = 1e-4
NUM_WORKERS    = 0
NUM_CLASSES    = 5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform_expB = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.10, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

CHECKPOINT_PATH  = CHECKPOINT_DIR / "expB_augmented_best.pth"
HISTORY_PATH     = RESULTS_DIR    / "expB_augmented_history.json"
CONFIG_PATH      = RESULTS_DIR    / "expB_augmented_config.json"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_resnet18(num_classes: int = NUM_CLASSES) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def train_one_epoch(model, loader, criterion, optimiser, device):
    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimiser.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimiser.step()
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct  += preds.eq(labels).sum().item()
        total    += images.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct  += preds.eq(labels).sum().item()
        total    += images.size(0)
    return running_loss / total, correct / total

def main():
    set_seed(SEED)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Exp B on {device}...")

    train_ds = MaizeLeafDataset(CSV_PATH, "train", transform=train_transform_expB)
    val_ds   = MaizeLeafDataset(CSV_PATH, "val",   transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = build_resnet18(num_classes=NUM_CLASSES)
    model = model.to(device)

    head_params     = list(model.fc.parameters())
    head_param_ids  = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]

    optimiser = torch.optim.Adam([
        {"params": backbone_params, "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY},
        {"params": head_params,     "lr": LR_HEAD,     "weight_decay": WEIGHT_DECAY},
    ])
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    wall_start = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimiser, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        else:
            patience_counter += 1

        print(f"E{epoch:02d}: TL {train_loss:.4f} TA {train_acc:.1%} | VL {val_loss:.4f} VA {val_acc:.1%}")
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    elapsed = time.time() - wall_start
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss
    history["time"] = elapsed
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    with open(CONFIG_PATH, "w") as f:
        json.dump({"exp": "B", "seed": SEED, "lr_bb": LR_BACKBONE, "lr_h": LR_HEAD}, f)
    print("Exp B training complete.")

if __name__ == "__main__":
    main()
