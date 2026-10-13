"""
Step 2 of 2: Train ExpC using pre-saved masks from masks_cache/.
Loads masks instantly from .npy files — no GrabCut during training.
"""
import json
import time
import random
from pathlib import Path
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.models import ResNet18_Weights

from dataset import MaizeLeafDataset, CLASS_NAMES
from transforms import train_transform, val_transform

PROJECT_ROOT   = Path(__file__).resolve().parent
CSV_PATH       = PROJECT_ROOT / "dataset_split.csv"
MASK_DIR       = PROJECT_ROOT / "masks_cache"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR    = PROJECT_ROOT / "results"

SEED         = 42
BATCH_SIZE   = 32
NUM_EPOCHS   = 40
PATIENCE     = 8
LR_BACKBONE  = 5e-5
LR_HEAD      = 1e-3
WEIGHT_DECAY = 1e-4
NUM_CLASSES  = 5
MASK_PROB    = 0.5

CHECKPOINT_PATH = CHECKPOINT_DIR / "expC_masked_bg_best.pth"
HISTORY_PATH    = RESULTS_DIR    / "expC_masked_bg_history.json"
CONFIG_PATH     = RESULTS_DIR    / "expC_masked_bg_config.json"
AUDIT_PATH      = RESULTS_DIR    / "expC_mask_audit_grid.png"

NEUTRAL_U8 = np.array([124, 116, 104], dtype=np.uint8)


class MaskedMaizeLeafDataset(Dataset):
    def __init__(self, base_dataset, mask_dir, prob=0.5):
        self.base     = base_dataset
        self.mask_dir = Path(mask_dir)
        self.prob     = prob
        # Pre-load mask availability map
        self.has_mask = {
            i: (self.mask_dir / f"mask_{i:04d}.npy").exists()
            for i in range(len(base_dataset))
        }
        accepted = sum(self.has_mask.values())
        rejected = len(base_dataset) - accepted
        print(f"[Dataset] Masks: {accepted} accepted, {rejected} no mask file.", flush=True)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img_path, _, label = self.base.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.has_mask[idx] and random.random() < self.prob:
            mask = np.load(self.mask_dir / f"mask_{idx:04d}.npy")
            img_np = np.array(image)
            img_np[mask == 0] = NEUTRAL_U8
            image = Image.fromarray(img_np)
        if self.base.transform is not None:
            image = self.base.transform(image)
        return image, label


def build_resnet18():
    m = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, opt, device):
    model.train()
    ls = cor = tot = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        opt.zero_grad()
        out  = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        opt.step()
        ls  += loss.item() * imgs.size(0)
        cor += out.argmax(1).eq(lbls).sum().item()
        tot += imgs.size(0)
    return ls / tot, cor / tot


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    ls = cor = tot = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        out  = model(imgs)
        loss = criterion(out, lbls)
        ls  += loss.item() * imgs.size(0)
        cor += out.argmax(1).eq(lbls).sum().item()
        tot += imgs.size(0)
    return ls / tot, cor / tot


def save_audit_grid(base_ds, mask_dir, audit_path, n_rows=5):
    fig, axes = plt.subplots(n_rows, 3, figsize=(10, n_rows * 3))
    plotted = 0
    for i in range(len(base_ds)):
        mask_path = Path(mask_dir) / f"mask_{i:04d}.npy"
        if not mask_path.exists():
            continue
        img_path, cls_name, _ = base_ds.samples[i]
        img_np  = np.array(Image.open(img_path).convert("RGB"))
        mask    = np.load(mask_path)
        masked  = img_np.copy(); masked[mask == 0] = NEUTRAL_U8
        r = plotted
        axes[r][0].imshow(img_np);                          axes[r][0].set_title(f"{cls_name} Orig"); axes[r][0].axis("off")
        axes[r][1].imshow(mask, cmap="gray", vmin=0, vmax=1); axes[r][1].set_title("Mask");           axes[r][1].axis("off")
        axes[r][2].imshow(masked);                          axes[r][2].set_title("Neutral BG");       axes[r][2].axis("off")
        plotted += 1
        if plotted >= n_rows:
            break
    plt.tight_layout()
    plt.savefig(audit_path, dpi=120)
    plt.close()
    print(f"[ExpC] Audit grid saved -> {audit_path}", flush=True)


def main():
    if not MASK_DIR.exists() or not any(MASK_DIR.glob("mask_*.npy")):
        print("ERROR: No masks found. Run precompute_masks.py first.", flush=True)
        return

    set_seed(SEED)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ExpC] Device: {device}", flush=True)

    base_ds = MaizeLeafDataset(CSV_PATH, "train", transform=train_transform)
    val_ds  = MaizeLeafDataset(CSV_PATH, "val",   transform=val_transform)

    save_audit_grid(base_ds, MASK_DIR, AUDIT_PATH)

    masked_ds    = MaskedMaizeLeafDataset(base_ds, MASK_DIR, prob=MASK_PROB)
    train_loader = DataLoader(masked_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_resnet18().to(device)
    head_ids   = {id(p) for p in model.fc.parameters()}
    backbone_p = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.Adam([
        {"params": backbone_p,              "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY},
        {"params": model.fc.parameters(),   "lr": LR_HEAD,     "weight_decay": WEIGHT_DECAY},
    ])
    criterion = nn.CrossEntropyLoss()

    history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_epoch    = -1
    patience_ctr  = 0
    t_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        tl, ta = train_one_epoch(model, train_loader, criterion, opt, device)
        vl, va = validate(model, val_loader, criterion, device)
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)

        if vl < best_val_loss:
            best_val_loss = vl; best_epoch = epoch; patience_ctr = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            flag = " <-- saved"
        else:
            patience_ctr += 1; flag = ""

        print(f"E{epoch:02d}: TL {tl:.4f} TA {ta:.1%} | VL {vl:.4f} VA {va:.1%}{flag}", flush=True)
        if patience_ctr >= PATIENCE:
            print("[ExpC] Early stopping.", flush=True); break

    elapsed = time.time() - t_start
    history.update({"best_epoch": best_epoch, "best_val_loss": best_val_loss, "time_seconds": elapsed})
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    mask_count = len(list(MASK_DIR.glob("mask_*.npy")))
    with open(CONFIG_PATH, "w") as f:
        json.dump({"exp": "C", "seed": SEED, "lr_backbone": LR_BACKBONE,
                   "lr_head": LR_HEAD, "weight_decay": WEIGHT_DECAY,
                   "batch_size": BATCH_SIZE, "patience": PATIENCE,
                   "mask_prob": MASK_PROB, "accepted_masks": mask_count,
                   "best_epoch": best_epoch}, f, indent=2)
    print(f"[ExpC] Complete in {elapsed/60:.1f} min. "
          f"Best val_loss={best_val_loss:.4f} at epoch {best_epoch}.", flush=True)


if __name__ == "__main__":
    main()
