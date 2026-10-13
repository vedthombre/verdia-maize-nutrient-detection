"""
backend/inference.py
====================
Loads the Experiment 5 EfficientNet-B0 checkpoint ONCE and exposes
`predict_pil(image)` for the FastAPI endpoint to call.

Design notes
------------
* Model is loaded at module import time so FastAPI can call
  `from backend.inference import predict_pil` during startup —
  the checkpoint is NOT re-read on every request.
* All preprocessing and architecture decisions are identical to
  predict.py (Resize(256) → CenterCrop(224) → ImageNet normalise).
* Works on CPU; uses torch.no_grad() during inference.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

# ── resolve the project root so we can import shared modules ───────────
# backend/ lives one level below the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── import the EXACT same building blocks used by predict.py ──────────
from model import build_efficientnet_b0       # EfficientNet-B0 builder
from transforms import test_transform          # Resize(256)→CenterCrop(224)→Normalize
from dataset import DISPLAY_NAMES             # ['Healthy','Magnesium','Nitrogen','Phosphorus','Potassium']

# ── constants (must match predict.py exactly) ─────────────────────────
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "efficientnet_b0_deeper_finetuned_best.pth"
UNFREEZE_BLKS = 5
NUM_CLASSES = 5

# ── device ────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model() -> torch.nn.Module:
    """
    Load the Experiment 5 checkpoint and return an eval-mode model.
    Called once at module import time.
    """
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}. "
            "Ensure the file exists relative to the project root."
        )

    model = build_efficientnet_b0(
        num_classes=NUM_CLASSES,
        freeze_backbone=False,
        unfreeze_blocks=UNFREEZE_BLKS,
    )
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    )
    model = model.to(DEVICE)
    model.eval()
    return model


# Load the model ONCE when this module is first imported.
_model: torch.nn.Module = _load_model()


def predict_pil(image: Image.Image) -> dict:
    """
    Run inference on a PIL image (already converted to RGB by the caller).

    Parameters
    ----------
    image : PIL.Image.Image
        An RGB PIL image of a maize leaf.

    Returns
    -------
    dict with keys:
        predicted_class  (str)   – highest-probability class label
        confidence       (float) – probability of the predicted class
        probabilities    (dict)  – {class_name: probability} for all 5 classes
    """
    # Apply the same deterministic test transform used in predict.py
    image_tensor = test_transform(image)         # shape: (3, 224, 224)
    batch_tensor = image_tensor.unsqueeze(0)     # shape: (1, 3, 224, 224)
    batch_tensor = batch_tensor.to(DEVICE)

    with torch.no_grad():
        logits = _model(batch_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    predicted_idx = probs.argmax()
    predicted_class = DISPLAY_NAMES[predicted_idx]
    confidence = float(probs[predicted_idx])

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": {
            DISPLAY_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)
        },
    }
