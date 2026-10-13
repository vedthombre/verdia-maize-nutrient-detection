"""
Maize Leaf Nutrient-Deficiency -- EfficientNet-B0 Model Builder
================================================================
Builds an EfficientNet-B0 for 5-class classification using official
pretrained ImageNet weights.

Architecture decisions
----------------------
* **Pretrained backbone (frozen)**:
  EfficientNet-B0 was trained on 1.28 M ImageNet images and has
  learned rich, general-purpose visual features (edges, textures,
  colours, shapes).  With only 324 training images, fine-tuning all
  5.3 M backbone parameters would almost certainly overfit.  By
  freezing the backbone we use it as a fixed feature extractor and
  only train the small classifier head (~6.4 K parameters).

* **Classifier head replacement**:
  The original EfficientNet-B0 head is:
      Dropout(p=0.2) -> Linear(1280, 1000)
  We replace it with:
      Dropout(p=0.2) -> Linear(1280, 5)
  This is the only trainable part of the network when fully frozen.

* **Partial Fine-tuning**:
  We can optionally unfreeze the later stages (blocks) of the
  backbone while keeping early layers frozen. This allows the model
  to learn dataset-specific high-level features while retaining
  general low-level edge/texture detectors.

Usage
-----
    from model import build_efficientnet_b0
    # Fully frozen backbone:
    model = build_efficientnet_b0(num_classes=5, freeze_backbone=True)
    # Partially unfrozen backbone (e.g., last 3 blocks):
    model = build_efficientnet_b0(num_classes=5, freeze_backbone=False, unfreeze_blocks=3)
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


def build_efficientnet_b0(num_classes: int = 5,
                          freeze_backbone: bool = True,
                          unfreeze_blocks: int = 0) -> nn.Module:
    """
    Build an EfficientNet-B0 model for multi-class classification.

    Parameters
    ----------
    num_classes     : Number of output classes (5 for this project).
    freeze_backbone : If True, all convolutional / feature-extraction
                      layers are frozen. Only the classifier head is
                      trainable.
    unfreeze_blocks : If freeze_backbone is False and this is > 0,
                      unfreezes only the last N blocks of the features.
                      The rest of the backbone remains frozen.

    Returns
    -------
    model : nn.Module
    """
    # Load EfficientNet-B0 with official ImageNet-1K pretrained weights.
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)

    # ── Freeze backbone ──────────────────────────────────────────────
    # model.features contains all convolutional blocks (0 to 8).
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    elif unfreeze_blocks > 0:
        # Freeze everything first
        for param in model.features.parameters():
            param.requires_grad = False
        # Unfreeze the last `unfreeze_blocks`
        num_blocks = len(model.features)
        for i in range(max(0, num_blocks - unfreeze_blocks), num_blocks):
            for param in model.features[i].parameters():
                param.requires_grad = True

    # ── Replace classifier head ──────────────────────────────────────
    # The original head: Sequential(Dropout(0.2), Linear(1280, 1000))
    # We keep the same dropout rate and replace the linear layer.
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    return model


def count_parameters(model: nn.Module):
    """Return (total_params, trainable_params) for a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # Quick sanity check.
    model = build_efficientnet_b0(num_classes=5, freeze_backbone=True)
    total, trainable = count_parameters(model)
    print(f"EfficientNet-B0 for 5-class classification")
    print(f"  Total parameters    : {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Frozen parameters   : {total - trainable:,}")
    print(f"  Classifier head     : {model.classifier}")
