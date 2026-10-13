"""
Maize Leaf Nutrient-Deficiency -- Image Transform Pipelines
============================================================
Defines three separate transform pipelines for train / val / test,
tailored for a **pretrained EfficientNet-B0** backbone.

EfficientNet-B0 (torchvision, IMAGENET1K_V1 weights)
-----------------------------------------------------
* Expected input : 224 x 224 RGB tensor
* Normalisation  : ImageNet  mean=[0.485, 0.456, 0.406]
                              std =[0.229, 0.224, 0.225]
* Inference pre-processing: Resize(256) -> CenterCrop(224) -> Normalize

Design rationale for augmentation choices
------------------------------------------
We are classifying *nutrient deficiency* in maize leaves.  The primary
visual cues are **colour / discolouration patterns** (yellowing,
interveinal chlorosis, purpling, etc.), so we must keep colour
augmentation very conservative to avoid destroying the signal.

Selected training augmentations and why:

1. RandomResizedCrop(224, scale=(0.8, 1.0))
   - Simulates slight zoom / framing variation as would occur in
     hand-held field photography.
   - Scale range (0.8-1.0) keeps at least 80 % of the leaf visible,
     avoiding crops that miss the deficiency symptoms.

2. RandomHorizontalFlip(p=0.5)
   - Maize leaves are roughly bilaterally symmetric; nutrient-
     deficiency symptoms (colour changes) are NOT laterally biased.
   - Effectively doubles the training set at negligible cost.

3. RandomRotation(degrees=15)
   - Leaves are photographed at arbitrary angles in the field.
   - Limited to +/-15 deg to avoid unrealistic orientations.

4. ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02)
   - brightness/contrast: mimics natural lighting variation (sun/shade).
   - saturation: very mild -- saturation changes can mimic/mask
     chlorosis, so we keep this small.
   - hue: almost zero (0.02) -- hue IS the primary diagnostic signal
     (green vs. yellow vs. purple).  Large hue shifts would teach the
     model to ignore colour, which is exactly what we need it to use.

Augmentations deliberately EXCLUDED:
   - Vertical flip     : leaves have a natural tip-base orientation.
   - Perspective warp  : can distort leaf shape unrealistically.
   - Gaussian blur     : would obscure fine chlorosis patterns.
   - Cutout / erasing  : could erase the deficiency symptom region.
   - Heavy colour jitter: colour is the primary diagnostic cue.
"""

from torchvision import transforms

# ──────────────────────────────────────────────────────────────────────
# CONSTANTS (EfficientNet-B0, IMAGENET1K_V1)
# ──────────────────────────────────────────────────────────────────────
INPUT_SIZE  = 224          # model input resolution
RESIZE_SIZE = 256          # intermediate resize for val/test
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ──────────────────────────────────────────────────────────────────────
# TRAINING TRANSFORM
# ──────────────────────────────────────────────────────────────────────
# Includes realistic, mild augmentation + normalisation.

train_transform = transforms.Compose([
    # Randomly crop 80-100 % of the image area and resize to 224x224.
    transforms.RandomResizedCrop(
        INPUT_SIZE,
        scale=(0.8, 1.0),
        interpolation=transforms.InterpolationMode.BILINEAR,
    ),
    # Mirror horizontally with 50 % probability.
    transforms.RandomHorizontalFlip(p=0.5),
    # Small random rotation.
    transforms.RandomRotation(degrees=15),
    # Mild colour / brightness variation (conservative for colour-cue task).
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
        hue=0.02,
    ),
    # PIL Image -> float32 tensor in [0, 1] with shape (C, H, W).
    transforms.ToTensor(),
    # Shift to ImageNet distribution expected by the pretrained backbone.
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ──────────────────────────────────────────────────────────────────────
# VALIDATION TRANSFORM
# ──────────────────────────────────────────────────────────────────────
# Deterministic.  No random augmentation -- we want a stable metric.

val_transform = transforms.Compose([
    # Resize shortest side to 256, then take a centred 224x224 crop.
    transforms.Resize(RESIZE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ──────────────────────────────────────────────────────────────────────
# TEST TRANSFORM
# ──────────────────────────────────────────────────────────────────────
# Identical to validation: deterministic, no augmentation.

test_transform = transforms.Compose([
    transforms.Resize(RESIZE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ──────────────────────────────────────────────────────────────────────
# UTILITY: inverse normalisation (for visualisation only)
# ──────────────────────────────────────────────────────────────────────
# Undoes the ImageNet normalisation so we can display tensors as images.

inv_normalize = transforms.Compose([
    transforms.Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[1.0 / s for s in IMAGENET_STD],
    ),
    transforms.Normalize(
        mean=[-m for m in IMAGENET_MEAN],
        std=[1.0, 1.0, 1.0],
    ),
])
