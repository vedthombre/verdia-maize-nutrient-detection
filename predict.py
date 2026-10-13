"""
Maize Leaf Nutrient-Deficiency -- Inference Pipeline
====================================================
Predicts nutrient deficiency from a single maize leaf image using the 
Experiment 5 verified checkpoint.
"""

import sys
import argparse
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F

from model import build_efficientnet_b0
from transforms import test_transform
from dataset import DISPLAY_NAMES

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "efficientnet_b0_deeper_finetuned_best.pth"
UNFREEZE_BLKS = 5

def predict_image(image_input):
    """
    Runs inference on a single image.
    
    Args:
        image_input (str or PIL.Image.Image): Path to the image file or a PIL Image object.
        
    Returns:
        dict: A dictionary containing the predicted class, confidence, 
              and probabilities for all 5 classes.
    """
    if isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        img_path = Path(image_input)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at {img_path}")

        # 1 & 2. Load the image and convert to RGB
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Could not open image {img_path}: {e}")

    # 3. Apply the deterministic test transform (Resize, CenterCrop, ToTensor, Normalize)
    image_tensor = test_transform(image)

    # 4. Add batch dimension
    batch_tensor = image_tensor.unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_tensor = batch_tensor.to(device)

    # 5. Load the checkpoint and architecture
    model = build_efficientnet_b0(num_classes=5, freeze_backbone=False, unfreeze_blocks=UNFREEZE_BLKS)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model = model.to(device)

    # 6. Set model to eval mode
    model.eval()

    # 7 & 8. Run inference to compute logits
    with torch.no_grad():
        logits = model(batch_tensor)

        # 9. Apply softmax across the 5 classes
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    # 10. Select the class with the highest probability
    predicted_idx = probs.argmax()
    predicted_class = DISPLAY_NAMES[predicted_idx]
    confidence = probs[predicted_idx]

    # Structure the results
    results = {
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "probabilities": {DISPLAY_NAMES[i]: float(probs[i]) for i in range(len(DISPLAY_NAMES))}
    }

    return results

def main():
    parser = argparse.ArgumentParser(description="Run inference on a single maize leaf image.")
    parser.add_argument("image_path", type=str, help="Path to the image to classify.")
    args = parser.parse_args()

    print(f"\nProcessing: {args.image_path}")
    try:
        results = predict_image(args.image_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    predicted_class = results["predicted_class"]
    confidence = results["confidence"]
    probs = results["probabilities"]

    print("=" * 50)
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    print("=" * 50)
    print("Class Probabilities:")
    for cls_name, prob in probs.items():
        print(f"  {cls_name}: {prob:.2%}")
    print("=" * 50)
    
    # Verify probabilities sum to ~1
    total_prob = sum(probs.values())
    print(f"[Debug] Sum of probabilities: {total_prob:.4f}")

if __name__ == "__main__":
    main()
