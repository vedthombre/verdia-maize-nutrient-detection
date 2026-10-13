"""
backend/main.py
===============
FastAPI application for the Maize Leaf Nutrient-Deficiency classifier.

Endpoints
---------
GET  /health   – liveness check
POST /predict  – accepts a maize leaf image, returns prediction JSON

Run with:
    uvicorn backend.main:app --reload --port 8000

(from the project root, i.e. the directory containing backend/)
"""

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
import io

# Model is loaded at import time (once at startup).
from backend.inference import predict_pil

# ── application ───────────────────────────────────────────────────────
app = FastAPI(
    title="Maize Leaf Nutrient-Deficiency API",
    description=(
        "Classifies maize leaf images into one of five nutrient-deficiency "
        "categories using EfficientNet-B0 (Experiment 5 checkpoint)."
    ),
    version="1.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────
# Only allow the local Next.js development server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Allowed MIME types for uploaded images
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ── routes ────────────────────────────────────────────────────────────

@app.get(
    "/health",
    summary="Health check",
    tags=["Health"],
)
def health_check():
    """
    Returns `{"status": "ok"}` when the API is running and the model
    has been loaded successfully.
    """
    return {
        "status": "ok",
        "model": "efficientnet_b0_deeper_finetuned",
    }


@app.post(
    "/predict",
    summary="Predict nutrient deficiency from a maize leaf image",
    tags=["Prediction"],
)
async def predict(file: UploadFile = File(...)):
    """
    **POST /predict**

    Upload a maize leaf image (JPG or PNG) and receive:

    ```json
    {
        "predicted_class": "Potassium",
        "confidence": 0.734,
        "probabilities": {
            "Healthy": 0.021,
            "Magnesium": 0.042,
            "Nitrogen": 0.087,
            "Phosphorus": 0.116,
            "Potassium": 0.734
        }
    }
    ```
    """
    # 1. Validate that a file was uploaded
    if file is None or file.filename == "":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file uploaded. Please attach an image file to the request.",
        )

    # 2. Validate content type / extension
    content_type = (file.content_type or "").lower()
    filename_lower = (file.filename or "").lower()
    extension = ""
    if "." in filename_lower:
        extension = "." + filename_lower.rsplit(".", 1)[-1]

    if content_type not in ALLOWED_CONTENT_TYPES and extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{content_type}'. "
                "Only JPG and PNG images are accepted."
            ),
        )

    # 3. Read the raw bytes
    try:
        image_bytes = await file.read()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read the uploaded file.",
        )

    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    # 4. Decode and convert to RGB
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The uploaded file could not be decoded as an image. "
                   "It may be corrupted or in an unsupported format.",
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="An error occurred while processing the image.",
        )

    # 5. Run inference via the shared pipeline
    try:
        result = predict_pil(image)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed. Please try again or contact support.",
        )

    # 6. Return the structured JSON response
    return result
