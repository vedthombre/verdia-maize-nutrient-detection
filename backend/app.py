"""
backend/app.py
==============
FastAPI application for the Verdia Maize Leaf Nutrient-Deficiency classifier.

Endpoints
---------
GET  /health   – liveness / model-loaded check
POST /predict  – accepts a maize leaf image, returns Experiment 5 prediction JSON

Start from the project root with:
    uvicorn backend.app:app --reload --port 8000

Response schema (POST /predict):
    {
        "prediction":    "Potassium",
        "confidence":    0.734,
        "probabilities": {
            "Healthy":    0.021,
            "Magnesium":  0.042,
            "Nitrogen":   0.087,
            "Phosphorus": 0.116,
            "Potassium":  0.734
        }
    }
"""

import io

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

# Model is loaded at import time (once at startup, not per-request).
from backend.inference import predict_pil

# ── application ───────────────────────────────────────────────────────
app = FastAPI(
    title="Verdia — Maize Leaf Nutrient-Deficiency API",
    description=(
        "Classifies maize leaf images into one of five nutrient-deficiency "
        "categories using EfficientNet-B0 (Experiment 5, Verdia)."
    ),
    version="1.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────
# Only the local Next.js development server is allowed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Accepted MIME types / file extensions for uploaded images
_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
_ALLOWED_EXTENSIONS    = {".jpg", ".jpeg", ".png", ".webp"}


# ── routes ────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check", tags=["Health"])
def health_check():
    """
    Returns `{"status": "ok"}` once the model has loaded successfully.
    The server will fail to start entirely if the checkpoint is missing,
    so reaching this endpoint guarantees the model is in memory.
    """
    return {
        "status": "ok",
        "model": "efficientnet_b0_deeper_finetuned",
    }


@app.post("/predict", summary="Predict nutrient deficiency", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    **POST /predict**

    Upload a maize leaf image (JPG or PNG) as `multipart/form-data`
    under the field name `file`.

    Returns the predicted nutrient class, its confidence, and the full
    probability distribution over all five classes.
    """
    # ── 1. Validate that a real file was sent ─────────────────────────
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file was uploaded. Attach an image via the 'file' field.",
        )

    # ── 2. Validate content-type / extension ──────────────────────────
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    fname_lower  = file.filename.lower()
    ext          = "." + fname_lower.rsplit(".", 1)[-1] if "." in fname_lower else ""

    if content_type not in _ALLOWED_CONTENT_TYPES and ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                "Accepted formats: JPG, JPEG, PNG."
            ),
        )

    # ── 3. Read raw bytes ─────────────────────────────────────────────
    try:
        raw = await file.read()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read the uploaded file.",
        )

    if not raw:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    # ── 4. Decode image and convert to RGB ───────────────────────────
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "The file could not be decoded as an image. "
                "It may be corrupted or in an unsupported format."
            ),
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="An error occurred while processing the image.",
        )

    # ── 5. Run Experiment 5 inference ─────────────────────────────────
    try:
        result = predict_pil(image)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed. Please try again.",
        )

    # ── 6. Return response (key: "prediction" per Verdia spec) ────────
    return {
        "prediction":    result["predicted_class"],
        "confidence":    result["confidence"],
        "probabilities": result["probabilities"],
    }
