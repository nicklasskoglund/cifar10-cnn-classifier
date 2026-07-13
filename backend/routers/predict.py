"""
Prediction-router för CIFAR-10 CNN-klassificeraren.

Laddar den tränade Keras-modellen en gång vid uppstart och exponerar
en POST /predict-endpoint som tar emot en uppladdad bild och returnerar
den predikterade klassen tillsammans med confidence-poäng för alla 10 klasser.
"""

from io import BytesIO
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

router = APIRouter()

# CIFAR-10-klassnamn, i exakt samma ordning som användes vid träning
# (denna ordning matchar one-hot-kodningen som data_loader.py skapar)
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

IMAGE_SIZE = (32, 32)

# Modellsökvägen räknas ut relativt den här filen, så det fungerar
# oavsett vilken working directory servern startas ifrån.
MODEL_PATH = Path(__file__).resolve().parents[2] / "model" / "saved_models" / "best_model.keras"

_model = None


def get_model() -> tf.keras.Model:
    """Ladda den tränade modellen in i minnet vid första anropet (lazy singleton)."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modellfilen hittades inte på {MODEL_PATH}. "
                "Kontrollera att best_model.keras finns i model/saved_models/."
            )
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """
    Konvertera rå uppladdad bilddata till en numpy-array som modellen kan ta emot.

    Stegen speglar preprocessingen som användes vid träning:
    - konvertera till RGB (tar bort alfakanal / hanterar gråskale-uppladdningar)
    - ändra storlek till 32x32
    - skala pixelvärden till [0, 1]
    - lägg till en batch-dimension -> shape (1, 32, 32, 3)
    """
    image = Image.open(BytesIO(file_bytes)).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Ta emot en uppladdad bild och returnera den predikterade CIFAR-10-klassen
    tillsammans med confidence-poäng för alla 10 klasser.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Den uppladdade filen måste vara en bild.")

    file_bytes = await file.read()

    try:
        input_array = preprocess_image(file_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Kunde inte bearbeta bilden: {exc}")

    model = get_model()
    predictions = model.predict(input_array, verbose=0)[0]  # shape (10,)

    predicted_index = int(np.argmax(predictions))
    confidences = {
        CLASS_NAMES[i]: round(float(predictions[i]) * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": confidences[CLASS_NAMES[predicted_index]],
        "all_confidences": confidences,
    }