from fastapi import APIRouter
from app.schemas import URLRequest, PredictionResponse
from app.model_loader import load_model
import numpy as np

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
def predict(request: URLRequest):
    model = load_model()

    prediction = model.predict([request.url])[0]
    probabilities = model.predict_proba([request.url])[0]

    # get confidence of predicted class
    class_index = list(model.classes_).index(prediction)
    confidence = float(probabilities[class_index])

    return PredictionResponse(
        url=request.url,
        prediction=prediction,
        confidence=confidence
    )

@router.get("/health")
def health_check():
    return {"status": "API is running"}

@router.get("/model-info")
def model_info():
    model = load_model()
    return {
        "classes": model.classes_.tolist(),
        "vectorizer_type": str(type(model.named_steps["vectorizer"])),
        "classifier_type": str(type(model.named_steps["clf"]))
    }