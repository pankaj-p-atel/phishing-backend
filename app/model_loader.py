# app/model_loader.py

import os
import joblib
import gdown
from app.config import MODEL_PATH, GOOGLE_DRIVE_FILE_ID

_model = None

def download_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        print("Downloading model from Google Drive...")
        gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
        print("Download complete.")

def load_model():
    global _model
    if _model is None:
        download_model()
        print("Loading model into memory...")
        _model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    return _model

def reload_model():
    global _model
    print("Reloading model...")
    _model = joblib.load(MODEL_PATH)
    return _model