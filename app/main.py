from fastapi import FastAPI
from app.routes import router
from app.model_loader import load_model

app = FastAPI(title="Phishing URL Detection API")

app.include_router(router)

@app.on_event("startup")
def startup_event():
    load_model()