from fastapi import FastAPI
from pydantic import BaseModel
from scripts.inference import load_model, predict_text

app = FastAPI(
    title="Portfolio NLP API",
    description="Detect technologies and classify project domains",
    version="1.0.0"
)

# Load model once when API starts
model, tokenizer, device = load_model()

import os

WEIGHTS_PATH = "models/best_model.pt"

if not os.path.exists(WEIGHTS_PATH):
    print("⚠️ AI model missing. Please download best_model.pt into /models")
    model = None
else:
    model = torch.load(WEIGHTS_PATH)
    

class PredictionRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {
        "message": "Portfolio NLP API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": "xlm-roberta-base"
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    result = predict_text(
        text=request.text,
        model=model,
        tokenizer=tokenizer,
        device=device,
        domain_threshold=0.1
    )

    return result