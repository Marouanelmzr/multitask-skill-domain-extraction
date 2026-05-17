from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from scripts.inference import load_model, predict_text

@asynccontextmanager
async def lifespan(app: FastAPI):
    global session, tokenizer, tech_normaliser
    session, tokenizer, tech_normaliser = load_model()
    yield

app = FastAPI(
    title="Portfolio NLP API",
    description="Detect technologies and classify project domains",
    version="1.0.0",
    lifespan=lifespan
)

class PredictionRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "Portfolio NLP API is running"}


@app.get("/health")
def health_check():
    return {"status": "ok", "model": "xlm-roberta-base"}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        return predict_text(
            text=request.text,
            session=session,
            tokenizer=tokenizer,
            tech_normaliser=tech_normaliser,
            domain_threshold=0.2,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))