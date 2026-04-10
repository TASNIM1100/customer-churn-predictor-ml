"""
main.py — FastAPI REST API for the Churn Predictor
Entry point for the prediction service.

Run:
    python -m uvicorn main:app --reload

Endpoints:
    GET  /           — health check
    GET  /status     — model info
    POST /predict    — predict churn for one customer
    POST /predict/batch — predict for multiple customers
    GET  /docs       — Swagger UI
"""

import os
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "./models/churn_pipeline.joblib")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

# ── Load model at startup ─────────────────────────────────────────────────────
pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("Loading churn prediction model...")
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Run python app/train.py first.")
    else:
        pipeline = joblib.load(MODEL_PATH)
        print(f"Model loaded from: {MODEL_PATH}")
    yield
    print("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Predictor",
    description=(
        "Predicts the probability that a telecom customer will churn. "
        "Built with XGBoost and scikit-learn pipelines."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response models ─────────────────────────────────────────────────
class Customer(BaseModel):
    gender: str = Field(example="Female")
    SeniorCitizen: int = Field(example=0)
    Partner: str = Field(example="No")
    Dependents: str = Field(example="No")
    tenure: int = Field(example=2)
    PhoneService: str = Field(example="Yes")
    MultipleLines: str = Field(example="No")
    InternetService: str = Field(example="Fiber optic")
    OnlineSecurity: str = Field(example="No")
    OnlineBackup: str = Field(example="No")
    DeviceProtection: str = Field(example="No")
    TechSupport: str = Field(example="No")
    StreamingTV: str = Field(example="No")
    StreamingMovies: str = Field(example="No")
    Contract: str = Field(example="Month-to-month")
    PaperlessBilling: str = Field(example="Yes")
    PaymentMethod: str = Field(example="Electronic check")
    MonthlyCharges: float = Field(example=85.0)
    TotalCharges: float = Field(example=170.0)


class Prediction(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: str
    threshold_used: float


class BatchPrediction(BaseModel):
    predictions: list[Prediction]
    total_customers: int
    predicted_churners: int
    churn_rate: float


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "running",
        "message": "Churn Predictor API is live. POST to /predict.",
        "docs": "/docs",
    }


@app.get("/status", tags=["Health"])
def status():
    return {
        "model_path": MODEL_PATH,
        "model_loaded": pipeline is not None,
        "threshold": THRESHOLD,
    }


@app.post("/predict", response_model=Prediction, tags=["Prediction"])
def predict(customer: Customer):
    """
    Predict churn probability for a single customer.
    Returns probability, binary prediction, and risk level.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run python app/train.py first."
        )

    try:
        df = pd.DataFrame([customer.model_dump()])
        prob = float(pipeline.predict_proba(df)[0][1])
        prediction = int(prob >= THRESHOLD)

        return Prediction(
            churn_probability=round(prob, 4),
            churn_prediction=prediction,
            risk_level=(
                "High" if prob >= 0.7
                else "Medium" if prob >= 0.4
                else "Low"
            ),
            threshold_used=THRESHOLD,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPrediction, tags=["Prediction"])
def predict_batch(customers: list[Customer]):
    """
    Predict churn for a list of customers in one call.
    Returns individual predictions plus aggregate stats.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if not customers:
        raise HTTPException(status_code=400, detail="Customer list cannot be empty.")

    try:
        df = pd.DataFrame([c.model_dump() for c in customers])
        probs = pipeline.predict_proba(df)[:, 1]

        predictions = []
        for prob in probs:
            prob = float(prob)
            predictions.append(Prediction(
                churn_probability=round(prob, 4),
                churn_prediction=int(prob >= THRESHOLD),
                risk_level=(
                    "High" if prob >= 0.7
                    else "Medium" if prob >= 0.4
                    else "Low"
                ),
                threshold_used=THRESHOLD,
            ))

        churners = sum(p.churn_prediction for p in predictions)

        return BatchPrediction(
            predictions=predictions,
            total_customers=len(predictions),
            predicted_churners=churners,
            churn_rate=round(churners / len(predictions), 4),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
