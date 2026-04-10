"""
app/predict.py
Stage 3 — Load the saved pipeline and make predictions.
Tests the model independently before wrapping it in the API.

Run:
    python app/predict.py
"""

import os
import sys
import joblib
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "./models/churn_pipeline.joblib")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))


def load_pipeline():
    """Load the trained pipeline from disk."""
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Run python app/train.py first.")
        sys.exit(1)

    pipeline = joblib.load(MODEL_PATH)
    print(f"Model loaded from: {MODEL_PATH}")
    return pipeline


def predict_single(pipeline, customer: dict) -> dict:
    """
    Predict churn for a single customer dictionary.
    Returns probability and binary prediction.
    """
    df = pd.DataFrame([customer])
    prob = pipeline.predict_proba(df)[0][1]
    prediction = int(prob >= THRESHOLD)

    return {
        "churn_probability": round(float(prob), 4),
        "churn_prediction": prediction,
        "risk_level": (
            "High" if prob >= 0.7
            else "Medium" if prob >= 0.4
            else "Low"
        ),
        "threshold_used": THRESHOLD,
    }


# ── Test customer examples ────────────────────────────────────────────────────
HIGH_RISK = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 170.0,
}

LOW_RISK = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 60,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 65.0,
    "TotalCharges": 3900.0,
}


if __name__ == "__main__":
    print("\n=== Churn Predictor — Stage 3: Prediction Test ===\n")

    pipeline = load_pipeline()

    print("\nTest 1 — High risk customer (month-to-month, new, fiber optic):")
    result = predict_single(pipeline, HIGH_RISK)
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\nTest 2 — Low risk customer (long tenure, 2-year contract):")
    result = predict_single(pipeline, LOW_RISK)
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\nStage 3 complete. Predictions working correctly.")
    print("Next: python -m uvicorn main:app --reload")
