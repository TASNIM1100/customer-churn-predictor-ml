# Customer Churn Predictor

Predicts whether a telecom customer will cancel their subscription using XGBoost
and scikit-learn pipelines. Served as a REST API with FastAPI.

## Tech Stack
| Tool | Role |
|------|------|
| pandas + NumPy | Data loading and feature engineering |
| scikit-learn Pipeline | Preprocessing + model in one deployable object |
| XGBoost | Gradient boosting classifier |
| matplotlib + seaborn | Evaluation charts |
| joblib | Model serialisation |
| FastAPI | REST API |
| Docker | Containerisation |

## Dataset
IBM Telco Customer Churn — 7,043 customers, 21 features.
Download: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Project Structure
```
churn-predictor/
├── app/
│   ├── prepare_data.py   # Stage 1: load, clean, explore data
│   ├── train.py          # Stage 2: build pipeline, train, evaluate
│   └── predict.py        # Stage 3: test predictions independently
├── data/                 # place telco_churn.csv here
├── models/               # saved pipeline saved here after training
├── main.py               # FastAPI entry point
├── requirements.txt
└── Dockerfile
```

## Setup & Run
```bash
# 1. Set up environment
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. Add dataset
# Download telco_churn.csv from Kaggle and place in data/

# 3. Run each stage in order
python app/prepare_data.py
python app/train.py
python app/predict.py

# 4. Start the API
python -m uvicorn main:app --reload
```

## API Usage
```bash
POST /predict
Content-Type: application/json

{
  "gender": "Female",
  "tenure": 2,
  "Contract": "Month-to-month",
  "MonthlyCharges": 85.0,
  ...
}
```

Response:
```json
{
  "churn_probability": 0.847,
  "churn_prediction": 1,
  "risk_level": "High",
  "threshold_used": 0.5
}
```

Interactive docs: http://localhost:8000/docs
