import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from xgboost import XGBClassifier

load_dotenv()

DATA_PATH = "./data/telco_churn_clean.csv"
MODEL_PATH = os.getenv("MODEL_PATH", "./models/churn_pipeline.joblib")

# ── Feature definitions ───────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges"
]

CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

TARGET = "Churn"


def load_data(path: str):
    if not os.path.exists(path):
        print(f"ERROR: Clean dataset not found at {path}")
        print("Run python app/prepare_data.py first.")
        sys.exit(1)

    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Churn rate: {y.mean()*100:.1f}%")
    return X, y


def build_pipeline() -> Pipeline:
    """
    scikit-learn Pipeline chains preprocessing + model into one object.
    This is how production ML systems are built — the pipeline handles
    everything from raw input to prediction in a single .predict() call.
    """

    # Numeric features: fill missing → scale to same range
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categorical features: fill missing → one-hot encode
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # ColumnTransformer applies different processing to different columns
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    # XGBoost: gradient boosting — the most common algorithm in production ML
    # scale_pos_weight handles class imbalance (more non-churners than churners)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=2.7,   # ratio of negative/positive class
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )

    # Final pipeline: preprocessor → XGBoost
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return pipeline


def evaluate(pipeline, X_test, y_test):
    """Print all evaluation metrics — what interviewers ask about."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.3f}")

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred,
                                target_names=["No Churn", "Churn"]))

    return y_pred, y_prob


def plot_confusion_matrix(y_test, y_pred):
    """Visualise true vs predicted labels."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("./data/confusion_matrix.png", dpi=120, bbox_inches="tight")
    print("Saved: data/confusion_matrix.png")
    plt.close()


def plot_roc_curve(y_test, y_prob):
    """ROC curve shows model performance across all thresholds."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#2E5FAB", lw=2, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./data/roc_curve.png", dpi=120, bbox_inches="tight")
    print("Saved: data/roc_curve.png")
    plt.close()


def plot_feature_importance(pipeline, top_n=15):
    """Show which features drive churn the most."""
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    # Get feature names after one-hot encoding
    cat_features = (pipeline.named_steps["preprocessor"]
                    .named_transformers_["cat"]
                    .named_steps["encoder"]
                    .get_feature_names_out(CATEGORICAL_FEATURES))
    all_features = NUMERIC_FEATURES + list(cat_features)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(8, 5))
    plt.barh(
        [all_features[i] for i in indices][::-1],
        importances[indices][::-1],
        color="#2E5FAB"
    )
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("./data/feature_importance.png", dpi=120, bbox_inches="tight")
    print("Saved: data/feature_importance.png")
    plt.close()


def cross_validate(pipeline, X, y):
    """5-fold cross-validation gives a more reliable score than one split."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv,
                             scoring="roc_auc", n_jobs=-1)
    print(f"\n5-Fold Cross-Validation ROC-AUC: {scores.mean():.3f} (+/- {scores.std():.3f})")


if __name__ == "__main__":
    print("\n=== Churn Predictor — Stage 2: Training ===\n")

    # Load data
    X, y = load_data(DATA_PATH)

    # Train / test split — stratified keeps churn ratio consistent
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # Build pipeline
    print("\nBuilding pipeline...")
    pipeline = build_pipeline()

    # Train
    print("Training XGBoost...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # Cross-validate
    cross_validate(pipeline, X_train, y_train)

    # Evaluate on test set
    y_pred, y_prob = evaluate(pipeline, X_test, y_test)

    # Plots
    print("\nGenerating charts...")
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)
    plot_feature_importance(pipeline)

    # Save the trained pipeline
    os.makedirs("./models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved: {MODEL_PATH}")

    print("\nStage 2 complete.")
    print("Next: run python app/predict.py to test predictions")
    print("Then: python -m uvicorn main:app --reload to start the API")
