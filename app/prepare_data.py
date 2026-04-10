"""
app/prepare_data.py
Stage 1 — Load and explore the Telco Customer Churn dataset.

Dataset: IBM Telco Customer Churn (free, from Kaggle or IBM)
Link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Download the CSV and place it at: data/telco_churn.csv

Run:
    python app/prepare_data.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "./data/telco_churn.csv")


def load_data(path: str) -> pd.DataFrame:
    """Load the CSV and do basic validation."""
    if not os.path.exists(path):
        print(f"ERROR: Dataset not found at {path}")
        print("Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        print("Then place it at: data/telco_churn.csv")
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df


def explore(df: pd.DataFrame) -> pd.DataFrame:
    """Print key stats and return cleaned dataframe."""
    print("\n--- Column types ---")
    print(df.dtypes)

    print("\n--- Missing values ---")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    print("\n--- Target distribution ---")
    print(df["Churn"].value_counts())
    churn_rate = df["Churn"].value_counts(normalize=True)["Yes"] * 100
    print(f"Churn rate: {churn_rate:.1f}%")

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset:
    - Fix TotalCharges which is stored as string with spaces
    - Convert Churn to binary 0/1
    - Drop customerID (not a feature)
    """
    # TotalCharges has empty strings — convert to numeric, fill with 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Convert target to binary
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Drop ID column — not a predictive feature
    df = df.drop(columns=["customerID"])

    print(f"\nCleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def plot_churn_distribution(df: pd.DataFrame):
    """Save a bar chart of churn vs no-churn counts."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Churn counts
    churn_counts = df["Churn"].value_counts()
    axes[0].bar(["No Churn", "Churn"], churn_counts.values,
                color=["#2E5FAB", "#993C1D"])
    axes[0].set_title("Churn Distribution")
    axes[0].set_ylabel("Number of customers")
    for i, v in enumerate(churn_counts.values):
        axes[0].text(i, v + 20, str(v), ha="center", fontweight="bold")

    # Monthly charges by churn
    df.groupby("Churn")["MonthlyCharges"].hist(
        ax=axes[1], bins=30, alpha=0.7,
        color=["#2E5FAB", "#993C1D"], label=["No Churn", "Churn"])
    axes[1].set_title("Monthly Charges by Churn")
    axes[1].set_xlabel("Monthly Charges ($)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("./data/churn_distribution.png", dpi=120, bbox_inches="tight")
    print("Saved chart: data/churn_distribution.png")
    plt.close()


def save_clean(df: pd.DataFrame):
    """Save the cleaned dataset for the pipeline to use."""
    out = "./data/telco_churn_clean.csv"
    df.to_csv(out, index=False)
    print(f"Saved clean dataset: {out}")


if __name__ == "__main__":
    print("\n=== Churn Predictor — Stage 1: Data Preparation ===\n")

    df = load_data(DATA_PATH)
    df = explore(df)
    df = clean(df)
    #plot_churn_distribution(df)
    save_clean(df)

    print("\nStage 1 complete.")
    print("Next: run python app/train.py")
