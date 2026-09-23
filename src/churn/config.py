from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = ROOT / "data" / "raw" / "telco_customer_churn.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = MODELS_DIR / "churn_model.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"
IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
SCORES_PATH = PROCESSED_DIR / "customer_scores.csv"

TARGET = "churn"
ID_COL = "customer_id"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Revenue projection horizon in months.
HORIZON_MONTHS = 12

# Risk bands on churn probability.
RISK_BANDS = [(0.0, 0.30, "low"), (0.30, 0.60, "medium"), (0.60, 1.01, "high")]

# Database. Postgres in Docker / cloud, SQLite fallback for a laptop without Postgres.
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'data' / 'churn.db'}")

NUMERIC_FEATURES = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "senior_citizen",
    "avg_monthly_spend",
    "num_services",
]

CATEGORICAL_FEATURES = [
    "gender",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
    "tenure_band",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
