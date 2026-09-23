"""Load and clean the IBM Telco Customer Churn dataset."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_DATA, TARGET

SERVICE_COLS = [
    "phone_service",
    "multiple_lines",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]

_ACRONYM = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL = re.compile(r"([a-z0-9])([A-Z])")


def _snake(name: str) -> str:
    """CamelCase to snake_case, keeping acronyms together (StreamingTV -> streaming_tv)."""
    name = name.replace("customerID", "customer_id")
    name = _ACRONYM.sub(lambda m: m.group(1) + "_" + m.group(2), name)
    name = _CAMEL.sub(lambda m: m.group(1) + "_" + m.group(2), name)
    return name.lower()


def load_raw(path: Path = RAW_DATA) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [_snake(c) for c in df.columns]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types, fill the 11 blank total_charges rows, and map the target to 0/1."""
    df = df.copy()
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    # Blank total_charges only happens for tenure == 0 (new customers, nothing billed yet).
    df["total_charges"] = df["total_charges"].fillna(0.0)
    df["senior_citizen"] = df["senior_citizen"].astype(int)
    df[TARGET] = (df[TARGET] == "Yes").astype(int)

    cat_cols = df.select_dtypes(exclude="number").columns.difference(["customer_id"])
    for c in cat_cols:
        df[c] = df[c].astype(object)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered features. Applied identically at train and inference time."""
    df = df.copy()
    tenure = df["tenure"].replace(0, np.nan)
    df["avg_monthly_spend"] = (df["total_charges"] / tenure).fillna(df["monthly_charges"])
    df["num_services"] = sum((df[c] == "Yes").astype(int) for c in SERVICE_COLS)
    df["tenure_band"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, 200],
        labels=["0-6m", "7-12m", "13-24m", "25-48m", "49m+"],
    ).astype(object)
    return df


def load_dataset(path: Path = RAW_DATA) -> pd.DataFrame:
    return add_features(clean(load_raw(path)))
