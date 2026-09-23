"""Inference helpers used by the API and the CLI."""
from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from .config import FEATURES, MODEL_PATH
from .data import add_features
from .revenue import expected_revenue, revenue_at_risk, risk_band


@lru_cache(maxsize=1)
def load_artifact(path=MODEL_PATH) -> dict:
    return joblib.load(path)


def prepare(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce").fillna(0.0)
    df["senior_citizen"] = df["senior_citizen"].astype(int)
    for c in df.select_dtypes(exclude="number").columns:
        df[c] = df[c].astype(object)
    return add_features(df)


def predict(records: list[dict], artifact: dict | None = None) -> list[dict]:
    artifact = artifact or load_artifact()
    df = prepare(records)
    proba = artifact["model"].predict_proba(df[FEATURES])[:, 1]
    thr = artifact["threshold"]
    out = []
    for i, p in enumerate(proba):
        mc = float(df.loc[i, "monthly_charges"])
        out.append(
            {
                "churn_probability": round(float(p), 4),
                "churn_predicted": bool(p >= thr),
                "risk_band": str(risk_band([p])[0]),
                "expected_revenue_12m": round(float(expected_revenue(mc, p)), 2),
                "revenue_at_risk_12m": round(float(revenue_at_risk(mc, p)), 2),
            }
        )
    return out
