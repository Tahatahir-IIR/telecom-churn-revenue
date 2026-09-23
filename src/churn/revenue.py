"""Revenue projection from churn probability.

The dataset labels churn over one billing month, so the model's probability p is
treated as a monthly churn hazard. Expected revenue over an H-month horizon is a
geometric sum of monthly charges weighted by survival:

    E[revenue_H] = monthly_charges * sum_{m=0}^{H-1} (1 - p)^m

Revenue at risk is what is lost versus a customer who never churns:

    revenue_at_risk_H = monthly_charges * H - E[revenue_H]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HORIZON_MONTHS, RISK_BANDS


def expected_revenue(monthly_charges, churn_prob, horizon: int = HORIZON_MONTHS):
    p = np.clip(np.asarray(churn_prob, dtype=float), 0.0, 1.0)
    mc = np.asarray(monthly_charges, dtype=float)
    survival = 1.0 - p
    # Geometric series, with the p == 0 edge case handled explicitly.
    with np.errstate(divide="ignore", invalid="ignore"):
        months = np.where(p > 0, (1.0 - survival**horizon) / p, float(horizon))
    return mc * months


def revenue_at_risk(monthly_charges, churn_prob, horizon: int = HORIZON_MONTHS):
    mc = np.asarray(monthly_charges, dtype=float)
    return mc * horizon - expected_revenue(mc, churn_prob, horizon)


def risk_band(churn_prob) -> np.ndarray:
    p = np.asarray(churn_prob, dtype=float)
    out = np.full(p.shape, "low", dtype=object)
    for lo, hi, label in RISK_BANDS:
        out[(p >= lo) & (p < hi)] = label
    return out


def score_frame(df: pd.DataFrame, churn_prob, horizon: int = HORIZON_MONTHS) -> pd.DataFrame:
    """Attach probability, band and revenue columns to a customer frame."""
    out = df[["customer_id", "monthly_charges", "contract", "internet_service", "tenure", "tenure_band", "payment_method"]].copy()
    out["churn_probability"] = np.round(np.asarray(churn_prob, dtype=float), 4)
    out["risk_band"] = risk_band(churn_prob)
    out["expected_revenue_12m"] = np.round(expected_revenue(out["monthly_charges"], churn_prob, horizon), 2)
    out["revenue_at_risk_12m"] = np.round(revenue_at_risk(out["monthly_charges"], churn_prob, horizon), 2)
    return out
