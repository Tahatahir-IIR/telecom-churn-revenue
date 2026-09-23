"""FastAPI scoring service for the churn model.

    uvicorn api.main:app --reload
"""
from __future__ import annotations

import json
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from churn import __version__
from churn.config import METRICS_PATH
from churn.predict import load_artifact, predict

YesNo = Literal["Yes", "No"]

app = FastAPI(
    title="Telecom churn and revenue-at-risk API",
    version=__version__,
    description="Scores a customer's monthly churn probability and the 12-month revenue at risk.",
)


class Customer(BaseModel):
    customer_id: str = "unknown"
    gender: Literal["Male", "Female"]
    senior_citizen: int = Field(ge=0, le=1)
    partner: YesNo
    dependents: YesNo
    tenure: int = Field(ge=0, le=120)
    phone_service: YesNo
    multiple_lines: Literal["Yes", "No", "No phone service"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: Literal["Yes", "No", "No internet service"]
    online_backup: Literal["Yes", "No", "No internet service"]
    device_protection: Literal["Yes", "No", "No internet service"]
    tech_support: Literal["Yes", "No", "No internet service"]
    streaming_tv: Literal["Yes", "No", "No internet service"]
    streaming_movies: Literal["Yes", "No", "No internet service"]
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: YesNo
    payment_method: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    monthly_charges: float = Field(ge=0)
    total_charges: float | None = Field(default=None, ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_id": "7590-VHVEG",
                "gender": "Female",
                "senior_citizen": 0,
                "partner": "Yes",
                "dependents": "No",
                "tenure": 1,
                "phone_service": "No",
                "multiple_lines": "No phone service",
                "internet_service": "DSL",
                "online_security": "No",
                "online_backup": "Yes",
                "device_protection": "No",
                "tech_support": "No",
                "streaming_tv": "No",
                "streaming_movies": "No",
                "contract": "Month-to-month",
                "paperless_billing": "Yes",
                "payment_method": "Electronic check",
                "monthly_charges": 29.85,
                "total_charges": 29.85,
            }
        }
    }


class Prediction(BaseModel):
    customer_id: str
    churn_probability: float
    churn_predicted: bool
    risk_band: Literal["low", "medium", "high"]
    expected_revenue_12m: float
    revenue_at_risk_12m: float


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=5000)


def _records(customers: list[Customer]) -> list[dict]:
    recs = []
    for c in customers:
        d = c.model_dump()
        if d["total_charges"] is None:
            d["total_charges"] = d["monthly_charges"] * d["tenure"]
        recs.append(d)
    return recs


@app.get("/health")
def health() -> dict:
    try:
        art = load_artifact()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="model not trained; run `python -m churn.train`")
    return {"status": "ok", "model_version": art["version"], "base_model": art["base_model"]}


@app.get("/model")
def model_info() -> dict:
    art = load_artifact()
    info = {
        "version": art["version"],
        "base_model": art["base_model"],
        "threshold": art["threshold"],
        "trained_at": art["trained_at"],
        "features": art["features"],
    }
    if METRICS_PATH.exists():
        m = json.loads(METRICS_PATH.read_text())
        info["test_metrics"] = m["models"]["selected_calibrated"]["test"]
        info["revenue"] = m["revenue"]
    return info


@app.post("/predict", response_model=Prediction)
def predict_one(customer: Customer) -> Prediction:
    res = predict(_records([customer]))[0]
    return Prediction(customer_id=customer.customer_id, **res)


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(body: BatchRequest) -> list[Prediction]:
    results = predict(_records(body.customers))
    return [Prediction(customer_id=c.customer_id, **r) for c, r in zip(body.customers, results)]
