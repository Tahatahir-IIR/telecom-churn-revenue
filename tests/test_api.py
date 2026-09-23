import pytest
from fastapi.testclient import TestClient

from churn.config import MODEL_PATH

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="model not trained")

from api.main import app  # noqa: E402

client = TestClient(app)

EXAMPLE = {
    "customer_id": "test-1",
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure": 1,
    "phone_service": "No",
    "multiple_lines": "No phone service",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 95.0,
    "total_charges": 95.0,
}

LOYAL = {
    **EXAMPLE,
    "customer_id": "test-2",
    "tenure": 70,
    "contract": "Two year",
    "tech_support": "Yes",
    "online_security": "Yes",
    "payment_method": "Credit card (automatic)",
    "total_charges": 6650.0,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_high_risk_vs_loyal():
    hi = client.post("/predict", json=EXAMPLE).json()
    lo = client.post("/predict", json=LOYAL).json()
    assert 0 <= hi["churn_probability"] <= 1
    assert hi["churn_probability"] > lo["churn_probability"]
    assert hi["revenue_at_risk_12m"] > lo["revenue_at_risk_12m"]
    assert hi["risk_band"] in ("medium", "high")
    assert lo["risk_band"] == "low"


def test_batch_and_validation():
    r = client.post("/predict/batch", json={"customers": [EXAMPLE, LOYAL]})
    assert r.status_code == 200 and len(r.json()) == 2
    bad = client.post("/predict", json={**EXAMPLE, "contract": "Forever"})
    assert bad.status_code == 422
