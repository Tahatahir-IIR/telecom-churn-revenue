import numpy as np

from churn.config import FEATURES, TARGET
from churn.data import load_dataset
from churn.revenue import expected_revenue, revenue_at_risk, risk_band


def test_dataset_shape_and_types():
    df = load_dataset()
    assert len(df) == 7043
    assert df[TARGET].isin([0, 1]).all()
    assert df["total_charges"].isna().sum() == 0
    assert set(FEATURES) <= set(df.columns)
    assert df["num_services"].between(0, 8).all()


def test_revenue_formulas():
    # Never churns: full horizon revenue, nothing at risk.
    assert expected_revenue(100.0, 0.0) == 1200.0
    assert revenue_at_risk(100.0, 0.0) == 0.0
    # Certain churn after this month: one month of revenue.
    assert np.isclose(expected_revenue(100.0, 1.0), 100.0)
    assert np.isclose(revenue_at_risk(100.0, 1.0), 1100.0)
    # Monotonic in probability.
    p = np.linspace(0, 1, 11)
    e = expected_revenue(50.0, p)
    assert np.all(np.diff(e) <= 0)


def test_risk_bands():
    assert list(risk_band([0.1, 0.3, 0.59, 0.6, 0.99])) == ["low", "medium", "medium", "high", "high"]
