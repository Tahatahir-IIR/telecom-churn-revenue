-- Reference schema for PostgreSQL. `python -m churn.db` creates these tables with
-- pandas.to_sql; this file documents the shape for reviewers and Power BI users.

CREATE TABLE IF NOT EXISTS customers (
    customer_id        TEXT PRIMARY KEY,
    gender             TEXT,
    senior_citizen     INTEGER,
    partner            TEXT,
    dependents         TEXT,
    tenure             INTEGER,
    phone_service      TEXT,
    multiple_lines     TEXT,
    internet_service   TEXT,
    online_security    TEXT,
    online_backup      TEXT,
    device_protection  TEXT,
    tech_support       TEXT,
    streaming_tv       TEXT,
    streaming_movies   TEXT,
    contract           TEXT,
    paperless_billing  TEXT,
    payment_method     TEXT,
    monthly_charges    NUMERIC(10, 2),
    total_charges      NUMERIC(12, 2),
    churn              INTEGER,
    tenure_band        TEXT
);

CREATE TABLE IF NOT EXISTS customer_scores (
    customer_id           TEXT PRIMARY KEY REFERENCES customers (customer_id),
    monthly_charges       NUMERIC(10, 2),
    contract              TEXT,
    internet_service      TEXT,
    tenure                INTEGER,
    tenure_band           TEXT,
    payment_method        TEXT,
    churn_probability     DOUBLE PRECISION,
    risk_band             TEXT,
    expected_revenue_12m  NUMERIC(12, 2),
    revenue_at_risk_12m   NUMERIC(12, 2),
    actual_churn          INTEGER,
    is_test_row           INTEGER,
    scored_at             TIMESTAMPTZ,
    model_version         TEXT
);

CREATE TABLE IF NOT EXISTS model_metrics (
    model_name     TEXT,
    split          TEXT,          -- 'cv' or 'test'
    metric         TEXT,
    value          DOUBLE PRECISION,
    model_version  TEXT,
    trained_at     TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS feature_importance (
    feature          TEXT,
    importance_mean  DOUBLE PRECISION,
    importance_std   DOUBLE PRECISION,
    model_version    TEXT
);
