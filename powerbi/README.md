# Power BI report

The report reads the views created by `python -m churn.db`. No transformation happens in Power BI: every number on the dashboard comes from a SQL view, so it can be traced back to `customer_scores`.

## Connect

1. Power BI Desktop, Get data, PostgreSQL database.
2. Server `localhost:5432` (or the Azure host), database `churn`. User `churn`, password `churn` for the Docker setup.
3. Import mode. Select these views:
   - `v_customer_risk` (one row per customer, the fact table)
   - `v_churn_by_contract`
   - `v_churn_by_tenure_band`
   - `v_risk_band_summary`
   - `v_top_revenue_at_risk`
   - `model_metrics` and `feature_importance` (tables)

No Postgres available? Point Power BI at the SQLite file `data/churn.db` through the ODBC connector, or import `data/processed/customer_scores.csv` directly. The view names are the same.

## Measures (DAX)

Create these on `v_customer_risk`:

```dax
Customers = COUNTROWS('v_customer_risk')

Churn rate (actual) = AVERAGE('v_customer_risk'[actual_churn])

Churn rate (predicted) = AVERAGE('v_customer_risk'[churn_probability])

Monthly revenue = SUM('v_customer_risk'[monthly_charges])

Revenue at risk 12m = SUM('v_customer_risk'[revenue_at_risk_12m])

Expected revenue 12m = SUM('v_customer_risk'[expected_revenue_12m])

Revenue at risk share =
DIVIDE([Revenue at risk 12m], [Monthly revenue] * 12)

High-risk customers =
CALCULATE([Customers], 'v_customer_risk'[risk_band] = "high")

High-risk revenue at risk =
CALCULATE([Revenue at risk 12m], 'v_customer_risk'[risk_band] = "high")

Test AUC =
CALCULATE(
    MAX('model_metrics'[value]),
    'model_metrics'[model_name] = "selected_calibrated",
    'model_metrics'[split] = "test",
    'model_metrics'[metric] = "roc_auc"
)
```

## Pages

**1. Overview**
- KPI cards: Customers, Churn rate (actual), Revenue at risk 12m, High-risk customers, Test AUC.
- Clustered bar: `v_risk_band_summary`, customers and revenue_at_risk_12m by risk_band.
- Slicers: contract, internet_service, tenure_band, payment_method.

**2. Segments**
- Bar: actual vs predicted churn rate by contract (`v_churn_by_contract`). The two bars should sit close together; if they diverge the model is miscalibrated on that segment.
- Bar: revenue at risk by tenure_band.
- Matrix: contract × internet_service, values Churn rate (actual) and Revenue at risk 12m.

**3. Retention list**
- Table from `v_top_revenue_at_risk`: customer_id, contract, tenure, monthly_charges, churn_probability, revenue_at_risk_12m, sorted by revenue at risk. This is the call list for a retention team.

**4. Model**
- Bar: `feature_importance`, importance_mean by feature, sorted.
- Table: `model_metrics` filtered to split = test, pivoted by model_name.

## Refresh

Re-run `python -m churn.train && python -m churn.db` and refresh the report. `scored_at` and `model_version` on every row tell you which run you are looking at.
