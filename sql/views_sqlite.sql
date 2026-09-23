-- Views consumed by the Power BI report (PostgreSQL).

DROP VIEW IF EXISTS v_customer_risk;
CREATE VIEW v_customer_risk AS
SELECT
    s.customer_id,
    c.gender,
    c.senior_citizen,
    c.partner,
    c.dependents,
    c.tenure,
    s.tenure_band,
    c.contract,
    c.internet_service,
    c.payment_method,
    c.paperless_billing,
    c.tech_support,
    c.online_security,
    s.monthly_charges,
    c.total_charges,
    s.churn_probability,
    s.risk_band,
    s.expected_revenue_12m,
    s.revenue_at_risk_12m,
    s.actual_churn,
    s.is_test_row,
    s.scored_at,
    s.model_version
FROM customer_scores s
JOIN customers c ON c.customer_id = s.customer_id;

DROP VIEW IF EXISTS v_churn_by_contract;
CREATE VIEW v_churn_by_contract AS
SELECT contract,
       COUNT(*)                 AS customers,
       AVG(actual_churn)        AS actual_churn_rate,
       AVG(churn_probability)   AS predicted_churn_rate,
       SUM(monthly_charges)     AS monthly_revenue,
       SUM(revenue_at_risk_12m) AS revenue_at_risk_12m
FROM v_customer_risk
GROUP BY contract;

DROP VIEW IF EXISTS v_churn_by_tenure_band;
CREATE VIEW v_churn_by_tenure_band AS
SELECT tenure_band,
       COUNT(*)                 AS customers,
       AVG(actual_churn)        AS actual_churn_rate,
       AVG(churn_probability)   AS predicted_churn_rate,
       SUM(revenue_at_risk_12m) AS revenue_at_risk_12m
FROM v_customer_risk
GROUP BY tenure_band;

DROP VIEW IF EXISTS v_risk_band_summary;
CREATE VIEW v_risk_band_summary AS
SELECT risk_band,
       COUNT(*)                 AS customers,
       AVG(actual_churn)        AS actual_churn_rate,
       SUM(monthly_charges)     AS monthly_revenue,
       SUM(revenue_at_risk_12m) AS revenue_at_risk_12m
FROM v_customer_risk
GROUP BY risk_band;

DROP VIEW IF EXISTS v_top_revenue_at_risk;
CREATE VIEW v_top_revenue_at_risk AS
SELECT customer_id, contract, internet_service, tenure, monthly_charges,
       churn_probability, risk_band, revenue_at_risk_12m
FROM v_customer_risk
ORDER BY revenue_at_risk_12m DESC
LIMIT 500;
