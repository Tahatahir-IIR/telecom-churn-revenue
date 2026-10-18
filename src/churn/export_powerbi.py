"""Export the reporting views to CSV for Power BI Desktop.

Usage:  python -m churn.export_powerbi

Writes powerbi/data/*.csv, the files churn_report.pbix imports, so the report
opens without PostgreSQL or an ODBC driver. Run it after python -m churn.db.

v_customer_risk gets two extra columns, tenure_band_order and risk_band_order.
Power BI sorts text alphabetically ("13-24m" before "7-12m"); with "Sort by
column" on these, tenure bands and risk levels keep their natural order.
"""
from __future__ import annotations

import csv
import sys

from sqlalchemy import create_engine, text

from .config import DATABASE_URL, ROOT

OUT_DIR = ROOT / "powerbi" / "data"

TABLES = [
    "v_customer_risk",
    "v_churn_by_contract",
    "v_churn_by_tenure_band",
    "v_risk_band_summary",
    "v_top_revenue_at_risk",
    "model_metrics",
    "feature_importance",
]

TENURE_ORDER = {"0-6m": 1, "7-12m": 2, "13-24m": 3, "25-48m": 4, "49m+": 5}
RISK_ORDER = {"low": 1, "medium": 2, "high": 3}


def main() -> int:
    engine = create_engine(DATABASE_URL)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with engine.connect() as conn:
        for name in TABLES:
            result = conn.execute(text(f"SELECT * FROM {name}"))
            cols = list(result.keys())
            rows = [list(r) for r in result]
            if name == "v_customer_risk":
                t, r = cols.index("tenure_band"), cols.index("risk_band")
                cols += ["tenure_band_order", "risk_band_order"]
                rows = [row + [TENURE_ORDER[row[t]], RISK_ORDER[row[r]]] for row in rows]
            # utf-8-sig so Power BI and Excel detect the encoding
            with open(OUT_DIR / f"{name}.csv", "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(cols)
                writer.writerows(rows)
            print(f"{name}.csv  {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
