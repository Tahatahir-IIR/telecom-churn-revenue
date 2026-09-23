"""Load scores, metrics and feature importance into the database used by Power BI.

Usage:  python -m churn.db

Uses DATABASE_URL (PostgreSQL in docker compose or in the cloud). Without it,
falls back to SQLite in data/churn.db so the whole pipeline runs on a laptop.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import create_engine, text

from .config import DATABASE_URL, IMPORTANCE_PATH, METRICS_PATH, ROOT, SCORES_PATH
from .data import load_dataset

SQL_DIR = ROOT / "sql"


def run_sql_file(engine, path) -> None:
    raw = path.read_text(encoding="utf-8")
    with engine.begin() as conn:
        for stmt in [s.strip() for s in raw.split(";") if s.strip()]:
            conn.execute(text(stmt))


def metrics_long(metrics: dict) -> pd.DataFrame:
    rows = []
    for model_name, res in metrics["models"].items():
        for split in ("cv", "test"):
            if split not in res:
                continue
            for metric, value in res[split].items():
                v = value["mean"] if isinstance(value, dict) else value
                rows.append(
                    {
                        "model_name": model_name,
                        "split": split,
                        "metric": metric,
                        "value": float(v),
                        "model_version": metrics["version"],
                        "trained_at": metrics["trained_at"],
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    engine = create_engine(DATABASE_URL)
    is_sqlite = engine.dialect.name == "sqlite"
    print(f"database: {engine.dialect.name}")

    customers = load_dataset().drop(columns=["avg_monthly_spend", "num_services"])
    scores = pd.read_csv(SCORES_PATH)
    metrics = json.loads(METRICS_PATH.read_text())
    imp = pd.read_csv(IMPORTANCE_PATH)

    scores["scored_at"] = datetime.now(timezone.utc)
    scores["model_version"] = metrics["version"]
    imp["model_version"] = metrics["version"]

    customers.to_sql("customers", engine, if_exists="replace", index=False)
    scores.to_sql("customer_scores", engine, if_exists="replace", index=False)
    metrics_long(metrics).to_sql("model_metrics", engine, if_exists="replace", index=False)
    imp.to_sql("feature_importance", engine, if_exists="replace", index=False)

    run_sql_file(engine, SQL_DIR / ("views_sqlite.sql" if is_sqlite else "views.sql"))
    with engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM v_customer_risk")).scalar()
    print(f"loaded customers={len(customers)} scores={len(scores)} views ok (v_customer_risk rows={n})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
