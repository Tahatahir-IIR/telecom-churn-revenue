"""Static preview of the Power BI report, drawn from the same SQL views.

Usage:  python -m churn.dashboard_preview

This is a stand-in until the real Power BI screenshot is added. Every panel
reads a view from the database, so the numbers match what Power BI shows.
"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402

from .config import DATABASE_URL, FIGURES_DIR  # noqa: E402

BLUE, ORANGE, GREY, RED = "#2F5D8A", "#E8792B", "#9AA5B1", "#C0392B"
BAND_ORDER = ["low", "medium", "high"]
TENURE_ORDER = ["0-6m", "7-12m", "13-24m", "25-48m", "49m+"]


def main() -> int:
    engine = create_engine(DATABASE_URL)
    risk = pd.read_sql("SELECT * FROM v_customer_risk", engine)
    by_band = pd.read_sql("SELECT * FROM v_risk_band_summary", engine).set_index("risk_band").loc[BAND_ORDER]
    by_contract = pd.read_sql("SELECT * FROM v_churn_by_contract", engine).set_index("contract")
    by_tenure = pd.read_sql("SELECT * FROM v_churn_by_tenure_band", engine).set_index("tenure_band").loc[TENURE_ORDER]
    top = pd.read_sql("SELECT * FROM v_top_revenue_at_risk LIMIT 8", engine)
    metrics = pd.read_sql(
        "SELECT metric, value FROM model_metrics WHERE model_name='selected_calibrated' AND split='test'", engine
    ).set_index("metric")["value"]

    fig = plt.figure(figsize=(15, 9.5), facecolor="white")
    fig.suptitle("Telecom churn and revenue at risk", x=0.02, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.02, 0.945, "Dashboard preview generated from the Power BI views (v_customer_risk, v_risk_band_summary, "
             "v_churn_by_contract, v_churn_by_tenure_band, v_top_revenue_at_risk)", fontsize=9, color=GREY)
    gs = fig.add_gridspec(3, 4, left=0.04, right=0.98, top=0.90, bottom=0.05, hspace=0.55, wspace=0.35)

    # KPI cards.
    kpis = [
        ("Customers", f"{len(risk):,}"),
        ("Churn rate (actual)", f"{risk['actual_churn'].mean():.1%}"),
        ("Revenue at risk, 12m", f"{risk['revenue_at_risk_12m'].sum():,.0f}"),
        ("High-risk customers", f"{int(by_band.loc['high', 'customers']):,}"),
        ("Test ROC AUC", f"{metrics['roc_auc']:.3f}"),
    ]
    kpi_gs = gs[0, :].subgridspec(1, 5, wspace=0.15)
    for i, (label, value) in enumerate(kpis):
        ax = fig.add_subplot(kpi_gs[0, i])
        ax.set_axis_off()
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color="#F2F4F7", zorder=0))
        ax.text(0.08, 0.62, value, fontsize=20, fontweight="bold", color=BLUE, transform=ax.transAxes)
        ax.text(0.08, 0.22, label, fontsize=10, color="#3D4852", transform=ax.transAxes)

    # Risk bands.
    ax = fig.add_subplot(gs[1, 0])
    ax.bar(by_band.index, by_band["customers"], color=[GREY, ORANGE, RED])
    ax.set_title("Customers by risk band", fontsize=11, loc="left")
    ax.bar_label(ax.containers[0], fmt="{:,.0f}", fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    ax.bar(by_band.index, by_band["revenue_at_risk_12m"], color=[GREY, ORANGE, RED])
    ax.set_title("Revenue at risk (12m) by band", fontsize=11, loc="left")
    ax.bar_label(ax.containers[0], fmt="{:,.0f}", fontsize=8)

    # Actual vs predicted by contract.
    ax = fig.add_subplot(gs[1, 2])
    x = range(len(by_contract))
    ax.bar([i - 0.2 for i in x], by_contract["actual_churn_rate"], width=0.4, color=BLUE, label="actual")
    ax.bar([i + 0.2 for i in x], by_contract["predicted_churn_rate"], width=0.4, color=ORANGE, label="predicted")
    ax.set_xticks(list(x), by_contract.index, fontsize=8)
    ax.set_title("Churn rate by contract: actual vs predicted", fontsize=11, loc="left")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(fontsize=8, frameon=False)

    # Tenure bands.
    ax = fig.add_subplot(gs[1, 3])
    ax.bar(by_tenure.index, by_tenure["revenue_at_risk_12m"], color=BLUE)
    ax.set_title("Revenue at risk (12m) by tenure", fontsize=11, loc="left")
    ax.tick_params(axis="x", labelsize=8)

    # Probability distribution.
    ax = fig.add_subplot(gs[2, :2])
    ax.hist(risk.loc[risk["actual_churn"] == 0, "churn_probability"], bins=30, alpha=0.7, color=BLUE, label="stayed")
    ax.hist(risk.loc[risk["actual_churn"] == 1, "churn_probability"], bins=30, alpha=0.7, color=ORANGE, label="churned")
    ax.axvline(0.3, color=GREY, ls="--", lw=0.8)
    ax.axvline(0.6, color=GREY, ls="--", lw=0.8)
    ax.set_title("Churn probability distribution, with band cut-offs", fontsize=11, loc="left")
    ax.legend(fontsize=8, frameon=False)

    # Retention list.
    ax = fig.add_subplot(gs[2, 2:])
    ax.set_axis_off()
    ax.set_title("Retention call list: top revenue at risk", fontsize=11, loc="left")
    cols = ["customer_id", "contract", "tenure", "monthly_charges", "churn_probability", "revenue_at_risk_12m"]
    cell = top[cols].copy()
    cell["churn_probability"] = cell["churn_probability"].map("{:.2f}".format)
    cell["revenue_at_risk_12m"] = cell["revenue_at_risk_12m"].map("{:,.0f}".format)
    tbl = ax.table(cellText=cell.values, colLabels=["id", "contract", "tenure", "monthly", "p(churn)", "at risk 12m"],
                   loc="upper center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.25)
    for (r, _c), c in tbl.get_celld().items():
        c.set_edgecolor("#E2E6EA")
        if r == 0:
            c.set_facecolor("#F2F4F7")
            c.set_text_props(fontweight="bold")

    for a in fig.axes:
        a.spines[["top", "right"]].set_visible(False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "dashboard_preview.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
