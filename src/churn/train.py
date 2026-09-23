"""Train, cross-validate, evaluate and persist the churn model.

Usage:  python -m churn.train
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from . import __version__  # noqa: E402
from .config import (  # noqa: E402
    CV_FOLDS,
    FEATURES,
    FIGURES_DIR,
    HORIZON_MONTHS,
    IMPORTANCE_PATH,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    SCORES_PATH,
    TARGET,
    TEST_SIZE,
)
from .data import load_dataset  # noqa: E402
from .features import build_preprocessor  # noqa: E402
from .revenue import score_frame  # noqa: E402

SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "f1": "f1",
    "recall": "recall",
    "precision": "precision",
}


def candidate_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("prep", build_preprocessor(scale_numeric=True)),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("prep", build_preprocessor(scale_numeric=False)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("prep", build_preprocessor(scale_numeric=False)),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=300,
                        max_leaf_nodes=15,
                        l2_regularization=1.0,
                        class_weight="balanced",
                        early_stopping=True,
                        validation_fraction=0.15,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def best_threshold(y_true, proba) -> float:
    """Threshold maximising F1 on the training data.

    Recall matters more than precision for a retention campaign, and F1 is a
    reasonable compromise that does not flood the call centre with false alarms.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-9, None)
    return float(thresholds[int(np.argmax(f1))])


def holdout_metrics(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "f1": round(float(f1_score(y_true, pred)), 4),
        "recall": round(float(recall_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred)), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
        "threshold": round(threshold, 4),
    }


def plot_curves(y_test, probas: dict[str, np.ndarray]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_test, p):.3f})")
        prec, rec, _ = precision_recall_curve(y_test, p)
        axes[1].plot(rec, prec, label=f"{name} (AP {average_precision_score(y_test, p):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(title="ROC, hold-out test", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].axhline(float(np.mean(y_test)), color="k", ls="--", lw=0.8)
    axes[1].set(title="Precision-recall, hold-out test", xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_pr_curves.png", dpi=130)
    plt.close(fig)


def plot_importance(imp: pd.DataFrame) -> None:
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#4C72B0")
    ax.set(title="Permutation importance (drop in ROC AUC), hold-out test", xlabel="Mean AUC drop")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=130)
    plt.close(fig)


def main() -> int:
    t0 = time.time()
    df = load_dataset()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"rows={len(df)} train={len(X_train)} test={len(X_test)} churn_rate={y.mean():.3f}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    test_probas: dict[str, np.ndarray] = {}

    for name, pipe in candidate_models().items():
        t = time.time()
        cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
        cv_summary = {
            k: {
                "mean": round(float(cv_res[f"test_{k}"].mean()), 4),
                "std": round(float(cv_res[f"test_{k}"].std()), 4),
            }
            for k in SCORING
        }
        pipe.fit(X_train, y_train)
        train_proba = pipe.predict_proba(X_train)[:, 1]
        thr = best_threshold(y_train, train_proba)
        proba = pipe.predict_proba(X_test)[:, 1]
        results[name] = {
            "cv": cv_summary,
            "test": holdout_metrics(y_test, proba, thr),
            "fit_seconds": round(time.time() - t, 1),
        }
        fitted[name] = pipe
        test_probas[name] = proba
        print(
            f"{name:24s} cv_auc={cv_summary['roc_auc']['mean']:.4f}+/-{cv_summary['roc_auc']['std']:.4f} "
            f"test_auc={results[name]['test']['roc_auc']:.4f} test_recall={results[name]['test']['recall']:.3f}"
        )

    # Model selection on cross-validated PR AUC: churners are the minority class.
    best_name = max(results, key=lambda n: results[n]["cv"]["pr_auc"]["mean"])
    best = fitted[best_name]

    # Calibrate probabilities so that revenue-at-risk numbers are meaningful.
    calibrated = CalibratedClassifierCV(best, method="isotonic", cv=cv)
    calibrated.fit(X_train, y_train)
    cal_proba = calibrated.predict_proba(X_test)[:, 1]
    cal_thr = best_threshold(y_train, calibrated.predict_proba(X_train)[:, 1])
    results["selected_calibrated"] = {
        "base_model": best_name,
        "test": holdout_metrics(y_test, cal_proba, cal_thr),
    }
    test_probas[f"{best_name} (calibrated)"] = cal_proba
    print(
        f"selected={best_name} calibrated test_auc={results['selected_calibrated']['test']['roc_auc']:.4f} "
        f"brier={results['selected_calibrated']['test']['brier']:.4f}"
    )

    # Permutation importance on the raw input columns of the selected pipeline.
    perm = permutation_importance(
        best, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    imp = (
        pd.DataFrame(
            {"feature": FEATURES, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    imp.to_csv(IMPORTANCE_PATH, index=False)
    plot_importance(imp)
    plot_curves(y_test, test_probas)

    # Score every customer with the calibrated model and persist for the API and Power BI.
    all_proba = calibrated.predict_proba(X)[:, 1]
    scores = score_frame(df, all_proba)
    scores["actual_churn"] = y.values
    scores["is_test_row"] = df.index.isin(X_test.index).astype(int)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(SCORES_PATH, index=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": calibrated,
        "threshold": cal_thr,
        "features": FEATURES,
        "base_model": best_name,
        "version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(artifact, MODEL_PATH)

    high = scores["risk_band"] == "high"
    summary = {
        "version": __version__,
        "trained_at": artifact["trained_at"],
        "rows": int(len(df)),
        "churn_rate": round(float(y.mean()), 4),
        "cv_folds": CV_FOLDS,
        "selected_model": best_name,
        "models": results,
        "revenue": {
            "horizon_months": HORIZON_MONTHS,
            "total_monthly_revenue": round(float(df["monthly_charges"].sum()), 2),
            "total_revenue_at_risk_12m": round(float(scores["revenue_at_risk_12m"].sum()), 2),
            "high_risk_customers": int(high.sum()),
            "high_risk_revenue_at_risk_12m": round(float(scores.loc[high, "revenue_at_risk_12m"].sum()), 2),
        },
        "top_features": imp.head(10).to_dict(orient="records"),
        "total_seconds": round(time.time() - t0, 1),
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2))
    print(
        f"saved {MODEL_PATH.name}, metrics.json, feature_importance.csv, customer_scores.csv "
        f"in {summary['total_seconds']}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
