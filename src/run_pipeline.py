"""Light-mode EDA and churn modeling pipeline for try_4_music.

The script never overwrites the source CSVs. It produces EDA plots, split/model
metadata, comparison metrics, a provisional threshold, and an inference file
for the unlabeled provided test set.
"""

from __future__ import annotations

import json
import hashlib
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    fbeta_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.features import MusicFeatureEngineer

DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
EDA_DIR = ARTIFACT_DIR / "eda"
MODEL_DIR = ARTIFACT_DIR / "model"
RANDOM_STATE = 42
FN_COST = 3.0
FP_COST = 1.0


def json_safe(value):
    if isinstance(value, (np.integer, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float64)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_safe), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_dirs() -> None:
    for path in [ARTIFACT_DIR, EDA_DIR, MODEL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, test


def plot_target(train: pd.DataFrame) -> None:
    counts = train["churned"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(["Active (0)", "Churned (1)"], counts.values, color=["#315A7D", "#D95D39"])
    ax.set_title("Target distribution")
    ax.set_ylabel("Customers")
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{count:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "target_distribution.png", dpi=150)
    plt.close(fig)


def plot_numeric_target(train: pd.DataFrame, column: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, color in [(0, "#315A7D"), (1, "#D95D39")]:
        values = train.loc[train["churned"] == label, column].dropna()
        ax.hist(values, bins=30, alpha=0.58, density=True, label=f"churned={label}", color=color)
    ax.set_title(f"{column} by churn label")
    ax.set_xlabel(column)
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(EDA_DIR / f"numeric_{column}.png", dpi=130)
    plt.close(fig)


def plot_categorical_target(train: pd.DataFrame, column: str) -> None:
    rates = train.groupby(column, dropna=False)["churned"].agg(["mean", "size"]).sort_values("mean", ascending=False)
    labels = rates.index.astype(str).tolist()
    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.35 * len(labels))))
    bars = ax.barh(labels[::-1], rates["mean"].values[::-1], color="#D95D39")
    ax.set_title(f"Churn rate by {column}")
    ax.set_xlabel("Churn rate")
    ax.set_xlim(0, max(0.4, float(rates["mean"].max()) * 1.2))
    for bar, rate, size in zip(bars, rates["mean"].values[::-1], rates["size"].values[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f"{rate:.1%} (n={size:,})", va="center", fontsize=8)
    fig.tight_layout()
    safe = column.replace("/", "_")
    fig.savefig(EDA_DIR / f"categorical_{safe}.png", dpi=130)
    plt.close(fig)


def plot_signup_cohort(train: pd.DataFrame) -> None:
    dates = pd.to_datetime(train["signup_date"], errors="coerce")
    cohort = train.assign(signup_year=dates.dt.year).groupby("signup_year")["churned"].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(cohort.index, cohort["mean"], marker="o", color="#D95D39")
    ax.set_title("Churn rate by signup year (cohort view)")
    ax.set_xlabel("Signup year")
    ax.set_ylabel("Churn rate")
    ax.set_xticks(cohort.index)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "signup_date_target_rate.png", dpi=150)
    plt.close(fig)


def plot_numeric_correlations(train: pd.DataFrame, numeric_columns: list[str]) -> None:
    corr = train[numeric_columns + ["churned"]].corr(numeric_only=True)[["churned"]].drop(index="churned")
    corr = corr.sort_values("churned")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = ["#315A7D" if v < 0 else "#D95D39" for v in corr["churned"]]
    ax.barh(corr.index, corr["churned"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title("Numeric feature correlation with churned")
    ax.set_xlabel("Pearson correlation")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "numeric_target_correlations.png", dpi=150)
    plt.close(fig)


def run_eda(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    numeric_columns = train.select_dtypes(include=np.number).columns.tolist()
    numeric_columns = [c for c in numeric_columns if c not in ["customer_id", "churned"]]
    categorical_columns = [c for c in train.select_dtypes(include="object").columns if c != "signup_date"]

    plot_target(train)
    for column in numeric_columns:
        plot_numeric_target(train, column)
    for column in categorical_columns:
        plot_categorical_target(train, column)
    plot_signup_cohort(train)
    plot_numeric_correlations(train, numeric_columns)

    numeric_corr = train[numeric_columns + ["churned"]].corr(numeric_only=True)["churned"].drop("churned").sort_values()
    categorical_rates = {}
    for column in categorical_columns:
        grouped = train.groupby(column, dropna=False)["churned"].agg(["mean", "size"]).sort_values("mean", ascending=False)
        categorical_rates[column] = {
            str(key): {"churn_rate": float(row["mean"]), "n": int(row["size"])}
            for key, row in grouped.iterrows()
        }
    summary = {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "missing_values_train": int(train.isna().sum().sum()),
        "missing_values_test": int(test.isna().sum().sum()),
        "duplicate_rows_train": int(train.duplicated().sum()),
        "duplicate_rows_test": int(test.duplicated().sum()),
        "target_counts": {str(k): int(v) for k, v in train["churned"].value_counts().sort_index().items()},
        "target_rate": float(train["churned"].mean()),
        "numeric_correlations": {str(k): float(v) for k, v in numeric_corr.items()},
        "categorical_churn_rates": categorical_rates,
        "chart_count": len(list(EDA_DIR.glob("*.png"))),
    }
    write_json(ARTIFACT_DIR / "eda_summary.json", summary)

    lines = [
        "# EDA output",
        "",
        f"- Train: `{train.shape[0]:,}` rows × `{train.shape[1]}` columns",
        f"- Test: `{test.shape[0]:,}` rows × `{test.shape[1]}` columns (no target)",
        f"- Churn rate: `{train['churned'].mean():.2%}`",
        f"- Generated PNG charts: `{summary['chart_count']}`",
        "",
        "## Interpretation guardrail",
        "",
        "Charts show association with the dataset label, not causal effects. `signup_date` is shown as a cohort view because no prediction snapshot or churn event date is provided.",
        "",
        "## Chart index",
        "",
    ]
    for chart in sorted(EDA_DIR.glob("*.png")):
        lines.append(f"- `{chart.name}`")
    (EDA_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def transformed_columns(X: pd.DataFrame, engineer: bool) -> tuple[list[str], list[str]]:
    transformed = MusicFeatureEngineer(engineer=engineer).fit_transform(X.head(5))
    numeric = transformed.select_dtypes(include=np.number).columns.tolist()
    categorical = transformed.select_dtypes(exclude=np.number).columns.tolist()
    return numeric, categorical


def make_preprocessor(X: pd.DataFrame, engineer: bool, scale_numeric: bool = False) -> ColumnTransformer:
    numeric, categorical = transformed_columns(X, engineer)
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def model_pipeline(X: pd.DataFrame, model, engineer: bool = True, scale_numeric: bool = False) -> Pipeline:
    return Pipeline(
        [
            ("features", MusicFeatureEngineer(engineer=engineer)),
            ("preprocess", make_preprocessor(X, engineer=engineer, scale_numeric=scale_numeric)),
            ("model", model),
        ]
    )


def metrics_at_threshold(y_true: pd.Series, probability: np.ndarray, threshold: float) -> dict:
    prediction = (probability >= threshold).astype(int)
    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
    }


def probability_metrics(y_true: pd.Series, probability: np.ndarray) -> dict:
    return {
        "pr_auc": float(average_precision_score(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
    }


def choose_operating_threshold(y_true: pd.Series, probability: np.ndarray) -> tuple[float, pd.DataFrame]:
    thresholds = np.linspace(0.05, 0.95, 91)
    rows = []
    for threshold in thresholds:
        prediction = (probability >= threshold).astype(int)
        matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        rows.append(
            {
                "threshold": float(threshold),
                "recall": float(recall_score(y_true, prediction, zero_division=0)),
                "precision": float(precision_score(y_true, prediction, zero_division=0)),
                "f1": float(f1_score(y_true, prediction, zero_division=0)),
                "f2": float(fbeta_score(y_true, prediction, beta=2, zero_division=0)),
                "positive_rate": float(prediction.mean()),
                "expected_cost_per_customer": float((FN_COST * fn + FP_COST * fp) / len(y_true)),
                "false_negative_cost": FN_COST,
                "false_positive_cost": FP_COST,
            }
        )
    table = pd.DataFrame(rows)
    best = table.sort_values(
        ["expected_cost_per_customer", "recall", "precision"],
        ascending=[True, False, False],
    ).iloc[0]
    return float(best["threshold"]), table


def fit_and_compare(train: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    X = train.drop(columns=["churned"])
    y = train["churned"].astype(int)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )

    candidates = {
        "dummy_prior": (DummyClassifier(strategy="prior"), True, False),
        "logistic_raw": (LogisticRegression(max_iter=1200, class_weight="balanced", random_state=RANDOM_STATE), False, True),
        "logistic_engineered": (LogisticRegression(max_iter=1200, class_weight="balanced", random_state=RANDOM_STATE), True, True),
        "decision_tree": (DecisionTreeClassifier(max_depth=8, min_samples_leaf=40, class_weight="balanced", random_state=RANDOM_STATE), True, False),
        "random_forest": (RandomForestClassifier(n_estimators=160, max_depth=14, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE), True, False),
        "extra_trees": (ExtraTreesClassifier(n_estimators=160, max_depth=16, min_samples_leaf=4, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE), True, False),
        "gradient_boosting": (GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE), True, False),
    }

    results = []
    fitted = {}
    validation_probabilities = {}
    for name, (model, engineer, scale_numeric) in candidates.items():
        pipe = model_pipeline(X_train, model, engineer=engineer, scale_numeric=scale_numeric)
        pipe.fit(X_train, y_train)
        val_prob = pipe.predict_proba(X_val)[:, 1]
        test_prob = pipe.predict_proba(X_test)[:, 1]
        threshold, threshold_table = choose_operating_threshold(y_val, val_prob)
        val_default = metrics_at_threshold(y_val, val_prob, 0.5)
        val_operating = metrics_at_threshold(y_val, val_prob, threshold)
        test_default = metrics_at_threshold(y_test, test_prob, 0.5)
        test_operating = metrics_at_threshold(y_test, test_prob, threshold)
        row = {
            "model": name,
            "engineered_features": engineer,
            "validation_pr_auc": probability_metrics(y_val, val_prob)["pr_auc"],
            "validation_roc_auc": probability_metrics(y_val, val_prob)["roc_auc"],
            "validation_brier": probability_metrics(y_val, val_prob)["brier"],
            "validation_recall_at_0.5": val_default["recall"],
            "validation_precision_at_0.5": val_default["precision"],
            "validation_f1_at_0.5": val_default["f1"],
            "validation_operating_threshold": threshold,
            "validation_operating_recall": val_operating["recall"],
            "validation_operating_precision": val_operating["precision"],
            "validation_operating_f2": float(threshold_table.loc[threshold_table["threshold"] == threshold, "f2"].iloc[0]),
            "validation_expected_cost_per_customer": float(threshold_table.loc[threshold_table["threshold"] == threshold, "expected_cost_per_customer"].iloc[0]),
            "test_pr_auc": probability_metrics(y_test, test_prob)["pr_auc"],
            "test_roc_auc": probability_metrics(y_test, test_prob)["roc_auc"],
            "test_recall_at_0.5": test_default["recall"],
            "test_precision_at_0.5": test_default["precision"],
            "test_f1_at_0.5": test_default["f1"],
            "test_operating_recall": test_operating["recall"],
            "test_operating_precision": test_operating["precision"],
            "test_operating_f1": test_operating["f1"],
            "test_expected_cost_per_customer": float((FN_COST * test_operating["fn"] + FP_COST * test_operating["fp"]) / len(y_test)),
            "test_tn": test_operating["tn"],
            "test_fp": test_operating["fp"],
            "test_fn": test_operating["fn"],
            "test_tp": test_operating["tp"],
        }
        results.append(row)
        fitted[name] = pipe
        validation_probabilities[name] = {"probability": val_prob, "y": y_val, "threshold_table": threshold_table}

    comparison = pd.DataFrame(results).sort_values(
        ["validation_expected_cost_per_customer", "validation_pr_auc", "validation_operating_recall"],
        ascending=[True, False, False],
    )
    comparison.to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")

    recommended_name = str(comparison.iloc[0]["model"])
    recommended_pipe = fitted[recommended_name]
    recommended_info = {
        "model": recommended_name,
        "selection_rule": f"lowest Validation expected cost with FN:FP={FN_COST:g}:1, then PR-AUC, then Recall; provisional and awaiting user approval",
        "false_negative_cost": FN_COST,
        "false_positive_cost": FP_COST,
        "validation_threshold": float(comparison.iloc[0]["validation_operating_threshold"]),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "split": {"train": 0.6, "validation": 0.2, "test": 0.2},
    }
    joblib.dump(recommended_pipe, MODEL_DIR / "music_churn_pipeline.joblib")
    write_json(MODEL_DIR / "metadata.json", recommended_info)

    best_val = validation_probabilities[recommended_name]
    best_val["threshold_table"].to_csv(ARTIFACT_DIR / "threshold_sweep_validation.csv", index=False)
    return comparison, recommended_info, {"pipeline": recommended_pipe, "X_test": X_test, "y_test": y_test, "fitted": fitted}


def write_feature_importance(pipeline: Pipeline) -> None:
    """Persist model-native feature importance for the technical recommendation."""
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]
    values = getattr(model, "feature_importances_", None)
    if values is None:
        return

    names = preprocessor.get_feature_names_out()
    importance = (
        pd.DataFrame({"feature": names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance["rank"] = np.arange(1, len(importance) + 1)
    importance.to_csv(ARTIFACT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")

    top = importance.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance"], color="#e4573d")
    ax.set_title("Top model feature importance (association, not causation)")
    ax.set_xlabel("Gradient boosting importance")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_model_report(comparison: pd.DataFrame, recommended_info: dict, train: pd.DataFrame, test: pd.DataFrame) -> None:
    best = comparison.iloc[0]
    lines = [
        "# Model Run Report",
        "",
        f"- Run time (UTC): `{datetime.now(timezone.utc).isoformat()}`",
        f"- Python: `{platform.python_version()}`",
        f"- Recommended candidate (provisional): `{recommended_info['model']}`",
        f"- Threshold rule: `{recommended_info['selection_rule']}`",
        f"- Validation operating threshold: `{recommended_info['validation_threshold']:.2f}`",
        "",
        "## Data and split",
        "",
        f"- Local train: `{len(train):,}` rows with target; local test: `{len(test):,}` rows without target",
        f"- Internal split: `{recommended_info['train_rows']:,}` / `{recommended_info['validation_rows']:,}` / `{recommended_info['test_rows']:,}`",
        "- Split: stratified random split with random_state=42; switch to chronological split if event timestamps become available",
        "- `customer_id` excluded from model input; raw `signup_date` excluded and date-derived/ratio features are generated inside the Pipeline",
        "",
        "## Comparison",
        "",
        "| Model | Val PR-AUC | Val Recall@operating | Val Precision@operating | Val cost/customer | Test PR-AUC | Test Recall@operating | Test Precision@operating | Test cost/customer |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparison.iterrows():
        lines.append(
            f"| {row['model']} | {row['validation_pr_auc']:.4f} | {row['validation_operating_recall']:.4f} | {row['validation_operating_precision']:.4f} | {row['validation_expected_cost_per_customer']:.4f} | {row['test_pr_auc']:.4f} | {row['test_operating_recall']:.4f} | {row['test_operating_precision']:.4f} | {row['test_expected_cost_per_customer']:.4f} |"
        )
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- The recommended candidate is not final approval; it is a technical provisional recommendation.",
        "- Metrics are for the dataset-provided `churned` label. They do not validate a future 30-day churn horizon.",
        f"- The provisional threshold uses FN:FP={FN_COST:g}:1; a high Recall operating point can still increase false positives, so the threshold and campaign capacity require business approval.",
        "- Feature importance and EDA relationships are associations, not causal churn drivers; see `artifacts/feature_importance.csv`.",
    ]
    (ARTIFACT_DIR / "model_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_test_predictions(train: pd.DataFrame, test: pd.DataFrame, recommended_info: dict, pipeline: Pipeline) -> None:
    probability = pipeline.predict_proba(test)[:, 1]
    threshold = recommended_info["validation_threshold"]
    output = pd.DataFrame(
        {
            "customer_id": test["customer_id"].astype(int),
            "churn_probability": probability,
            "predicted_churn": (probability >= threshold).astype(int),
            "operating_threshold": threshold,
            "model": recommended_info["model"],
        }
    )
    output.to_csv(ARTIFACT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    ensure_dirs()
    train, test = load_data()
    if "churned" not in train or "churned" in test:
        raise ValueError("Expected churned in train only")
    eda_summary = run_eda(train, test)
    comparison, recommended_info, run_context = fit_and_compare(train)
    write_feature_importance(run_context["pipeline"])
    write_model_report(comparison, recommended_info, train, test)
    write_test_predictions(train, test, recommended_info, run_context["pipeline"])

    metadata = {
        **recommended_info,
        "source_files": {
            "train": {"path": "data/train.csv", "sha256": file_sha256(DATA_DIR / "train.csv")},
            "test": {"path": "data/test.csv", "sha256": file_sha256(DATA_DIR / "test.csv")},
        },
        "eda": eda_summary,
        "artifacts": [
            "artifacts/eda_summary.json",
            "artifacts/model_comparison.csv",
            "artifacts/threshold_sweep_validation.csv",
            "artifacts/feature_importance.csv",
            "artifacts/eda/feature_importance.png",
            "artifacts/test_predictions.csv",
            "artifacts/model/music_churn_pipeline.joblib",
            "artifacts/model/metadata.json",
        ],
    }
    write_json(ARTIFACT_DIR / "run_metadata.json", metadata)
    print(json.dumps({"recommended": recommended_info, "comparison": comparison.to_dict("records"), "eda": eda_summary}, ensure_ascii=False, indent=2, default=json_safe))


if __name__ == "__main__":
    main()
