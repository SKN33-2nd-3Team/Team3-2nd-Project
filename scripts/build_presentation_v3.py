"""Build presentation-ready figures from saved full-fair run evidence.

This script performs no model training.  It reuses committed CSV/JSON evidence,
saved OOF probabilities, and the already-fitted candidate pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUN_ID = "20260721_full_fair_v1"
RUN = ROOT / "experiments" / "submission_full_comparison" / RUN_ID
ARTIFACTS = ROOT / "artifacts"
SOURCE_DIR = ARTIFACTS / "presentation_v3"
FIGURE_DIR = ROOT / "figures" / "presentation_v3"
CANDIDATE = ROOT / "models" / "candidates" / RUN_ID / "candidate_pipeline.joblib"

COLORS = {
    "primary": "#2563EB",
    "secondary": "#0F766E",
    "accent": "#D97706",
    "muted": "#94A3B8",
    "danger": "#DC2626",
    "grid": "#D7DEE8",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#64748B",
            "axes.labelcolor": "#111827",
            "axes.titleweight": "bold",
            "font.size": 10,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.65,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
        }
    )


def save_source(frame: pd.DataFrame, stem: str) -> pd.DataFrame:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(SOURCE_DIR / f"{stem}.csv", index=False, encoding="utf-8-sig")
    return frame


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def annotate_bars(ax, values, fmt="{:.4f}", offset=0.00015) -> None:
    for index, value in enumerate(values):
        ax.text(value + offset, index, fmt.format(value), va="center", fontsize=8)


def metric_at_threshold(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction, zero_division=0),
        "f1": f1_score(y, prediction, zero_division=0),
        "target_customers": int(prediction.sum()),
        "target_rate": float(prediction.mean()),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
        "tp": int(tp),
    }


def load_candidate():
    try:
        return joblib.load(CANDIDATE)
    except (AttributeError, ModuleNotFoundError):
        # Compatibility for the original checkpoint before the transformer was
        # moved to an importable module.  Finalization replaces this artifact.
        from scripts.run_full_fair_comparison import Features

        setattr(sys.modules["__main__"], "Features", Features)
        return joblib.load(CANDIDATE)


def build_progression() -> None:
    feature = pd.read_csv(ARTIFACTS / "preprocessing_experiment_results.csv")
    baseline = pd.read_csv(ARTIFACTS / "model_comparison_fair.csv").set_index("model")
    search = pd.read_csv(ARTIFACTS / "random_search_summary.csv").set_index("model")
    fine = pd.read_csv(ARTIFACTS / "top3_fine_tuning_summary.csv").set_index("model")
    rows = [
        {"stage": "Raw Logistic", **feature.loc[feature.variant.eq("raw")].iloc[0][["pr_auc", "recall", "precision", "f1"]].to_dict()},
        {"stage": "Log-feature Logistic", **feature.loc[feature.variant.eq("log_numeric")].iloc[0][["pr_auc", "recall", "precision", "f1"]].to_dict()},
        {"stage": "CatBoost baseline", **baseline.loc["catboost", ["pr_auc", "recall", "precision", "f1"]].to_dict()},
        {"stage": "CatBoost search", **search.loc["catboost", ["pr_auc", "recall", "precision", "f1"]].to_dict()},
        {"stage": "CatBoost fine-tuned", **fine.loc["catboost", ["pr_auc", "recall", "precision", "f1"]].to_dict()},
    ]
    frame = save_source(pd.DataFrame(rows), "01_performance_progression")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(frame))
    ax.plot(x, frame.pr_auc, marker="o", linewidth=2.5, color=COLORS["primary"], label="PR-AUC")
    ax.plot(x, frame.f1, marker="s", linewidth=2, color=COLORS["secondary"], label="F1 @ 0.5")
    for i, value in enumerate(frame.pr_auc):
        ax.annotate(f"{value:.4f}", (i, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x, frame.stage, rotation=18, ha="right")
    ax.set_ylim(0.78, 0.97)
    ax.set_ylabel("Score")
    ax.set_title("From raw baseline to the selected CatBoost candidate")
    ax.legend(ncol=2, loc="lower right")
    save_figure(fig, "01_performance_progression")


def build_preprocessing() -> None:
    frame = pd.read_csv(ARTIFACTS / "preprocessing_experiment_results.csv")
    frame["delta_pr_auc_pp"] = frame["delta_pr_auc_vs_raw"] * 100
    frame["status"] = np.where(frame.variant.eq("log_numeric"), "ADOPTED", "REJECTED")
    frame = save_source(frame, "02_preprocessing_experiments")
    plot = frame.sort_values("delta_pr_auc_pp")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = [COLORS["primary"] if x == "ADOPTED" else COLORS["muted"] for x in plot.status]
    ax.barh(plot.variant, plot.delta_pr_auc_pp, color=colors)
    for i, value in enumerate(plot.delta_pr_auc_pp):
        ax.text(value + (0.01 if value >= 0 else -0.01), i, f"{value:+.3f} pp", va="center", ha="left" if value >= 0 else "right", fontsize=8)
    ax.axvline(0, color="#334155", linewidth=0.9)
    ax.set_xlabel("PR-AUC change vs raw logistic (percentage points)")
    ax.set_title("Only log-numeric features produced material measured lift")
    save_figure(fig, "02_preprocessing_experiments")


def build_adopt_reject() -> None:
    frame = pd.read_csv(ARTIFACTS / "insight_preprocessing_register.csv")
    frame = save_source(frame, "03_adopted_rejected_experiments")
    plot = frame.sort_values("logistic_pr_auc")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = [COLORS["primary"] if x == "adopt" else COLORS["muted"] for x in plot.decision]
    ax.barh(plot.variant, plot.logistic_pr_auc, color=colors)
    annotate_bars(ax, plot.logistic_pr_auc.to_numpy())
    ax.set_xlim(plot.logistic_pr_auc.min() - 0.001, plot.logistic_pr_auc.max() + 0.002)
    ax.set_xlabel("5-fold OOF PR-AUC")
    ax.set_title("Adopted and rejected feature experiments")
    save_figure(fig, "03_adopted_rejected_experiments")


def build_eight_models() -> None:
    frame = pd.read_csv(ARTIFACTS / "model_comparison_fair.csv").sort_values("pr_auc")
    frame = save_source(frame, "04_eight_model_fair_comparison")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = [COLORS["primary"] if x == "catboost" else COLORS["muted"] for x in frame.model]
    ax.barh(frame.model, frame.pr_auc, color=colors)
    annotate_bars(ax, frame.pr_auc.to_numpy())
    ax.set_xlim(max(0.48, frame.pr_auc.min() - 0.02), 0.97)
    ax.set_xlabel("5-fold OOF PR-AUC")
    ax.set_title("Eight models compared on identical features and folds")
    save_figure(fig, "04_eight_model_fair_comparison")


def build_search_before_after() -> None:
    before = pd.read_csv(ARTIFACTS / "model_comparison_fair.csv").set_index("model")
    after = pd.read_csv(ARTIFACTS / "random_search_summary.csv").set_index("model")
    models = [m for m in after.index if m in before.index]
    frame = pd.DataFrame({"model": models, "baseline_pr_auc": before.loc[models, "pr_auc"].values, "searched_pr_auc": after.loc[models, "pr_auc"].values})
    frame["delta_pr_auc"] = frame.searched_pr_auc - frame.baseline_pr_auc
    frame = save_source(frame.sort_values("searched_pr_auc", ascending=False), "05_seven_model_search_before_after")
    fig, ax = plt.subplots(figsize=(9, 5.3))
    y = np.arange(len(frame))
    for i, row in frame.reset_index(drop=True).iterrows():
        ax.plot([row.baseline_pr_auc, row.searched_pr_auc], [i, i], color=COLORS["grid"], linewidth=2)
    ax.scatter(frame.baseline_pr_auc, y, color=COLORS["muted"], label="Baseline", s=45)
    ax.scatter(frame.searched_pr_auc, y, color=COLORS["primary"], label="After search", s=55)
    ax.set_yticks(y, frame.model)
    ax.set_xlabel("5-fold OOF PR-AUC")
    ax.set_title("Randomized search impact across seven trainable models")
    ax.legend(ncol=2, loc="lower right")
    save_figure(fig, "05_seven_model_search_before_after")


def build_top3_fine_tuning() -> None:
    baseline = pd.read_csv(ARTIFACTS / "model_comparison_fair.csv").set_index("model")
    search = pd.read_csv(ARTIFACTS / "random_search_summary.csv").set_index("model")
    fine = pd.read_csv(ARTIFACTS / "top3_fine_tuning_summary.csv").set_index("model")
    models = ["catboost", "xgboost", "lightgbm"]
    rows = []
    for model in models:
        rows.extend(
            [
                {"model": model, "stage": "Baseline", "pr_auc": baseline.loc[model, "pr_auc"]},
                {"model": model, "stage": "Random search", "pr_auc": search.loc[model, "pr_auc"]},
                {"model": model, "stage": "Fine tuning", "pr_auc": fine.loc[model, "pr_auc"]},
            ]
        )
    frame = save_source(pd.DataFrame(rows), "06_top3_fine_tuning_before_after")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(models)); width = 0.23
    for j, (stage, color) in enumerate(zip(["Baseline", "Random search", "Fine tuning"], [COLORS["muted"], COLORS["secondary"], COLORS["primary"]])):
        vals = [frame.loc[(frame.model == model) & (frame.stage == stage), "pr_auc"].iloc[0] for model in models]
        ax.bar(x + (j - 1) * width, vals, width, label=stage, color=color)
    ax.set_xticks(x, [m.title() for m in models])
    ax.set_ylim(0.9465, 0.9483)
    ax.set_ylabel("5-fold OOF PR-AUC")
    ax.set_title("Top-three tuning results are close")
    ax.legend(ncol=3, loc="lower center")
    save_figure(fig, "06_top3_fine_tuning_before_after")


def build_seed_stability() -> None:
    frame = pd.read_csv(RUN / "seed_stability_all.csv")
    frame = save_source(frame, "07_seed_stability")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    markers = ["o", "s", "^"]
    colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]]
    for model, marker, color in zip(["catboost", "xgboost", "lightgbm"], markers, colors):
        part = frame.loc[frame.model.eq(model)].sort_values("seed")
        ax.plot(part.seed.astype(str), part.pr_auc, marker=marker, linewidth=1.8, label=model.title(), color=color)
    ax.set_ylabel("5-fold OOF PR-AUC")
    ax.set_xlabel("Random seed")
    ax.set_title("Top-three rankings are stable across five seeds")
    ax.legend(ncol=3, loc="lower right")
    save_figure(fig, "07_seed_stability")


def build_bootstrap() -> None:
    frame = pd.read_csv(ARTIFACTS / "bootstrap_confidence_intervals.csv")
    frame = frame.loc[frame.metric.eq("pr_auc")].sort_values("estimate")
    frame = save_source(frame, "08_bootstrap_confidence_intervals")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xerr = np.vstack([frame.estimate - frame.ci_low, frame.ci_high - frame.estimate])
    ax.errorbar(frame.estimate, frame.model, xerr=xerr, fmt="o", capsize=5, color=COLORS["primary"], ecolor=COLORS["muted"])
    ax.set_xlabel("PR-AUC, 95% bootstrap interval (1,000 resamples)")
    ax.set_title("Bootstrap intervals overlap: score differences are small")
    save_figure(fig, "08_bootstrap_confidence_intervals")


def build_calibration(y: np.ndarray, probabilities: dict[str, np.ndarray]) -> None:
    rows = []
    for model, probability in probabilities.items():
        observed, predicted = calibration_curve(y, probability, n_bins=10, strategy="quantile")
        for index, (p_hat, p_obs) in enumerate(zip(predicted, observed), start=1):
            rows.append({"model": model, "bin": index, "mean_predicted_probability": p_hat, "observed_churn_rate": p_obs})
    frame = save_source(pd.DataFrame(rows), "09_calibration_curve")
    fig, ax = plt.subplots(figsize=(6.3, 5.3))
    ax.plot([0, 1], [0, 1], color="#475569", linestyle="--", label="Ideal")
    for model, color, marker in zip(probabilities, [COLORS["primary"], COLORS["secondary"], COLORS["accent"]], ["o", "s", "^"]):
        part = frame.loc[frame.model.eq(model)]
        ax.plot(part.mean_predicted_probability, part.observed_churn_rate, marker=marker, color=color, label=model.title())
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed churn rate")
    ax.set_title("OOF probability calibration")
    ax.legend(loc="upper left")
    save_figure(fig, "09_calibration_curve")


def build_thresholds(y: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame([metric_at_threshold(y, probability, float(t)) for t in np.round(np.arange(0.05, 0.951, 0.01), 2)])
    frame = save_source(frame, "10_11_threshold_sweep")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(frame.threshold, frame.precision, color=COLORS["primary"], label="Precision")
    ax.plot(frame.threshold, frame.recall, color=COLORS["danger"], label="Recall")
    ax.plot(frame.threshold, frame.f1, color=COLORS["secondary"], label="F1")
    best = frame.loc[frame.f1.idxmax()]
    ax.axvline(best.threshold, color="#475569", linestyle="--", linewidth=1)
    ax.annotate(f"Best F1 threshold {best.threshold:.2f}", (best.threshold, best.f1), xytext=(10, -28), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold changes precision-recall trade-off")
    ax.legend(ncol=3, loc="lower center")
    save_figure(fig, "10_threshold_precision_recall_f1")

    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax1.plot(frame.threshold, frame.target_customers, color=COLORS["primary"], label="Target customers")
    ax1.set_xlabel("Decision threshold")
    ax1.set_ylabel("Target customers", color=COLORS["primary"])
    ax2 = ax1.twinx(); ax2.grid(False)
    ax2.plot(frame.threshold, frame.fn, color=COLORS["danger"], label="FN")
    ax2.plot(frame.threshold, frame.fp, color=COLORS["accent"], label="FP")
    ax2.set_ylabel("Error count")
    handles = ax1.get_lines() + ax2.get_lines()
    ax1.legend(handles, [line.get_label() for line in handles], ncol=3, loc="upper center")
    ax1.set_title("Threshold sets campaign volume and error burden")
    save_figure(fig, "11_threshold_volume_fn_fp")
    return frame


def build_equal_contact() -> None:
    frame = save_source(pd.read_csv(ARTIFACTS / "equal_contact_comparison.csv"), "12_equal_contact_comparison")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for model, color, marker in zip(["catboost", "xgboost", "lightgbm"], [COLORS["primary"], COLORS["secondary"], COLORS["accent"]], ["o", "s", "^"]):
        part = frame.loc[frame.model.eq(model)]
        ax.plot(part.top_percent, part.capture_rate, marker=marker, color=color, label=model.title())
    ax.set_xlabel("Customers contacted (top %)")
    ax.set_ylabel("Observed churners captured")
    ax.set_title("Equal-contact comparison")
    ax.legend(ncol=3, loc="lower right")
    save_figure(fig, "12_equal_contact_comparison")


def build_equal_recall() -> None:
    frame = save_source(pd.read_csv(ARTIFACTS / "equal_recall_comparison.csv"), "13_equal_recall_comparison")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for model, color, marker in zip(["catboost", "xgboost", "lightgbm"], [COLORS["primary"], COLORS["secondary"], COLORS["accent"]], ["o", "s", "^"]):
        part = frame.loc[frame.model.eq(model)]
        ax.plot(part.target_recall, part.target_customers, marker=marker, color=color, label=model.title())
    ax.set_xlabel("Required recall")
    ax.set_ylabel("Customers that must be contacted")
    ax.set_title("Contact volume required for the same recall")
    ax.legend(ncol=3, loc="upper left")
    save_figure(fig, "13_equal_recall_comparison")


def build_topk() -> None:
    frame = save_source(pd.read_csv(ARTIFACTS / "topk_lift_v2.csv"), "14_topk_capture_lift")
    part = frame.loc[frame.model.eq("catboost")]
    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax1.plot(part.top_percent, part.capture_rate, marker="o", color=COLORS["primary"], label="Capture rate")
    ax1.set_xlabel("Customers contacted (top %)")
    ax1.set_ylabel("Observed churners captured", color=COLORS["primary"])
    ax2 = ax1.twinx(); ax2.grid(False)
    ax2.plot(part.top_percent, part.lift, marker="s", color=COLORS["accent"], label="Lift")
    ax2.set_ylabel("Lift vs random targeting")
    handles = ax1.get_lines() + ax2.get_lines()
    ax1.legend(handles, [line.get_label() for line in handles], ncol=2, loc="center right")
    ax1.set_title("CatBoost Top-K capture and lift")
    save_figure(fig, "14_topk_capture_lift")


def build_risk_decile() -> None:
    frame = save_source(pd.read_csv(ARTIFACTS / "risk_decile_v2.csv"), "15_risk_decile")
    part = frame.loc[frame.model.eq("catboost")].sort_values("decile")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(part)); width = 0.36
    ax.bar(x - width / 2, part.actual_churn_rate, width, color=COLORS["primary"], label="Observed")
    ax.bar(x + width / 2, part.mean_predicted_probability, width, color=COLORS["muted"], label="Predicted")
    ax.set_xticks(x, part.decile.astype(int))
    ax.set_xlabel("Risk decile (1 = lowest, 10 = highest)")
    ax.set_ylabel("Rate / probability")
    ax.set_title("Risk deciles separate low- and high-risk customers")
    ax.legend(ncol=2, loc="upper left")
    save_figure(fig, "15_risk_decile")


def build_confusion(y: np.ndarray, probability: np.ndarray) -> None:
    values = metric_at_threshold(y, probability, 0.5)
    frame = save_source(pd.DataFrame([
        {"actual": "Retained", "predicted": "Retained", "count": values["tn"]},
        {"actual": "Retained", "predicted": "Churn", "count": values["fp"]},
        {"actual": "Churn", "predicted": "Retained", "count": values["fn"]},
        {"actual": "Churn", "predicted": "Churn", "count": values["tp"]},
    ]), "16_final_confusion_matrix")
    matrix = np.array([[values["tn"], values["fp"]], [values["fn"], values["tp"]]])
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center", color="white" if matrix[i, j] > matrix.max() * 0.55 else "#111827", fontsize=13)
    ax.set_xticks([0, 1], ["Retained", "Churn"])
    ax.set_yticks([0, 1], ["Retained", "Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("CatBoost OOF confusion matrix at threshold 0.50")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, "16_final_confusion_matrix")


def build_feature_importance(pipeline) -> None:
    names = pipeline.named_steps["pre"].get_feature_names_out()
    values = pipeline.named_steps["model"].feature_importances_
    frame = pd.DataFrame({"feature": names, "importance": values}).sort_values("importance", ascending=False).reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    frame = save_source(frame, "17_final_feature_importance")
    top = frame.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.barh(top.feature, top.importance, color=COLORS["primary"])
    ax.set_xlabel("CatBoost feature importance")
    ax.set_title("Most influential features in the selected candidate")
    save_figure(fig, "17_final_feature_importance")


def build_segments() -> None:
    frame = save_source(pd.read_csv(ARTIFACTS / "segment_error_analysis.csv"), "18_segment_error_analysis")
    plot = frame.sort_values("recall")
    fig, ax = plt.subplots(figsize=(9, 5.3))
    y = np.arange(len(plot)); width = 0.34
    ax.barh(y - width / 2, plot.recall, width, color=COLORS["primary"], label="Recall")
    ax.barh(y + width / 2, plot.precision, width, color=COLORS["secondary"], label="Precision")
    ax.set_yticks(y, plot.segment)
    ax.set_xlim(0.68, 0.97)
    ax.set_xlabel("OOF score at threshold 0.50")
    ax.set_title("Error performance differs by customer segment")
    ax.legend(ncol=2, loc="lower right")
    save_figure(fig, "18_segment_error_analysis")


def build_correlations() -> None:
    summary = json.loads((ARTIFACTS / "eda_summary.json").read_text(encoding="utf-8"))
    frame = pd.DataFrame([{"feature": key, "target_correlation": value} for key, value in summary["numeric_correlations"].items()])
    frame["absolute_correlation"] = frame.target_correlation.abs()
    frame = save_source(frame.sort_values("target_correlation"), "19_numeric_target_correlations")
    fig, ax = plt.subplots(figsize=(9, 6.2))
    colors = [COLORS["danger"] if value > 0 else COLORS["primary"] for value in frame.target_correlation]
    ax.barh(frame.feature, frame.target_correlation, color=colors)
    ax.axvline(0, color="#334155", linewidth=0.9)
    ax.set_xlabel("Pearson correlation with observed churn label")
    ax.set_title("Numeric correlations are associations, not causal effects")
    save_figure(fig, "19_numeric_target_correlations")

    rows = []
    for feature in ["subscription_type", "customer_service_inquiries"]:
        for level, values in summary["categorical_churn_rates"][feature].items():
            rows.append({"feature": feature, "level": level, **values})
    categorical = save_source(pd.DataFrame(rows), "20_categorical_churn_associations")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), sharex=True)
    for ax, feature in zip(axes, ["subscription_type", "customer_service_inquiries"]):
        part = categorical.loc[categorical.feature.eq(feature)].sort_values("churn_rate")
        ax.barh(part.level, part.churn_rate, color=COLORS["primary"])
        ax.set_title(feature.replace("_", " ").title())
        ax.set_xlabel("Observed churn rate")
        ax.set_xlim(0.2, 0.85)
    fig.suptitle("Categorical churn-rate associations", fontweight="bold")
    save_figure(fig, "20_categorical_churn_associations")


def main() -> None:
    setup_style()
    train = pd.read_csv(ROOT / "data" / "train.csv")
    y = train["churned"].to_numpy(dtype=int)
    probabilities = {name: np.load(RUN / f"oof_fine_{name}.npy") for name in ["catboost", "xgboost", "lightgbm"]}
    pipeline = load_candidate()

    build_progression()
    build_preprocessing()
    build_adopt_reject()
    build_eight_models()
    build_search_before_after()
    build_top3_fine_tuning()
    build_seed_stability()
    build_bootstrap()
    build_calibration(y, probabilities)
    build_thresholds(y, probabilities["catboost"])
    build_equal_contact()
    build_equal_recall()
    build_topk()
    build_risk_decile()
    build_confusion(y, probabilities["catboost"])
    build_feature_importance(pipeline)
    build_segments()
    build_correlations()

    inventory = []
    for image_path in sorted(FIGURE_DIR.glob("*.png")):
        source_name = image_path.stem
        if source_name.startswith("10_threshold") or source_name.startswith("11_threshold"):
            source_name = "10_11_threshold_sweep"
        inventory.append(
            {
                "figure": str(image_path.relative_to(ROOT)).replace("\\", "/"),
                "source_csv": str((SOURCE_DIR / f"{source_name}.csv").relative_to(ROOT)).replace("\\", "/"),
                "bytes": image_path.stat().st_size,
            }
        )
    pd.DataFrame(inventory).to_csv(FIGURE_DIR / "figure_inventory.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"figures": len(inventory), "source_csvs": len(list(SOURCE_DIR.glob("*.csv")))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
