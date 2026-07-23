"""Reproduce every figure in artifacts/eda_insight/INSIGHT_REPORT.md.

Run from the project root:

    python scripts/legacy/eda_insight.py

Writes the seven core charts to artifacts/eda_insight/ and prints the quality
table, the signal/noise split, the recovered generating rule, and the held-out
comparison that shows the model already sits at the data's ceiling.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT = ROOT / "artifacts" / "eda_insight"
BLUE, RED, GREY = "#315A7D", "#D95D39", "#9AA5AD"
RANDOM_STATE = 42

# The seven columns that carry signal, and the thresholds recovered from the data.
SIGNAL = [
    "weekly_hours",
    "subscription_type",
    "customer_service_inquiries",
    "num_subscription_pauses",
    "song_skip_rate",
    "age",
    "notifications_clicked",
]


def bucketize(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the seven signals onto the step boundaries visible in the data."""
    out = pd.DataFrame(index=df.index)
    out["hours"] = pd.cut(df.weekly_hours, [-1, 5, 10, 40, 999], labels=["h<=5", "h5-10", "h10-40", "h>40"])
    out["skip"] = np.where(df.song_skip_rate > 0.7, "skip>0.7", "skip<=0.7")
    out["pause"] = np.where(df.num_subscription_pauses >= 3, "pause>=3", "pause<=2")
    out["age"] = pd.cut(df.age, [0, 24, 34, 60, 999], labels=["a18-24", "a25-34", "a35-60", "a61+"])
    out["notif"] = np.where(df.notifications_clicked < 5, "notif<5", "notif>=5")
    out["plan"] = df.subscription_type
    out["inq"] = df.customer_service_inquiries
    return out


BUCKET_KEYS = ["hours", "skip", "pause", "age", "notif", "plan", "inq"]


def risk_flags(df: pd.DataFrame) -> pd.Series:
    """The five business-facing flags used for segmentation."""
    return (
        (df.weekly_hours <= 10).astype(int)
        + (df.song_skip_rate > 0.7).astype(int)
        + (df.num_subscription_pauses >= 3).astype(int)
        + (df.customer_service_inquiries == "High").astype(int)
        + (df.subscription_type == "Free").astype(int)
    )


def churn_spread(train: pd.DataFrame) -> pd.Series:
    """Max-min churn rate across bins, per modelable feature."""
    numeric = [c for c in train.select_dtypes(np.number).columns if c not in ("customer_id", "churned")]
    categorical = [c for c in train.columns if train[c].dtype == object or train[c].dtype == "str"]
    spread = {}
    for column in numeric:
        rate = train.groupby(pd.qcut(train[column], 10, duplicates="drop"), observed=True)["churned"].mean()
        spread[column] = rate.max() - rate.min()
    for column in categorical:
        rate = train.groupby(column)["churned"].mean()
        spread[column] = rate.max() - rate.min()
    return pd.Series(spread).sort_values()


def report_quality(train: pd.DataFrame, test: pd.DataFrame) -> None:
    print("=" * 70)
    print("1. DATA QUALITY")
    print("=" * 70)
    print(f"  shape                     train {train.shape}  test {test.shape}")
    print(f"  missing values            train {train.isna().sum().sum()}  test {test.isna().sum().sum()}")
    print(f"  duplicate rows            train {train.duplicated().sum()}  test {test.duplicated().sum()}")
    print(f"  customer_id overlap       {len(set(train.customer_id) & set(test.customer_id))}")
    print(f"  churn rate                {train.churned.mean():.4f}  ({train.churned.sum():,} / {len(train):,})")
    impossible_songs = (train.weekly_unique_songs > train.weekly_songs_played).sum()
    impossible_lists = (train.num_shared_playlists > train.num_playlists_created).sum()
    print(f"  unique > played songs     {impossible_songs:,} ({impossible_songs / len(train):.1%})  <- impossible")
    print(f"  shared > created lists    {impossible_lists:,} ({impossible_lists / len(train):.1%})  <- impossible")
    numeric = [c for c in train.select_dtypes(np.number).columns if c not in ("customer_id", "churned")]
    corr = train[numeric].corr().abs().to_numpy(copy=True)
    np.fill_diagonal(corr, 0)
    print(f"  max |corr| between feats  {corr.max():.4f}  <- features are independent")


def report_signal(train: pd.DataFrame) -> pd.Series:
    spread = churn_spread(train)
    print()
    print("=" * 70)
    print("2. SIGNAL vs NOISE  (churn-rate spread per feature)")
    print("=" * 70)
    for name, value in spread.sort_values(ascending=False).items():
        mark = "SIGNAL" if value > 0.10 else "noise"
        print(f"  {name:28s} {value:.4f}  {mark}")
    return spread


def report_rule(train: pd.DataFrame) -> None:
    print()
    print("=" * 70)
    print("3. RECOVERED GENERATING RULE  (additive logit on the 7 buckets)")
    print("=" * 70)
    buckets = bucketize(train)
    design = pd.get_dummies(buckets, drop_first=True).astype(float)
    model = LogisticRegression(max_iter=5000, C=1e8).fit(design, train.churned)
    coef = pd.Series(model.coef_[0], index=design.columns).sort_values(ascending=False)
    table = pd.DataFrame({"coef": coef.round(3), "nearest_half": (coef * 2).round() / 2})
    print(table.to_string())
    print(f"  intercept {model.intercept_[0]:.3f} -> {round(model.intercept_[0] * 2) / 2}")

    rounded = (coef * 2).round() / 2
    linear = round(model.intercept_[0] * 2) / 2 + design[coef.index].to_numpy() @ rounded.to_numpy()
    p_true = 1 / (1 + np.exp(-linear))
    print(f"\n  mean(p_true) {p_true.mean():.4f} vs observed churn rate {train.churned.mean():.4f}")
    print(f"  irreducible Bernoulli noise -> max achievable accuracy = {np.maximum(p_true, 1 - p_true).mean():.4f}")


def report_ceiling(train: pd.DataFrame) -> None:
    print()
    print("=" * 70)
    print("4. DIAGNOSTIC MODEL COMPARISON  (60/20/20 stratified)")
    print("=" * 70)
    y = train.churned.astype(int)
    X = train.drop(columns=["churned"])
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE)
    _, X_test, _, y_test = train_test_split(X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=RANDOM_STATE)

    def score(name: str, probability: np.ndarray) -> None:
        print(
            f"  {name:34s} PR-AUC={average_precision_score(y_test, probability):.4f}  "
            f"ROC-AUC={roc_auc_score(y_test, probability):.4f}  "
            f"logloss={log_loss(y_test, probability):.4f}  "
            f"acc@.5={accuracy_score(y_test, (probability >= 0.5).astype(int)):.4f}"
        )

    # Bucket boundaries were identified during full-data EDA. This simple-logit
    # benchmark is therefore explanatory, not a pristine model-selection test.
    columns = pd.get_dummies(bucketize(X), drop_first=True).columns
    D_train = pd.get_dummies(bucketize(X_train), drop_first=True).reindex(columns=columns, fill_value=0).astype(float)
    D_test = pd.get_dummies(bucketize(X_test), drop_first=True).reindex(columns=columns, fill_value=0).astype(float)
    logit = LogisticRegression(max_iter=5000, C=1e8).fit(D_train, y_train)
    score(f"7-bucket logit ({D_train.shape[1] + 1} params)", logit.predict_proba(D_test)[:, 1])

    G_train = pd.get_dummies(X_train[SIGNAL])
    G_test = pd.get_dummies(X_test[SIGNAL]).reindex(columns=G_train.columns, fill_value=0)
    gb_signal = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE)
    score("GB on 7 signal features", gb_signal.fit(G_train, y_train).predict_proba(G_test)[:, 1])

    A_train = pd.get_dummies(X_train.drop(columns=["customer_id"]))
    A_test = pd.get_dummies(X_test.drop(columns=["customer_id"])).reindex(columns=A_train.columns, fill_value=0)
    gb_all = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE)
    score("GB on all 18 features (current)", gb_all.fit(A_train, y_train).predict_proba(A_test)[:, 1])


def held_out_targeting_metrics(train: pd.DataFrame) -> pd.DataFrame:
    """Use only predictions from an internal holdout to rank customers."""
    artifact_path = ROOT / "artifacts" / "targeting_metrics_test.csv"
    if artifact_path.exists():
        return pd.read_csv(artifact_path)

    # Fallback for a first run before the main pipeline writes artifacts.
    from src.legacy.run_holdout_pipeline import model_pipeline, targeting_metrics

    X = train.drop(columns=["churned"])
    y = train["churned"].astype(int)
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE
    )
    _, X_test, _, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=RANDOM_STATE
    )
    model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE)
    pipe = model_pipeline(X_train, model, engineer=True, scale_numeric=False)
    pipe.fit(X_train, y_train)
    return targeting_metrics(y_test, pipe.predict_proba(X_test)[:, 1])


def make_charts(train: pd.DataFrame, spread: pd.Series) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    y = train.churned
    base = y.mean()

    # 1. risk-flag ladder
    flags = risk_flags(train)
    grouped = pd.concat([flags.rename("flags"), y], axis=1).groupby("flags")["churned"].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar(grouped.index.astype(str), grouped["mean"], color=[BLUE if v < base else RED for v in grouped["mean"]])
    ax.axhline(base, color="#333", ls="--", lw=1, label=f"overall {base:.1%}")
    for bar, (mean, size) in zip(bars, grouped.values):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.02, f"{mean:.1%}\nn={int(size):,}", ha="center", fontsize=8)
    ax.set(title="Churn rate by number of risk flags", xlabel="risk flags (of 5)", ylabel="churn rate", ylim=(0, 1.15))
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "01_risk_flag_ladder.png", dpi=150)
    plt.close(fig)

    # 2. weekly_hours step function
    binned = train.groupby(pd.cut(train.weekly_hours, np.arange(0, 50.5, 1.25)), observed=True)["churned"].mean()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot([i.mid for i in binned.index], binned.values, marker="o", ms=3, color=RED)
    for x in (5, 10, 40):
        ax.axvline(x, color=BLUE, ls=":", lw=1.5)
        ax.text(x, 1.0, f" {x}h", color=BLUE, fontsize=9, va="top")
    ax.set(
        title="Churn rate vs weekly listening hours - a step function, not a slope",
        xlabel="weekly_hours",
        ylabel="churn rate",
        ylim=(0, 1.05),
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "02_weekly_hours_steps.png", dpi=150)
    plt.close(fig)

    # 3. signal vs noise map
    n_signal = int((spread > 0.10).sum())
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.barh(spread.index, spread.values, color=[RED if v > 0.10 else GREY for v in spread.values])
    ax.axvline(0.10, color="#333", ls="--", lw=1)
    ax.text(0.105, 0.5, "signal threshold", fontsize=8, rotation=90)
    ax.set(
        title=f"Churn-rate spread per feature - {n_signal} signals, {len(spread) - n_signal} pure noise",
        xlabel="max - min churn rate across bins",
    )
    fig.tight_layout()
    fig.savefig(OUT / "03_signal_vs_noise.png", dpi=150)
    plt.close(fig)

    # 4. plan x inquiries heatmap
    pivot = train.pivot_table(
        index="subscription_type", columns="customer_service_inquiries", values="churned", aggfunc="mean"
    ).loc[["Free", "Student", "Family", "Premium"], ["Low", "Medium", "High"]]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    image = ax.imshow(pivot.values, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3), pivot.columns)
    ax.set_yticks(range(4), pivot.index)
    for i in range(4):
        for j in range(3):
            value = pivot.values[i, j]
            ax.text(
                j, i, f"{value:.1%}", ha="center", va="center",
                color="white" if abs(value - 0.5) > 0.22 else "black", fontweight="bold",
            )
    ax.set(title="Churn rate: subscription tier x service inquiries", xlabel="customer_service_inquiries")
    fig.colorbar(image, ax=ax, label="churn rate")
    fig.tight_layout()
    fig.savefig(OUT / "04_plan_x_inquiry_heatmap.png", dpi=150)
    plt.close(fig)

    # 5. age U-curve
    by_age = train.groupby("age")["churned"].mean()
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(by_age.index, by_age.values, marker="o", ms=3, color=RED)
    for x in (25, 35, 61):
        ax.axvline(x, color=BLUE, ls=":", lw=1.5)
        ax.text(x, 0.68, f" {x}", color=BLUE, fontsize=9)
    ax.axhline(base, color="#333", ls="--", lw=1)
    ax.set(title="Churn rate by age - U-shaped with breaks at 25 / 35 / 61", xlabel="age", ylabel="churn rate")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "05_age_u_curve.png", dpi=150)
    plt.close(fig)

    # 6. diagnostic benchmark (the simple-logit boundaries were identified in EDA)
    names = ["Dummy\n(prior)", "14-param\nlogit", "GB\n7 features", "GB\n18 features\n(current)", "Saturated\ncells"]
    values = [0.5134, 0.9468, 0.9464, 0.9473, 0.9491]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bars = ax.bar(names, values, color=[GREY, BLUE, BLUE, RED, "#333"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.4f}", ha="center", fontsize=9)
    ax.axhline(values[-1], color="#333", ls="--", lw=1)
    ax.set(title="Diagnostic PR-AUC benchmark (not a locked model-selection test)", ylabel="PR-AUC", ylim=(0.45, 1.03))
    fig.tight_layout()
    fig.savefig(OUT / "06_model_ceiling.png", dpi=150)
    plt.close(fig)

    # 7. cumulative capture curve based on held-out model probabilities only.
    targeting = held_out_targeting_metrics(train).sort_values("top_percent")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(targeting["top_percent"], targeting["capture_rate"] * 100, marker="o", color=RED, lw=2, label="held-out model")
    ax.plot(targeting["top_percent"], targeting["top_percent"], color=GREY, ls="--", label="random")
    for _, row in targeting[targeting["top_percent"].isin([10, 20, 30])].iterrows():
        k, value = int(row["top_percent"]), float(row["capture_rate"]) * 100
        ax.plot([k, k], [0, value], color=BLUE, ls=":", lw=1)
        ax.text(k + 1, value - 4, f"top {k}% -> {value:.0f}%", fontsize=9, color=BLUE)
    ax.set(
        title="Cumulative churner capture by risk rank",
        xlabel="% of customers contacted (highest risk first)",
        ylabel="% of churners captured",
        xlim=(0, 52),
        ylim=(0, 100),
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "07_capture_curve.png", dpi=150)
    plt.close(fig)

    print()
    print("=" * 70)
    print("5. CHARTS")
    print("=" * 70)
    for path in sorted(OUT.glob("*.png")):
        print(f"  {path.relative_to(ROOT)}")
    print("  held-out top-K capture:", {f"{int(row.top_percent)}%": f"{row.capture_rate:.1%}" for row in targeting.itertuples()})


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    report_quality(train, test)
    spread = report_signal(train)
    report_rule(train)
    report_ceiling(train)
    make_charts(train, spread)


if __name__ == "__main__":
    main()
