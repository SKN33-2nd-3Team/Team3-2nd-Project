"""Descriptive retention-planning output; it does not estimate causal campaign uplift."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
pd.set_option("display.width", 220)


def risk_flags(df: pd.DataFrame) -> pd.Series:
    """Five observed risk signals used only for descriptive segmentation."""
    return (
        (df.weekly_hours <= 10).astype(int)
        + (df.song_skip_rate > 0.7).astype(int)
        + (df.num_subscription_pauses >= 3).astype(int)
        + (df.customer_service_inquiries == "High").astype(int)
        + (df.subscription_type == "Free").astype(int)
    )


def observed_segment_table(train: pd.DataFrame) -> pd.DataFrame:
    """Show candidate audiences and observed label rates, not intervention effects."""
    segments = {
        "서비스 마찰 신호": (train.customer_service_inquiries == "High") & (train.song_skip_rate > 0.7),
        "저사용·정지 신호": (train.weekly_hours <= 10) & (train.num_subscription_pauses >= 3),
        "Free 저사용": (train.subscription_type == "Free") & (train.weekly_hours <= 10),
        "Free 고사용·상담 Low": (
            (train.subscription_type == "Free")
            & (train.weekly_hours > 40)
            & (train.song_skip_rate <= 0.7)
            & (train.customer_service_inquiries == "Low")
        ),
    }
    rows = []
    for name, mask in segments.items():
        subset = train.loc[mask, "churned"]
        rows.append(
            {
                "candidate_segment": name,
                "customers": int(mask.sum()),
                "observed_churn_rate": float(subset.mean()),
                "observed_churners": int(subset.sum()),
                "interpretation": "Observed association only; validate a proposed action with an experiment.",
            }
        )
    return pd.DataFrame(rows).sort_values("observed_churn_rate", ascending=False)


def flag_tiers(train: pd.DataFrame) -> pd.DataFrame:
    flags = risk_flags(train)
    tiers = pd.cut(flags, [-1, 1, 2, 5], labels=["T3 모니터링(0-1개)", "T2 검토 후보(2개)", "T1 우선 검토(3개+)"])
    table = pd.concat([tiers.rename("tier"), train["churned"]], axis=1).groupby("tier", observed=True)["churned"].agg(["size", "mean", "sum"])
    table.columns = ["customers", "observed_churn_rate", "observed_churners"]
    table["customer_share"] = table.customers / len(train)
    table["churner_share"] = table.observed_churners / train.churned.sum()
    return table


def print_targeting_sensitivity() -> None:
    path = ARTIFACT_DIR / "targeting_metrics_test.csv"
    if not path.exists():
        print("\nTargeting metrics are not available. Run `python -m src.legacy.run_holdout_pipeline` first.")
        return
    targeting = pd.read_csv(path)
    print("\n" + "=" * 90)
    print("3. HELD-OUT TARGETING CAPTURE (ranking quality, not campaign uplift)")
    print("=" * 90)
    print(targeting[["top_percent", "target_customers", "precision", "capture_rate", "lift"]].to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    print("\n" + "=" * 90)
    print("4. ASSUMPTION-BASED CAMPAIGN SENSITIVITY (not observed campaign outcomes)")
    print("=" * 90)
    value_per_retained_customer = 120_000
    for row in targeting.itertuples():
        for assumed_success_rate in (0.10, 0.20):
            assumed_retained = row.actual_churners_captured * assumed_success_rate
            gross_benefit = assumed_retained * value_per_retained_customer
            breakeven_contact_cost = gross_benefit / row.target_customers
            print(
                f"  top {int(row.top_percent):>2}% | held-out captured churners {int(row.actual_churners_captured):>5,} | "
                f"assumed retention success {assumed_success_rate:.0%} | assumed retained {assumed_retained:>7,.0f} | "
                f"gross-benefit break-even/contact {breakeven_contact_cost:>7,.0f} KRW"
            )
    print("  Note: contact cost, retained value, and success rate are user inputs. This is a planning sensitivity, not an uplift estimate.")


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train.csv")

    print("=" * 90)
    print("1. OBSERVED SEGMENT SIGNALS (not causal levers)")
    print("=" * 90)
    print(observed_segment_table(train).to_string(index=False, float_format=lambda value: f"{value:.1%}" if 0 <= value <= 1 else f"{value:,.0f}"))

    print("\n" + "=" * 90)
    print("2. DESCRIPTIVE FLAG TIERS")
    print("=" * 90)
    print(flag_tiers(train).to_string(float_format=lambda value: f"{value:.1%}" if 0 <= value <= 1 else f"{value:,.0f}"))

    print_targeting_sensitivity()

    print("\n" + "=" * 90)
    print("5. ACTIONABILITY AND MEASUREMENT")
    print("=" * 90)
    print("  customer_service_inquiries  -> CS resolution experiment; KPI: repeat inquiry and later churn label")
    print("  weekly_hours / skip rate    -> reactivation or recommendation experiment; KPI: subsequent engagement and later churn label")
    print("  subscription_type           -> conversion offer experiment; KPI: offer acceptance and later churn label")
    print("  num_subscription_pauses     -> pause-flow intervention experiment; KPI: completed cancellation and later churn label")
    print("  age                         -> not actioned; use only as a fairness-reviewed context field")


if __name__ == "__main__":
    main()
