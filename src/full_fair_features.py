"""Importable feature transformer for the full fair-comparison candidate."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FullFairFeatureEngineer(BaseEstimator, TransformerMixin):
    """Reproduce the deterministic feature variants used by the full-fair run.

    The class intentionally keeps no fitted state.  Keeping it in ``src`` makes
    the saved pipeline loadable from a fresh Python process instead of relying
    on the original training script's ``__main__`` namespace.
    """

    def __init__(self, variant: str = "plus1_ratios"):
        self.variant = variant

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        data = X.copy().drop(columns=["customer_id"], errors="ignore")

        signup = pd.to_numeric(data["signup_date"], errors="coerce")
        data["signup_days_ago"] = -signup
        data = data.drop(columns=["signup_date"])

        if self.variant != "raw":
            pairs = {
                "unique_song_ratio": ("weekly_unique_songs", "weekly_songs_played"),
                "shared_playlist_ratio": ("num_shared_playlists", "num_playlists_created"),
                "hours_per_song": ("weekly_hours", "weekly_songs_played"),
                "friends_per_playlist": ("num_platform_friends", "num_playlists_created"),
            }
            for name, (numerator, denominator) in pairs.items():
                if self.variant == "zero_aware_ratios":
                    data[name] = np.where(
                        data[denominator].eq(0),
                        0.0,
                        data[numerator] / data[denominator],
                    )
                    data[f"{denominator}_is_zero"] = data[denominator].eq(0).astype(int)
                else:
                    data[name] = data[numerator] / (data[denominator] + 1.0)

        if self.variant == "log_numeric":
            for column in [
                "weekly_hours",
                "weekly_songs_played",
                "weekly_unique_songs",
                "num_platform_friends",
                "num_playlists_created",
                "num_shared_playlists",
            ]:
                data[f"log1p_{column}"] = np.log1p(data[column].clip(lower=0))

        if self.variant == "interaction":
            data["plan_x_inquiry"] = (
                data["subscription_type"].astype(str)
                + "__"
                + data["customer_service_inquiries"].astype(str)
            )

        if self.variant == "signal_pruned":
            keep = [
                "weekly_hours",
                "subscription_type",
                "customer_service_inquiries",
                "num_subscription_pauses",
                "song_skip_rate",
                "age",
                "notifications_clicked",
                "signup_days_ago",
            ]
            data = data[[column for column in keep if column in data]]

        return data.replace([np.inf, -np.inf], np.nan)
