"""Reusable prediction-time feature transformation for the music churn model."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MusicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create deterministic features without using the target or customer ID."""

    def __init__(self, engineer: bool = True):
        self.engineer = engineer

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        df = df.drop(columns=["customer_id"], errors="ignore")

        if self.engineer:
            if pd.api.types.is_numeric_dtype(df["signup_date"]):
                # 신규 데이터: signup_date가 기준일 대비 일수(음수=과거)로 제공됨.
                df["signup_days_ago"] = (-df["signup_date"]).astype("float64")
            else:
                signup = pd.to_datetime(df["signup_date"], errors="coerce")
                df["signup_year"] = signup.dt.year.astype("float64")
                df["signup_month"] = signup.dt.month.astype("float64")
            df["unique_song_ratio"] = df["weekly_unique_songs"] / (df["weekly_songs_played"] + 1.0)
            df["shared_playlist_ratio"] = df["num_shared_playlists"] / (df["num_playlists_created"] + 1.0)
            df["hours_per_song"] = df["weekly_hours"] / (df["weekly_songs_played"] + 1.0)
            df["friends_per_playlist"] = df["num_platform_friends"] / (df["num_playlists_created"] + 1.0)

        # signup_date has no verified prediction snapshot. Raw date strings are
        # therefore removed; only the optional derived fields remain.
        return df.drop(columns=["signup_date"], errors="ignore")
