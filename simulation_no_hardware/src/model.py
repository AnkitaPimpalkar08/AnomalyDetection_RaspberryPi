"""
model.py

Trains Isolation Forest and produces a normalized anomaly score in [0,1].

Important:
- This score is NOT a probability. It's a ranking-ish value for interpretability.
- The persistence layer is what turns "scores" into actionable alerts.
"""

from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelArtifacts:
    model: IsolationForest
    scaler: StandardScaler
    feature_columns: list[str]


def min_max(series: pd.Series) -> pd.Series:
    mn, mx = float(series.min()), float(series.max())
    if mx - mn < 1e-8:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def train_iforest(feature_df: pd.DataFrame, feature_cols: list[str]) -> tuple[ModelArtifacts, pd.Series]:
    X = feature_df[feature_cols].copy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # NOTE: contamination is a knob; we keep it modest but not tiny so
    # the baseline generates some false positives that persistence can reduce.
    model = IsolationForest(
        n_estimators=300,
        contamination=0.08,
        random_state=7,
        n_jobs=-1,
    )
    model.fit(Xs)

    # sklearn's score_samples: higher = more normal → invert
    raw = pd.Series(-model.score_samples(Xs), index=feature_df.index)
    score = min_max(raw)

    return ModelArtifacts(model=model, scaler=scaler, feature_columns=feature_cols), score