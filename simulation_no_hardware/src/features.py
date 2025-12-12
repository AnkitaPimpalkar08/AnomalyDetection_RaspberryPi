"""
features.py

Turns each window into a feature vector. These features are chosen so that:
1) the model can detect changes
2) the explanation layer can talk in human terms
   (e.g., "motion dropped 30% over 48h vs baseline")

This is not an exhaustive feature set; it's intentionally small and interpretable.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from .windowing import iter_windows


def _night_mask(ts: pd.Series) -> np.ndarray:
    h = ts.dt.hour
    return ((h < 6) | (h >= 23)).to_numpy()


def _safe_mean(x: np.ndarray) -> float:
    return float(np.nanmean(x)) if len(x) else 0.0


def _safe_std(x: np.ndarray) -> float:
    return float(np.nanstd(x)) if len(x) else 0.0


def _slope(y: np.ndarray) -> float:
    # Slope is helpful for drift-like anomalies.
    if len(y) < 10:
        return 0.0
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def window_features(window_df: pd.DataFrame, window_start, window_end) -> dict:
    ts = pd.to_datetime(window_df["timestamp"])
    motion = window_df["motion_count"].astype(float).to_numpy()
    temp = window_df["temp_c"].astype(float).to_numpy()
    hum = window_df["humidity_pct"].astype(float).to_numpy()

    night = _night_mask(ts)
    day = ~night

    feats = {
        # Window metadata (not fed into the model)
        "window_start": pd.to_datetime(window_start),
        "window_end": pd.to_datetime(window_end),

        # Motion: interpretable "behavior"
        "motion_rate": _safe_mean(motion),
        "motion_std": _safe_std(motion),
        "motion_zero_frac": float(np.mean(motion == 0)),

        "night_motion_rate": _safe_mean(motion[night]),
        "day_motion_rate": _safe_mean(motion[day]),

        # Environmental context: useful for drift explanations
        "temp_mean": _safe_mean(temp),
        "temp_std": _safe_std(temp),
        "temp_slope": _slope(temp),

        "humidity_mean": _safe_mean(hum),
        "humidity_std": _safe_std(hum),
        "humidity_slope": _slope(hum),
    }
    return feats


def build_feature_table(
    raw_df: pd.DataFrame,
    window_minutes: int = 120,
    step_minutes: int = 30,
) -> pd.DataFrame:
    rows = []
    for ws, we, chunk in iter_windows(raw_df, window_minutes, step_minutes):
        rows.append(window_features(chunk, ws, we))

    feat = pd.DataFrame(rows)
    feat = feat.sort_values("window_start").reset_index(drop=True)
    return feat