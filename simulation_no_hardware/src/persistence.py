"""
persistence.py

Turns window anomaly scores into alerts using the SOP insight:
- Real anomalies tend to persist across multiple time windows.
- Noise tends to spike once and disappear.

We provide two simple rules:
1) k-of-n: alert if >=k of the last n windows are suspicious
2) consecutive: alert if suspicious for >=m windows in a row

You can use one or both. For demo clarity we use both.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def apply_persistence(
    df: pd.DataFrame,
    score_col: str = "anomaly_score",
    threshold: float = 0.65,
    k: int = 3,
    n: int = 5,
    min_consecutive: int = 2,
) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)

    suspicious = (out[score_col] >= threshold).astype(int)
    out["suspicious"] = suspicious

    # k-of-n rule
    out["suspicious_in_last_n"] = suspicious.rolling(window=n, min_periods=1).sum()
    out["alert_kn"] = (out["suspicious_in_last_n"] >= k).astype(int)

    # consecutive rule
    consec = np.zeros(len(out), dtype=int)
    run = 0
    for i, s in enumerate(suspicious):
        run = run + 1 if s == 1 else 0
        consec[i] = 1 if run >= min_consecutive else 0
    out["alert_consecutive"] = consec

    out["alert"] = ((out["alert_kn"] == 1) | (out["alert_consecutive"] == 1)).astype(int)
    return out


def to_events(alert_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge adjacent alert windows into events (start/end).
    This is what we explain to humans, not each tiny window.
    """
    df = alert_df.sort_values("window_start").reset_index(drop=True)

    events = []
    in_event = False
    cur = None

    for _, row in df.iterrows():
        if row["alert"] == 1 and not in_event:
            in_event = True
            cur = {
                "event_start": row["window_start"],
                "event_end": row["window_end"],
                "max_score": float(row["anomaly_score"]),
                "mean_score_sum": float(row["anomaly_score"]),
                "num_windows": 1,
            }
        elif row["alert"] == 1 and in_event:
            cur["event_end"] = row["window_end"]
            cur["max_score"] = max(cur["max_score"], float(row["anomaly_score"]))
            cur["mean_score_sum"] += float(row["anomaly_score"])
            cur["num_windows"] += 1
        elif row["alert"] == 0 and in_event:
            cur["mean_score"] = cur["mean_score_sum"] / max(cur["num_windows"], 1)
            cur.pop("mean_score_sum", None)
            events.append(cur)
            in_event = False
            cur = None

    if in_event and cur is not None:
        cur["mean_score"] = cur["mean_score_sum"] / max(cur["num_windows"], 1)
        cur.pop("mean_score_sum", None)
        events.append(cur)

    return pd.DataFrame(events)