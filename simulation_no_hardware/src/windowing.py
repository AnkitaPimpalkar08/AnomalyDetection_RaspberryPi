"""
windowing.py

Windowing is the backbone of the "persistence" idea:
- we score each window
- we alert only if the score stays high across neighboring windows

We intentionally keep this simple (time-based slicing).
"""

from __future__ import annotations
import pandas as pd


def iter_windows(
    df: pd.DataFrame,
    window_minutes: int,
    step_minutes: int,
):
    """
    Yield (window_start, window_end, window_df) for overlapping windows.
    Assumes `timestamp` column exists.
    """
    tmp = df.copy()
    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"])
    tmp = tmp.sort_values("timestamp").set_index("timestamp")

    t0 = tmp.index.min()
    t1 = tmp.index.max()

    win = pd.Timedelta(minutes=window_minutes)
    step = pd.Timedelta(minutes=step_minutes)

    t = t0
    # NOTE: inclusive slicing means the window will have (window_minutes+1) rows;
    # for minute-level data it's not a big deal, and it's simpler to reason about.
    while t + win <= t1:
        chunk = tmp.loc[t : t + win].reset_index()
        yield (t, t + win, chunk)
        t += step