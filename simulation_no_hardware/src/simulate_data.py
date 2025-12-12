"""
simulate_data.py

Creates a long-running, minute-level "home sensor" dataset with:
- temperature (C)
- humidity (%)
- motion_count (events per minute)

Why simulated?
- No hardware needed, but you can still demonstrate the core claim:
  anomaly detection + persistence filtering + human-readable explanations.

We inject a few anomalies that are intentionally *persistent* (behavior change),
and we also inject short noise spikes to verify the persistence filter.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
import numpy as np
import pandas as pd


RAW_OUT = "data/raw/simulated_sensor_minute.csv"


@dataclass
class SimConfig:
    days: int = 240              # ~8 months
    seed: int = 7
    start: str = "2025-01-01"    # arbitrary; doesn't matter as long as it's long-running


def ensure_dirs() -> None:
    os.makedirs("data/raw", exist_ok=True)


def simulate(cfg: SimConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    idx = pd.date_range(cfg.start, periods=cfg.days * 24 * 60, freq="min")
    df = pd.DataFrame({"timestamp": idx})

    # Basic time context
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    dow = df["timestamp"].dt.dayofweek  # 0=Mon

    # --- Temperature: daily sinusoid + small weekly wobble + noise
    # This isn't "physics"; it's just stable enough to look realistic.
    temp_daily = 22 + 2.0 * np.sin(2 * np.pi * (hour - 14) / 24)  # warmer afternoons
    temp_week = 0.2 * np.cos(2 * np.pi * dow / 7)
    df["temp_c"] = temp_daily + temp_week + rng.normal(0, 0.25, len(df))

    # --- Humidity: shifted sinusoid + noise
    hum_daily = 45 + 6.0 * np.sin(2 * np.pi * (hour - 3) / 24)
    df["humidity_pct"] = hum_daily + rng.normal(0, 1.2, len(df))

    # --- Motion: simple circadian occupancy proxy
    is_night = (hour < 6) | (hour >= 23)
    base_rate = np.where(is_night, 0.05, 0.35)  # expected motion events/min
    weekend = (dow >= 5)
    base_rate = base_rate + np.where(weekend, 0.05, 0.0)
    lam = np.clip(base_rate, 0.01, 1.0)
    df["motion_count"] = rng.poisson(lam=lam, size=len(df)).astype(int)

    # We keep anomaly labels for evaluation/demo only.
    df["anomaly_type"] = "none"

    # -----------------------------
    # Inject anomalies (persistent)
    # -----------------------------
    # A1: persistent motion drop (10 days) – like prolonged inactivity
    a1_start = df.loc[int(len(df) * 0.35), "timestamp"]
    a1_end = a1_start + pd.Timedelta(days=10)
    m = (df["timestamp"] >= a1_start) & (df["timestamp"] < a1_end)
    df.loc[m, "motion_count"] = (df.loc[m, "motion_count"] * 0.25).astype(int)
    df.loc[m, "anomaly_type"] = "persistent_motion_drop"

    # A2: erratic night motion (7 nights) – like sleep disruption
    a2_start = df.loc[int(len(df) * 0.55), "timestamp"]
    a2_end = a2_start + pd.Timedelta(days=7)
    night = (df["timestamp"].dt.hour < 6) | (df["timestamp"].dt.hour >= 23)
    m2 = (df["timestamp"] >= a2_start) & (df["timestamp"] < a2_end) & night
    df.loc[m2, "motion_count"] += rng.poisson(lam=0.6, size=m2.sum()).astype(int)
    df.loc[m2, "anomaly_type"] = "erratic_night_motion"

    # A3: humidity drift (20 days) – could be environment change or sensor drift
    a3_start = df.loc[int(len(df) * 0.70), "timestamp"]
    a3_end = a3_start + pd.Timedelta(days=20)
    m3 = (df["timestamp"] >= a3_start) & (df["timestamp"] < a3_end)
    drift = np.linspace(0, 12, m3.sum())
    df.loc[m3, "humidity_pct"] = df.loc[m3, "humidity_pct"].to_numpy() + drift
    df.loc[m3, "anomaly_type"] = "humidity_drift"

    # -----------------------------
    # Inject short noise spikes
    # -----------------------------
    # These are deliberately brief so that a persistence rule should ignore them.
    spike_idx = rng.choice(len(df), size=60, replace=False)
    df.loc[spike_idx, "motion_count"] += rng.integers(3, 8, size=60)

    # Small missingness (keeps it honest; models should tolerate real logging mess)
    miss_idx = rng.choice(len(df), size=int(0.002 * len(df)), replace=False)
    df.loc[miss_idx, "temp_c"] = np.nan
    df.loc[miss_idx, "humidity_pct"] = np.nan

    # Lightweight imputation: forward fill is fine for this demo.
    df["temp_c"] = df["temp_c"].ffill()
    df["humidity_pct"] = df["humidity_pct"].ffill()

    return df


def main() -> None:
    ensure_dirs()
    df = simulate(SimConfig())
    df.to_csv(RAW_OUT, index=False)
    print(f"[OK] wrote {RAW_OUT} rows={len(df):,}")


if __name__ == "__main__":
    main()