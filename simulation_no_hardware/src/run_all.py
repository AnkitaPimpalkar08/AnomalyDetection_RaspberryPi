"""
run_all.py

One-button run for the pipeline:
1) simulate data
2) build features
3) train IF + score windows
4) apply persistence → events
5) generate explanations
6) evaluate

This avoids "mystery steps" when someone reviews your repo.
"""

from __future__ import annotations
import os
import joblib
import pandas as pd

from .simulate_data import main as simulate_main
from .features import build_feature_table
from .model import train_iforest
from .persistence import apply_persistence, to_events
from .explain import explain_all


def ensure_dirs() -> None:
    for d in ["data/processed", "data/models", "data/outputs"]:
        os.makedirs(d, exist_ok=True)


def main() -> None:
    ensure_dirs()

    # 1) data
    simulate_main()
    raw = pd.read_csv("data/raw/simulated_sensor_minute.csv")

    # 2) features
    feat = build_feature_table(raw, window_minutes=120, step_minutes=30)
    feat.to_csv("data/processed/window_features.csv", index=False)

    # 3) model
    non_feature = {"window_start", "window_end"}
    feature_cols = [c for c in feat.columns if c not in non_feature]

    artifacts, score = train_iforest(feat, feature_cols)
    feat["anomaly_score"] = score.values

    # 4) persistence → alerts + events
    alerts = apply_persistence(
        feat,
        score_col="anomaly_score",
        threshold=0.65,
        k=3,
        n=5,
        min_consecutive=2,
    )
    alerts.to_csv("data/outputs/window_alerts.csv", index=False)

    events = to_events(alerts)
    events.to_csv("data/outputs/anomaly_events.csv", index=False)

    # 5) explanations
    explanations = explain_all(alerts, events, baseline_days=14)
    explanations.to_csv("data/outputs/explanations.csv", index=False)

    # 6) persist model artifacts
    joblib.dump(
        {
            "model": artifacts.model,
            "scaler": artifacts.scaler,
            "feature_columns": artifacts.feature_columns,
        },
        "data/models/iforest.joblib",
    )

    print("\n[OK] pipeline complete")
    print(f"events_detected={len(events)} (see data/outputs/anomaly_events.csv)")


if __name__ == "__main__":
    main()