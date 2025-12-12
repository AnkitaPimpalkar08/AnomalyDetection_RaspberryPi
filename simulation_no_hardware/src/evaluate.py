"""
evaluate.py

Evaluation is window-level against the simulated anomaly_type label.
This isn't meant as "publication-grade ground truth", just a sanity check:
- baseline (score threshold only) tends to over-alert
- persistence reduces false positives while keeping recall reasonable
"""

from __future__ import annotations
import pandas as pd


def window_overlaps_labeled_anomaly(raw: pd.DataFrame, ws, we) -> int:
    m = (raw["timestamp"] >= ws) & (raw["timestamp"] <= we)
    if m.sum() == 0:
        return 0
    return int((raw.loc[m, "anomaly_type"] != "none").any())


def metrics(df: pd.DataFrame, pred_col: str, truth_col: str = "true_anomaly") -> dict:
    pred = df[pred_col].astype(int)
    truth = df[truth_col].astype(int)

    tp = int(((pred == 1) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "fpr": fpr}


def main() -> None:
    raw = pd.read_csv("data/raw/simulated_sensor_minute.csv")
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])

    alerts = pd.read_csv("data/outputs/window_alerts.csv")
    alerts["window_start"] = pd.to_datetime(alerts["window_start"])
    alerts["window_end"] = pd.to_datetime(alerts["window_end"])

    truth = []
    for _, r in alerts.iterrows():
        truth.append(window_overlaps_labeled_anomaly(raw, r["window_start"], r["window_end"]))
    alerts["true_anomaly"] = truth

    # Baseline: score-threshold only → "suspicious"
    base = metrics(alerts, "suspicious")
    # Final: persistence-based alert
    final = metrics(alerts, "alert")

    print("\n=== Window-level sanity check (simulated labels) ===")
    print("Baseline (threshold only):", base)
    print("After persistence:", final)

    if base["fpr"] > 0:
        reduction = (base["fpr"] - final["fpr"]) / base["fpr"] * 100.0
        print(f"False positive rate reduction: {reduction:.1f}%")

    # NOTE: If you want recall closer to your SOP claim, tune:
    # - threshold in persistence.apply_persistence()
    # - k/n/min_consecutive
    # - IsolationForest contamination
    # Those are the knobs you'd mention in a methods section.


if __name__ == "__main__":
    main()