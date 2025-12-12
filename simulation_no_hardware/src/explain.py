"""
explain.py

This is the "trust" layer: instead of showing only a number, we translate
what changed into baseline comparisons that a human can act on.

We deliberately keep explanations template-based (not LLM-based),
because it's:
- predictable
- auditable
- runs locally
"""

from __future__ import annotations
import pandas as pd


def pct_change(curr: float, base: float) -> float:
    if base == 0:
        return 0.0
    return (curr - base) / abs(base) * 100.0


def explain_event(
    window_df: pd.DataFrame,
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
    baseline_days: int = 14,
) -> dict:
    df = window_df.copy()
    df["window_start"] = pd.to_datetime(df["window_start"])
    df["window_end"] = pd.to_datetime(df["window_end"])

    event_mask = (df["window_start"] >= event_start) & (df["window_end"] <= event_end)

    base_start = event_start - pd.Timedelta(days=baseline_days)
    baseline_mask = (df["window_start"] >= base_start) & (df["window_start"] < event_start)

    event = df.loc[event_mask]
    base = df.loc[baseline_mask]

    if len(event) < 2 or len(base) < 10:
        return {
            "event_start": event_start,
            "event_end": event_end,
            "explanation": (
                "Not enough baseline context to explain this event confidently. "
                "Try collecting more history (or reduce baseline_days)."
            ),
            "top_changes": "",
        }

    cols = [
        "motion_rate", "night_motion_rate", "day_motion_rate",
        "motion_zero_frac",
        "humidity_mean", "humidity_slope",
        "temp_mean", "temp_slope",
    ]

    event_mean = event[cols].mean(numeric_only=True)
    base_mean = base[cols].mean(numeric_only=True)

    deltas = {c: pct_change(float(event_mean[c]), float(base_mean[c])) for c in cols}
    ranked = sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]

    bullets = []
    for name, pct in ranked:
        direction = "increased" if pct > 0 else "decreased"
        mag = abs(pct)

        if name == "motion_rate":
            bullets.append(f"Overall motion {direction} ~{mag:.0f}% vs baseline.")
        elif name == "night_motion_rate":
            bullets.append(f"Night motion {direction} ~{mag:.0f}% vs baseline.")
        elif name == "day_motion_rate":
            bullets.append(f"Day motion {direction} ~{mag:.0f}% vs baseline.")
        elif name == "motion_zero_frac":
            bullets.append(f"Inactive minutes {direction} ~{mag:.0f}% (more/less no-movement time).")
        elif name == "humidity_mean":
            bullets.append(f"Average humidity {direction} ~{mag:.0f}% vs baseline.")
        elif name == "humidity_slope":
            bullets.append(f"Humidity trend changed ~{mag:.0f}% vs baseline (drift-like behavior).")
        elif name == "temp_mean":
            bullets.append(f"Average temperature {direction} ~{mag:.0f}% vs baseline.")
        elif name == "temp_slope":
            bullets.append(f"Temperature trend changed ~{mag:.0f}% vs baseline.")

    duration_hours = (event_end - event_start) / pd.Timedelta(hours=1)

    explanation = (
        f"Why flagged: behavior changed compared to the prior {baseline_days} days.\n"
        f"Timeframe: {event_start} → {event_end} (~{duration_hours:.1f} hours).\n"
        f"Key changes: " + " ".join(bullets) + "\n"
        "Suggestion: If this pattern is unexpected for the person/space, consider checking in."
    )

    return {
        "event_start": event_start,
        "event_end": event_end,
        "explanation": explanation,
        "top_changes": " | ".join(bullets),
    }


def explain_all(
    window_df: pd.DataFrame,
    events_df: pd.DataFrame,
    baseline_days: int = 14,
) -> pd.DataFrame:
    rows = []
    for _, e in events_df.iterrows():
        rows.append(
            explain_event(
                window_df,
                pd.to_datetime(e["event_start"]),
                pd.to_datetime(e["event_end"]),
                baseline_days=baseline_days,
            )
        )
    return pd.DataFrame(rows)