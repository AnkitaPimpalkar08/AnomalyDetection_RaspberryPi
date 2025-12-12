import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Trustworthy Anomaly Detection", layout="wide")

RAW = "data/raw/simulated_sensor_minute.csv"
ALERTS = "data/outputs/window_alerts.csv"
EVENTS = "data/outputs/anomaly_events.csv"
EXPL = "data/outputs/explanations.csv"

@st.cache_data
def load():
    raw = pd.read_csv(RAW)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])

    alerts = pd.read_csv(ALERTS)
    alerts["window_start"] = pd.to_datetime(alerts["window_start"])
    alerts["window_end"] = pd.to_datetime(alerts["window_end"])

    events = pd.read_csv(EVENTS)
    if len(events):
        events["event_start"] = pd.to_datetime(events["event_start"])
        events["event_end"] = pd.to_datetime(events["event_end"])

    expl = pd.read_csv(EXPL)
    if len(expl):
        expl["event_start"] = pd.to_datetime(expl["event_start"])
        expl["event_end"] = pd.to_datetime(expl["event_end"])

    return raw, alerts, events, expl

raw, alerts, events, expl = load()

st.title("Trustworthy Behavioral Anomaly Detection (No Hardware)")
st.caption("Isolation Forest + persistence filtering + plain-language explanations")

left, right = st.columns([1, 1])

with left:
    min_t, max_t = raw["timestamp"].min(), raw["timestamp"].max()

    start = st.date_input("Start", value=min_t.date(), min_value=min_t.date(), max_value=max_t.date())
    end = st.date_input("End", value=min((min_t + pd.Timedelta(days=14)).date(), max_t.date()),
                        min_value=min_t.date(), max_value=max_t.date())

    show_truth = st.checkbox("Show simulated anomaly labels (demo only)", value=True)

start_ts = pd.Timestamp(start)
end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)

raw_slice = raw[(raw["timestamp"] >= start_ts) & (raw["timestamp"] <= end_ts)].copy()
alerts_slice = alerts[(alerts["window_start"] >= start_ts) & (alerts["window_end"] <= end_ts)].copy()

with right:
    st.markdown(
        """
**What you get (instead of only “0.87”):**
- A timeline of window scores
- Alerts that require *persistence* (filters one-off spikes)
- Event summaries + human-readable “what changed” explanations
        """
    )

st.divider()

c1, c2, c3 = st.columns(3)
c1.metric("Minutes shown", f"{len(raw_slice):,}")
c2.metric("Alert windows", f"{int((alerts_slice['alert'] == 1).sum()):,}")
c3.metric("Detected events (overall)", f"{len(events):,}")

st.subheader("Signals")
st.plotly_chart(px.line(raw_slice, x="timestamp", y="motion_count", title="Motion count (events/min)"),
                use_container_width=True)
st.plotly_chart(px.line(raw_slice, x="timestamp", y="temp_c", title="Temperature (C)"),
                use_container_width=True)
st.plotly_chart(px.line(raw_slice, x="timestamp", y="humidity_pct", title="Humidity (%)"),
                use_container_width=True)

if show_truth:
    st.subheader("Simulated labels (validation aid)")
    st.dataframe(raw_slice[raw_slice["anomaly_type"] != "none"][["timestamp", "anomaly_type"]].head(200),
                 use_container_width=True)

st.divider()

st.subheader("Window anomaly scores + alerts")
if len(alerts_slice) == 0:
    st.info("No windows in this date range. Expand the range.")
else:
    st.plotly_chart(px.line(alerts_slice, x="window_start", y="anomaly_score", title="Anomaly score per window"),
                    use_container_width=True)

    # show alerts as scatter overlay
    scatter = alerts_slice.copy()
    scatter["label"] = scatter["alert"].map({0: "no alert", 1: "ALERT"})
    st.plotly_chart(px.scatter(scatter, x="window_start", y="anomaly_score", symbol="label",
                               title="Alerts after persistence filter"),
                    use_container_width=True)

st.divider()

st.subheader("Events + explanations")
if len(events) == 0:
    st.warning("No events detected. If you want more alerts, lower the threshold in src/run_all.py.")
else:
    ev = events[(events["event_start"] >= start_ts) & (events["event_end"] <= end_ts)].copy()
    if len(ev) == 0:
        st.info("No events in this date range. Expand the range.")
    else:
        ev2 = ev.merge(expl, on=["event_start", "event_end"], how="left")
        st.dataframe(ev2[["event_start", "event_end", "max_score", "mean_score", "num_windows", "top_changes"]],
                     use_container_width=True)

        chosen = st.selectbox("Pick an event to read full explanation", ev2.index.tolist())
        row = ev2.loc[chosen]
        st.code(row["explanation"] if isinstance(row["explanation"], str) else "No explanation available.")