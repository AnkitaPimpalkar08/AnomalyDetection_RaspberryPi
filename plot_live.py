# plot_live.py
import pandas as pd
import plotly.graph_objects as go
import time
import json
import os

# Load config
with open("config.json") as f:
    CONFIG = json.load(f)

DATA_PATH = CONFIG["LOGGING"]["anomaly_log_file"]
REFRESH_INTERVAL = CONFIG["LOGGING"].get("interval_sec", 2)

print("📊 Live Plotting Started (HTML export mode). Press CTRL+C to stop...\n")

try:
    while True:
        if not os.path.exists(DATA_PATH):
            print("Waiting for data...")
            time.sleep(REFRESH_INTERVAL)
            continue

        df = pd.read_csv(DATA_PATH)
        if df.empty:
            print("No data to plot yet.")
            time.sleep(REFRESH_INTERVAL)
            continue

        fig = go.Figure()

        # Plot sensor values
        fig.add_trace(go.Scatter(
            x=df['Timestamp'], y=df['Temperature'],
            mode='lines+markers', name='Temperature', line=dict(color='red')))

        fig.add_trace(go.Scatter(
            x=df['Timestamp'], y=df['Humidity'],
            mode='lines+markers', name='Humidity', line=dict(color='blue')))

        fig.add_trace(go.Scatter(
            x=df['Timestamp'], y=df['Motion'],
            mode='lines+markers', name='Motion', line=dict(color='green')))

        # Highlight anomalies if present
        if "Prediction" in df.columns:
            anomalies = df[df['Prediction'] == -1]
            fig.add_trace(go.Scatter(
                x=anomalies['Timestamp'], y=anomalies['Temperature'],
                mode='markers', name='Anomalies (Temp)',
                marker=dict(size=10, color='orange', symbol='x')
            ))

        # Chart layout
        fig.update_layout(
            title="Live Sensor Data with Anomalies",
            xaxis_title="Timestamp",
            yaxis_title="Sensor Values",
            xaxis=dict(rangeslider_visible=True),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
            height=600
        )

        # Export as HTML
        fig.write_html("live_plot.html")
        print("📁 Chart saved as live_plot.html — download to your Mac to view.")

        time.sleep(REFRESH_INTERVAL * 5)

except KeyboardInterrupt:
    print("\n🛑 Live plotting stopped.")
