import json
import time
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Load configuration
with open("config.json") as f:
    config = json.load(f)

data_file = config["LOGGING"]["log_file"]
refresh_interval = config["LOGGING"]["interval_sec"]
output_html = "improved_live_plot.html"

# Load the sensor data CSV
def load_data():
    if not os.path.exists(data_file):
        print("[WARN] Data file not found.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(data_file)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])  # Match your CSV
        return df
    except Exception as e:
        print(f"[ERROR] Could not load data: {e}")
        return pd.DataFrame()

# Create 3-panel subplot chart
def plot_graph(df):
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            "🌡️ Temperature Over Time",
            "💧 Humidity Over Time",
            "🚶 Motion Over Time"
        )
    )

    # Temperature plot
    fig.add_trace(go.Scatter(
        x=df['Timestamp'], y=df['Temperature'],
        mode='lines+markers',
        name='Temperature',
        line=dict(color='orange')
    ), row=1, col=1)

    # Humidity plot
    fig.add_trace(go.Scatter(
        x=df['Timestamp'], y=df['Humidity'],
        mode='lines+markers',
        name='Humidity',
        line=dict(color='blue')
    ), row=2, col=1)

    # Motion plot
    fig.add_trace(go.Scatter(
        x=df['Timestamp'], y=df['Motion'],
        mode='lines+markers',
        name='Motion',
        line=dict(color='green')
    ), row=3, col=1)

    fig.update_layout(
        height=850,
        width=1000,
        title="📊 Live Sensor Monitoring (Temperature, Humidity, Motion)",
        template="plotly_white",
        xaxis_title="Time",
        showlegend=False
    )

    # Save to HTML
    fig.write_html(output_html)
    print(f"✅ Chart saved as {output_html}")

# Main loop
print("📊 Improved Live Plotting Started. Press CTRL+C to stop...\n")

try:
    while True:
        df = load_data()
        if not df.empty:
            plot_graph(df)
        time.sleep(refresh_interval)

except KeyboardInterrupt:
    print("\n🛑 Live plotting stopped.")
