# Trustworthy Behavioral Anomaly Detection (No Hardware)

- Synthetic "home sensor" streams (temp/humidity/motion proxy)
- Windowed feature engineering + Isolation Forest
- Alerts filtered using temporal persistence (real anomalies persist; spikes don't)
- Plain-language explanations (baseline comparisons over a lookback period)
- Streamlit dashboard to inspect "what changed" instead of only a score

## Setup
```bash
pip install -r requirements.txt
