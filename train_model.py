import numpy as np
import pandas as pd
import os
import json
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Load config
with open("config.json") as f:
    CONFIG = json.load(f)

LOG_FILE = CONFIG["LOGGING"]["log_file"]
MODEL_PATH = CONFIG["MODEL"]["model_path"]
ROLLING = CONFIG["MODEL"]["rolling_window"]
CONTAM = CONFIG["MODEL"]["contamination"]

# Load data
if not os.path.exists(LOG_FILE):
    raise FileNotFoundError(f"Sensor log file not found: {LOG_FILE}")

df = pd.read_csv(LOG_FILE)
df.dropna(inplace=True)  # Drop rows with missing values

# Apply rolling average
features = df[["Temperature", "Humidity", "Motion"]].rolling(ROLLING).mean().dropna()

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Train Isolation Forest model
model = IsolationForest(contamination=CONTAM, random_state=42)
model.fit(X_scaled)

# Save model and scaler
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)  # Ensure model directory exists
joblib.dump({"model": model, "scaler": scaler}, MODEL_PATH)

print("✅ Model trained and saved to:", MODEL_PATH)
print(f"Trained on {len(X_scaled)} records.")
