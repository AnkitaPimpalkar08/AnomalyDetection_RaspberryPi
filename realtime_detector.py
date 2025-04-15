# realtime_detector.py
import time
import board
import adafruit_dht
import RPi.GPIO as GPIO
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import os

# Load config
with open("config.json") as f:
    CONFIG = json.load(f)

# Pins and paths from config
DHT_PIN = CONFIG["GPIO"]["DHT_PIN"]
PIR_PIN = CONFIG["GPIO"]["PIR_PIN"]
BUZZER_PIN = CONFIG["GPIO"]["BUZZER_PIN"]
LED_PIN = CONFIG["GPIO"]["LED_PIN"]
MODEL_PATH = CONFIG["MODEL"]["model_path"]
ANOMALY_LOG = CONFIG["LOGGING"]["anomaly_log_file"]
INTERVAL = CONFIG["LOGGING"]["interval_sec"]
ROLLING = CONFIG["MODEL"]["rolling_window"]

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIR_PIN, GPIO.IN)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(LED_PIN, GPIO.OUT)
dht_sensor = adafruit_dht.DHT11(board.D4)

# Load model and scaler
model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
scaler = model_data["scaler"]

# Make sure log directory exists
os.makedirs(os.path.dirname(ANOMALY_LOG), exist_ok=True)

# Create anomaly CSV if not exists
if not os.path.isfile(ANOMALY_LOG):
    with open(ANOMALY_LOG, mode='w') as f:
        f.write("Timestamp,Temperature,Humidity,Motion,Prediction\n")

# Live buffer
data_buffer = []
print("🔍 Starting real-time anomaly detection...\n")

try:
    while True:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        motion = GPIO.input(PIR_PIN)

        try:
            time.sleep(1)  # Small delay to ensure sensor stability
            temp = dht_sensor.temperature
            hum = dht_sensor.humidity
            if temp is None or hum is None:
                raise ValueError("Invalid DHT reading")
        except Exception as e:
            print(f"[WARN] Sensor read error: {e}")
            time.sleep(INTERVAL)
            continue

        # Update buffer
        data_buffer.append([temp, hum, motion])
        if len(data_buffer) > ROLLING:
            data_buffer.pop(0)

        if len(data_buffer) < ROLLING:
            print(f"[{timestamp}] ⏳ Waiting for enough data...")
        else:
            df = pd.DataFrame(data_buffer, columns=["Temp", "Humidity", "Motion"])
            mean_row = df.mean().to_frame().T  # DataFrame with column names
            scaled_input = scaler.transform(mean_row)
            pred = model.predict(scaled_input)[0]
            status = "🚨 Anomaly" if pred == -1 else "✅ Normal"

            # Show result
            print(f"[{timestamp}] Temp: {temp}°C | Humidity: {hum}% | Motion: {motion} → {status}")

            # Trigger alerts
            if pred == -1:
                if CONFIG["ALERTS"]["use_buzzer"]:
                    GPIO.output(BUZZER_PIN, GPIO.HIGH)
                if CONFIG["ALERTS"]["use_led"]:
                    GPIO.output(LED_PIN, GPIO.HIGH)
            else:
                GPIO.output(BUZZER_PIN, GPIO.LOW)
                GPIO.output(LED_PIN, GPIO.LOW)

            # Log anomaly
            with open(ANOMALY_LOG, mode='a') as f:
                f.write(f"{timestamp},{temp},{hum},{motion},{pred}\n")

        time.sleep(INTERVAL)

except KeyboardInterrupt:
    print("\n🛑 Detection stopped by user.")

finally:
    dht_sensor.exit()
    GPIO.cleanup()
