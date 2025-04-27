#!/usr/bin/env python3
# realtime_detector.py

import time
import board
import adafruit_dht
import RPi.GPIO as GPIO
import pandas as pd
import joblib
import json
from datetime import datetime
import os

# Load config
with open("config.json") as f:
    CONFIG = json.load(f)

# Pins and paths from config
DHT_PIN      = CONFIG["GPIO"]["DHT_PIN"]
PIR_PIN      = CONFIG["GPIO"]["PIR_PIN"]
BUZZER_PIN   = CONFIG["GPIO"]["BUZZER_PIN"]
LED_PIN      = CONFIG["GPIO"]["LED_PIN"]
MODEL_PATH   = CONFIG["MODEL"]["model_path"]
ANOMALY_LOG  = CONFIG["LOGGING"]["anomaly_log_file"]
INTERVAL     = CONFIG["LOGGING"]["interval_sec"]
ROLLING      = CONFIG["MODEL"]["rolling_window"]

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIR_PIN, GPIO.IN)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(LED_PIN, GPIO.OUT)
dht_sensor = adafruit_dht.DHT11(board.D4)

# Load model + scaler from the single pickle
model_data = joblib.load(MODEL_PATH)
model       = model_data["model"]
scaler      = model_data["scaler"]

# Ensure anomaly log directory exists
os.makedirs(os.path.dirname(ANOMALY_LOG), exist_ok=True)
if not os.path.isfile(ANOMALY_LOG):
    with open(ANOMALY_LOG, 'w') as f:
        f.write("Timestamp,Temperature,Humidity,Motion,Prediction\n")

data_buffer = []
print("🔍 Starting real-time anomaly detection...\n")

try:
    while True:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        motion = GPIO.input(PIR_PIN)

        # Read DHT sensor (with a quick retry if needed)
        try:
            time.sleep(1)
            temp = dht_sensor.temperature
            hum  = dht_sensor.humidity
            if temp is None or hum is None:
                raise ValueError("Invalid DHT reading")
        except Exception as e:
            print(f"[WARN] Sensor read error: {e}")
            time.sleep(INTERVAL)
            continue

        # Append to rolling buffer
        data_buffer.append({'Temp': temp, 'Humidity': hum, 'Motion': motion})

        # Wait until we have enough data
        if len(data_buffer) < ROLLING:
            print(f"[{timestamp}] ⏳ Waiting for enough data...")
            time.sleep(INTERVAL)
            continue

        # Compute rolling means
        df = pd.DataFrame(data_buffer[-ROLLING:])
        avg_temp     = df['Temp'].mean()
        avg_humidity = df['Humidity'].mean()
        avg_motion   = df['Motion'].mean()

        # Build a DataFrame for prediction
        mean_row = pd.DataFrame([{
            'Temp':      avg_temp,
            'Humidity':  avg_humidity,
            'Motion':    avg_motion
        }])

        # Rename Temp → Temperature to match training
        mean_row = mean_row.rename(columns={'Temp': 'Temperature'})
        # Reorder columns exactly as in training
        mean_row = mean_row[['Temperature', 'Humidity', 'Motion']]

        # Scale and predict
        scaled_input = scaler.transform(mean_row)
        pred = model.predict(scaled_input)[0]

        # Act on the prediction
        if pred == -1:
            print(f"[{timestamp}] 🚨 Anomaly detected! Temp={avg_temp:.1f}°C Humidity={avg_humidity:.0f}% Motion={avg_motion}")
            GPIO.output(LED_PIN, GPIO.HIGH)
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            time.sleep(0.5)
            GPIO.output(BUZZER_PIN, GPIO.LOW)
        else:
            print(f"[{timestamp}] ✅ Normal. Temp={avg_temp:.1f}°C Humidity={avg_humidity:.0f}% Motion={avg_motion}")
            GPIO.output(LED_PIN, GPIO.LOW)

        # Log to CSV
        with open(ANOMALY_LOG, 'a') as f:
            f.write(f"{timestamp},{avg_temp:.1f},{avg_humidity:.0f},{avg_motion},{int(pred)}\n")

        time.sleep(INTERVAL)

except KeyboardInterrupt:
    print("🛑 Stopped by user.")
finally:
    GPIO.cleanup()
