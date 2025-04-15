# sensor_logger.py
import time
import board
import adafruit_dht
import RPi.GPIO as GPIO
import csv
import json
import os
from datetime import datetime

# Load config
with open("config.json") as f:
    CONFIG = json.load(f)

DHT_PIN = CONFIG["GPIO"]["DHT_PIN"]
PIR_PIN = CONFIG["GPIO"]["PIR_PIN"]
LOG_FILE = CONFIG["LOGGING"]["log_file"]
INTERVAL = CONFIG["LOGGING"]["interval_sec"]

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIR_PIN, GPIO.IN)

dht_sensor = adafruit_dht.DHT11(board.D4)

# Ensure data folder exists
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Initialize CSV
if not os.path.isfile(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Temperature", "Humidity", "Motion"])

print("📊 Logging sensor data... Press CTRL+C to stop.")

try:
    while True:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        motion = GPIO.input(PIR_PIN)

        try:
            temperature = dht_sensor.temperature
            humidity = dht_sensor.humidity
        except RuntimeError as e:
            print(f"[WARN] DHT Read Error: {e.args[0]}")
            temperature, humidity = None, None

        with open(LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, temperature, humidity, motion])

        print(f"[{timestamp}] Temp: {temperature}°C | Humidity: {humidity}% | Motion: {motion}")
        time.sleep(INTERVAL)

except KeyboardInterrupt:
    print("\n🛑 Logging stopped by user.")

finally:
    dht_sensor.exit()
    GPIO.cleanup()
