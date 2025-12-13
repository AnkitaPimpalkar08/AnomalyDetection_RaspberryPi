#!/usr/bin/env python3
"""
Trains Isolation Forest model with proper train/test split and evaluation metrics.
Saves model, scaler, and performance metrics for verification.
Author: Ankita vilas Pimpalkar
Date: 2024
"""

import numpy as np
import pandas as pd
import os
import json
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load configuration
with open("config.json") as f:
    CONFIG = json.load(f)

LOG_FILE = CONFIG["LOGGING"]["log_file"]
MODEL_PATH = CONFIG["MODEL"]["model_path"]
ROLLING = CONFIG["MODEL"]["rolling_window"]
CONTAM = CONFIG["MODEL"]["contamination"]

# Create models directory if it doesn't exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

print("="*60)
print("Isolation Forest Model Training with Evaluation")
print("="*60)

# Load sensor data
if not os.path.exists(LOG_FILE):
    raise FileNotFoundError(f"Sensor log file not found: {LOG_FILE}")

df = pd.read_csv(LOG_FILE)
print(f"\nLoaded {len(df)} records from {LOG_FILE}")

# Data cleaning
initial_count = len(df)
df.dropna(inplace=True)
cleaned_count = len(df)
print(f"Removed {initial_count - cleaned_count} records with missing values")

if cleaned_count < ROLLING:
    raise ValueError(f"Insufficient data: need at least {ROLLING} records, have {cleaned_count}")

# Apply rolling average for temporal consistency
print(f"\nApplying {ROLLING}-sample rolling window...")
features = df[["Temperature", "Humidity", "Motion"]].rolling(ROLLING).mean().dropna()
print(f"Features after rolling window: {len(features)} records")

# Split data into train and test sets (80/20 split)
print("\nSplitting data into train/test sets...")
X_train, X_test = train_test_split(features, test_size=0.2, random_state=42, shuffle=False)
print(f"Training set: {len(X_train)} records")
print(f"Test set: {len(X_test)} records")

# Standardize features using training data
print("\nStandardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Feature means: {scaler.mean_}")
print(f"Feature std devs: {scaler.scale_}")

# Train Isolation Forest model
print(f"\nTraining Isolation Forest model...")
print(f"  Contamination: {CONTAM}")
print(f"  Random state: 42")
model = IsolationForest(contamination=CONTAM, random_state=42, n_estimators=100)
model.fit(X_train_scaled)

# Make predictions on both sets
print("\nGenerating predictions...")
train_predictions = model.predict(X_train_scaled)
test_predictions = model.predict(X_test_scaled)

# Convert predictions to binary (1 = normal, -1 = anomaly)
# For evaluation, we treat -1 (anomaly) as positive class
train_labels = (train_predictions == -1).astype(int)
test_labels = (test_predictions == -1).astype(int)

# Calculate metrics
print("\n" + "="*60)
print("MODEL EVALUATION RESULTS")
print("="*60)

# Training set metrics
train_anomaly_count = train_labels.sum()
train_anomaly_rate = (train_anomaly_count / len(train_labels)) * 100
print(f"\nTraining Set:")
print(f"  Total samples: {len(train_labels)}")
print(f"  Anomalies detected: {train_anomaly_count}")
print(f"  Anomaly rate: {train_anomaly_rate:.2f}%")

# Test set metrics
test_anomaly_count = test_labels.sum()
test_anomaly_rate = (test_anomaly_count / len(test_labels)) * 100
print(f"\nTest Set:")
print(f"  Total samples: {len(test_labels)}")
print(f"  Anomalies detected: {test_anomaly_count}")
print(f"  Anomaly rate: {test_anomaly_rate:.2f}%")

# Since Isolation Forest is unsupervised, we create synthetic ground truth
# by labeling the most extreme deviations as true anomalies
def create_synthetic_labels(X, percentile=90):
    """
    Create synthetic anomaly labels based on statistical outliers.
    This simulates ground truth for evaluation purposes.
    """
    # Calculate Mahalanobis-like distance for each sample
    mean = X.mean(axis=0)
    cov = np.cov(X.T)
    inv_cov = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-6)
    
    distances = []
    for i in range(len(X)):
        diff = X[i] - mean
        distance = np.sqrt(diff @ inv_cov @ diff.T)
        distances.append(distance)
    
    # Label top percentile as anomalies
    threshold = np.percentile(distances, percentile)
    labels = (np.array(distances) > threshold).astype(int)
    return labels

# Create synthetic ground truth for evaluation
print("\nCreating synthetic ground truth for evaluation...")
synthetic_train_labels = create_synthetic_labels(X_train_scaled, percentile=90)
synthetic_test_labels = create_synthetic_labels(X_test_scaled, percentile=90)

# Calculate confusion matrix and metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("\n" + "-"*60)
print("Performance Metrics (using synthetic ground truth)")
print("-"*60)

# Test set performance
accuracy = accuracy_score(synthetic_test_labels, test_labels)
precision = precision_score(synthetic_test_labels, test_labels, zero_division=0)
recall = recall_score(synthetic_test_labels, test_labels, zero_division=0)
f1 = f1_score(synthetic_test_labels, test_labels, zero_division=0)

print(f"\nTest Set Performance:")
print(f"  Accuracy:  {accuracy*100:.2f}%")
print(f"  Precision: {precision*100:.2f}%")
print(f"  Recall:    {recall*100:.2f}%")
print(f"  F1-Score:  {f1*100:.2f}%")

# Confusion matrix
cm = confusion_matrix(synthetic_test_labels, test_labels)
print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {cm[0,0]}")
print(f"  False Positives: {cm[0,1]}")
print(f"  False Negatives: {cm[1,0]}")
print(f"  True Positives:  {cm[1,1]}")

# Calculate false positive rate
tn, fp, fn, tp = cm.ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
print(f"  False Positive Rate: {fpr*100:.2f}%")

# Save model and scaler
print("\n" + "="*60)
print("Saving model and metrics...")
joblib.dump({"model": model, "scaler": scaler}, MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")

# Save evaluation metrics
metrics = {
    "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "training_samples": len(X_train),
    "test_samples": len(X_test),
    "rolling_window": ROLLING,
    "contamination": CONTAM,
    "test_metrics": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "false_positive_rate": float(fpr)
    },
    "train_anomaly_rate": float(train_anomaly_rate),
    "test_anomaly_rate": float(test_anomaly_rate),
    "confusion_matrix": {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }
}

metrics_file = MODEL_PATH.replace('.pkl', '_metrics.json')
with open(metrics_file, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved to: {metrics_file}")

print("\n" + "="*60)
print("Training completed successfully!")
print("="*60)
