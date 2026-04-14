# -*- coding: utf-8 -*-
"""
IoT Elevator Predictive Maintenance System
XGBoost Model Training Script
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb

print("=" * 60)
print("  IoT Elevator Predictive Maintenance - Model Training")
print("=" * 60)

# Load dataset
print("\n[+] Loading dataset...")
df = pd.read_csv('data/sensor_data.csv')
print(f"   Rows: {len(df)}, Features: {df.shape[1]-1}")

FEATURES = [
    'vibration_rms', 'vibration_kurtosis', 'vibration_skewness',
    'temperature_mean', 'temperature_slope',
    'motor_current_mean', 'motor_current_variance',
    'door_signal_variance', 'acceleration_peak'
]
TARGET = 'fault_class'
class_names = ['Normal', 'Bearing Failure', 'Motor Overheating', 'Door Malfunction']

X = df[FEATURES].values
y = df[TARGET].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"   Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# Feature scaling
print("\n[+] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train XGBoost
print("\n[+] Training XGBoost classifier...")
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    num_class=4,
    eval_metric='mlogloss',
    random_state=42,
)
model.fit(X_train_scaled, y_train,
          eval_set=[(X_test_scaled, y_test)],
          verbose=False)
print("   Training complete!")

# Evaluate
y_pred = model.predict(X_test_scaled)
acc    = accuracy_score(y_test, y_pred)
print(f"\n[+] Evaluation Results:")
print(f"   Accuracy: {acc*100:.2f}%")
print("\n   Classification Report:")
report = classification_report(y_test, y_pred, target_names=class_names)
print(report)
print("   Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"   {cm}")

# Save model, scaler, metadata FIRST
os.makedirs('model', exist_ok=True)
with open('model/xgb_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

metadata = {
    'features':  FEATURES,
    'classes':   class_names,
    'accuracy':  float(acc),
    'n_samples': len(df)
}
with open('model/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("\n[OK] Model saved  -> model/xgb_model.pkl")
print("[OK] Scaler saved -> model/scaler.pkl")
print("[OK] Metadata     -> model/metadata.pkl")

# Feature importances (ASCII only)
print("\n[+] Feature Importances:")
importances = model.feature_importances_
for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
    bar = "#" * int(imp * 40)
    print(f"   {feat:<30} {bar} {imp:.4f}")

print(f"\n[**] Final Accuracy: {acc*100:.2f}%")
print("=" * 60)
print("\nRun: python app.py   -> to start the web dashboard!")
