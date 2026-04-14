# -*- coding: utf-8 -*-

"""
IoT Elevator Predictive Maintenance System
Synthetic Sensor Dataset Generator
"""

import numpy as np
import pandas as pd
import os

# Fix randomness

np.random.seed(42)

# Total samples

N = 5000
data = []

# Class distribution

class_distribution = {
0: int(N * 0.55),   # Normal
1: int(N * 0.15),   # Bearing Failure
2: int(N * 0.15),   # Motor Overheating
3: int(N * 0.15),   # Door Malfunction
}

# Noise function

def add_noise(val, scale=0.05):
    return val + np.random.normal(0, abs(val) * scale + 1e-6)

# -------- CLASS 0: NORMAL --------

for _ in range(class_distribution[0]):
    row = [
        add_noise(np.random.uniform(0.5, 2.0)),   # vibration_rms
        add_noise(np.random.uniform(2.5, 3.5)),   # kurtosis
        add_noise(np.random.uniform(-0.3, 0.3)),  # skewness
        add_noise(np.random.uniform(30, 50)),     # temp_mean
        add_noise(np.random.uniform(-0.1, 0.1)),  # temp_slope
        add_noise(np.random.uniform(10, 15)),     # current_mean
        add_noise(np.random.uniform(0.1, 1.0)),   # current_var
        add_noise(np.random.uniform(0.1, 0.5)),   # door_var
        add_noise(np.random.uniform(0.2, 1.0)),   # accel_peak
        0
    ]
    data.append(row)

# -------- CLASS 1: BEARING FAILURE --------

for _ in range(class_distribution[1]):
    row = [
        add_noise(np.random.uniform(5.0, 12.0)),
        add_noise(np.random.uniform(6.0, 15.0)),
        add_noise(np.random.uniform(1.5, 4.0)),
        add_noise(np.random.uniform(45, 65)),
        add_noise(np.random.uniform(0.05, 0.3)),
        add_noise(np.random.uniform(14, 20)),
        add_noise(np.random.uniform(1.5, 4.0)),
        add_noise(np.random.uniform(0.1, 0.6)),
        add_noise(np.random.uniform(3.0, 8.0)),
        1
    ]
    data.append(row)

# -------- CLASS 2: MOTOR OVERHEATING --------

for _ in range(class_distribution[2]):
    row = [
        add_noise(np.random.uniform(2.0, 5.0)),
        add_noise(np.random.uniform(3.0, 5.0)),
        add_noise(np.random.uniform(0.0, 0.5)),
        add_noise(np.random.uniform(75, 110)),
        add_noise(np.random.uniform(0.5, 2.0)),
        add_noise(np.random.uniform(20, 35)),
        add_noise(np.random.uniform(3.0, 8.0)),
        add_noise(np.random.uniform(0.1, 0.5)),
        add_noise(np.random.uniform(0.5, 2.0)),
        2
    ]
    data.append(row)

# -------- CLASS 3: DOOR MALFUNCTION --------

for _ in range(class_distribution[3]):
    row = [
        add_noise(np.random.uniform(1.0, 3.5)),
        add_noise(np.random.uniform(2.5, 4.0)),
        add_noise(np.random.uniform(-0.5, 0.5)),
        add_noise(np.random.uniform(35, 55)),
        add_noise(np.random.uniform(-0.1, 0.2)),
        add_noise(np.random.uniform(12, 18)),
        add_noise(np.random.uniform(2.0, 5.0)),
        add_noise(np.random.uniform(3.0, 10.0)),
        add_noise(np.random.uniform(0.3, 1.5)),
        3
    ]
    data.append(row)

# Column names

columns = [
'vibration_rms',
'vibration_kurtosis',
'vibration_skewness',
'temperature_mean',
'temperature_slope',
'motor_current_mean',
'motor_current_variance',
'door_signal_variance',
'acceleration_peak',
'fault_class'
]

# Create DataFrame

df = pd.DataFrame(data, columns=columns)

# Shuffle data

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Create folder and save

os.makedirs('data', exist_ok=True)
df.to_csv('data/sensor_data.csv', index=False)

# Print output

print(f"[OK] Dataset generated: {len(df)} rows")
print("Saved to: data/sensor_data.csv")
