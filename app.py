# -*- coding: utf-8 -*-
"""
IoT Elevator Predictive Maintenance System
Flask REST API Server
Serves real-time predictions, simulated sensor data, and the dashboard
"""

from flask import Flask, jsonify, render_template, request
import pickle
import numpy as np
import time
import random
import os
from datetime import datetime
from collections import deque

app = Flask(__name__)

# -----------------------------------------
# Load Model
# -----------------------------------------
MODEL_PATH = 'model/xgb_model.pkl'
SCALER_PATH = 'model/scaler.pkl'
META_PATH   = 'model/metadata.pkl'

if not os.path.exists(MODEL_PATH):
    print("[ERROR] Model not found! Run 'python generate_data.py' then 'python train_model.py' first.")
    exit(1)

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
with open(META_PATH, 'rb') as f:
    metadata = pickle.load(f)

FEATURES    = metadata['features']
CLASS_NAMES = metadata['classes']
FAULT_COLORS = ['#22c55e', '#ef4444', '#f97316', '#3b82f6']
FAULT_ICONS  = ['[OK]', '[GEAR]', '[HOT]', '[DOOR]']

# Frontend-friendly icons (returned in JSON, rendered as HTML in JS)
FAULT_ICONS_HTML = ['\u2705', '\u2699\ufe0f', '\U0001f321\ufe0f', '\U0001f6aa']

print(f"[OK] Model loaded | Accuracy: {metadata['accuracy']*100:.2f}%")

# -----------------------------------------
# In-memory history
# -----------------------------------------
MAX_HISTORY   = 200
history       = deque(maxlen=MAX_HISTORY)
alert_log     = deque(maxlen=50)
total_readings = 0
total_faults   = 0
start_time     = time.time()

# -----------------------------------------
# Sensor Simulation
# -----------------------------------------
SIMULATION_STATES = [0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 0]
sim_index = [0]

def generate_sensor_reading(force_class=None):
    """Generate a realistic sensor reading for a given fault class."""
    if force_class is None:
        cls = SIMULATION_STATES[sim_index[0] % len(SIMULATION_STATES)]
        sim_index[0] += 1
    else:
        cls = force_class

    noise = lambda v, pct=0.05: v + random.gauss(0, abs(v) * pct + 0.01)

    if cls == 0:  # Normal
        reading = {
            'vibration_rms':          noise(random.uniform(0.5, 2.0)),
            'vibration_kurtosis':     noise(random.uniform(2.5, 3.5)),
            'vibration_skewness':     noise(random.uniform(-0.3, 0.3)),
            'temperature_mean':       noise(random.uniform(30, 50)),
            'temperature_slope':      noise(random.uniform(-0.1, 0.1)),
            'motor_current_mean':     noise(random.uniform(10, 15)),
            'motor_current_variance': noise(random.uniform(0.1, 1.0)),
            'door_signal_variance':   noise(random.uniform(0.1, 0.5)),
            'acceleration_peak':      noise(random.uniform(0.2, 1.0)),
        }
    elif cls == 1:  # Bearing Failure
        reading = {
            'vibration_rms':          noise(random.uniform(5.0, 12.0)),
            'vibration_kurtosis':     noise(random.uniform(6.0, 15.0)),
            'vibration_skewness':     noise(random.uniform(1.5, 4.0)),
            'temperature_mean':       noise(random.uniform(45, 65)),
            'temperature_slope':      noise(random.uniform(0.05, 0.3)),
            'motor_current_mean':     noise(random.uniform(14, 20)),
            'motor_current_variance': noise(random.uniform(1.5, 4.0)),
            'door_signal_variance':   noise(random.uniform(0.1, 0.6)),
            'acceleration_peak':      noise(random.uniform(3.0, 8.0)),
        }
    elif cls == 2:  # Motor Overheating
        reading = {
            'vibration_rms':          noise(random.uniform(2.0, 5.0)),
            'vibration_kurtosis':     noise(random.uniform(3.0, 5.0)),
            'vibration_skewness':     noise(random.uniform(0.0, 0.5)),
            'temperature_mean':       noise(random.uniform(75, 110)),
            'temperature_slope':      noise(random.uniform(0.5, 2.0)),
            'motor_current_mean':     noise(random.uniform(20, 35)),
            'motor_current_variance': noise(random.uniform(3.0, 8.0)),
            'door_signal_variance':   noise(random.uniform(0.1, 0.5)),
            'acceleration_peak':      noise(random.uniform(0.5, 2.0)),
        }
    else:  # Door Malfunction
        reading = {
            'vibration_rms':          noise(random.uniform(1.0, 3.5)),
            'vibration_kurtosis':     noise(random.uniform(2.5, 4.0)),
            'vibration_skewness':     noise(random.uniform(-0.5, 0.5)),
            'temperature_mean':       noise(random.uniform(35, 55)),
            'temperature_slope':      noise(random.uniform(-0.1, 0.2)),
            'motor_current_mean':     noise(random.uniform(12, 18)),
            'motor_current_variance': noise(random.uniform(2.0, 5.0)),
            'door_signal_variance':   noise(random.uniform(3.0, 10.0)),
            'acceleration_peak':      noise(random.uniform(0.3, 1.5)),
        }

    reading['true_class'] = cls
    return reading

def run_prediction(reading):
    """Run model prediction on a sensor reading dict."""
    feat_vec    = np.array([[reading[f] for f in FEATURES]])
    feat_scaled = scaler.transform(feat_vec)
    pred_class  = int(model.predict(feat_scaled)[0])
    probs       = model.predict_proba(feat_scaled)[0].tolist()
    confidence  = float(max(probs))
    return pred_class, probs, confidence

# -----------------------------------------
# API Routes
# -----------------------------------------

@app.route('/')
def index():
    return render_template('index.html',
                           accuracy=f"{metadata['accuracy']*100:.1f}",
                           n_samples=metadata['n_samples'])

@app.route('/live-data', methods=['GET'])
def live_data():
    """Returns one simulated sensor reading + prediction."""
    global total_readings, total_faults

    reading = generate_sensor_reading()
    pred_class, probs, confidence = run_prediction(reading)

    total_readings += 1
    is_fault = pred_class != 0
    if is_fault:
        total_faults += 1

    timestamp = datetime.now().strftime('%H:%M:%S')

    record = {
        'timestamp': timestamp,
        'sensors': {
            'vibration_rms':          round(reading['vibration_rms'], 3),
            'vibration_kurtosis':     round(reading['vibration_kurtosis'], 3),
            'vibration_skewness':     round(reading['vibration_skewness'], 3),
            'temperature_mean':       round(reading['temperature_mean'], 2),
            'temperature_slope':      round(reading['temperature_slope'], 4),
            'motor_current_mean':     round(reading['motor_current_mean'], 2),
            'motor_current_variance': round(reading['motor_current_variance'], 3),
            'door_signal_variance':   round(reading['door_signal_variance'], 3),
            'acceleration_peak':      round(reading['acceleration_peak'], 3),
        },
        'prediction': {
            'class_id':     pred_class,
            'class_name':   CLASS_NAMES[pred_class],
            'color':        FAULT_COLORS[pred_class],
            'icon':         FAULT_ICONS_HTML[pred_class],
            'confidence':   round(confidence * 100, 1),
            'probabilities': {CLASS_NAMES[i]: round(probs[i] * 100, 1) for i in range(4)},
        },
        'is_fault': is_fault,
    }

    history.append(record)

    if is_fault:
        alert_log.appendleft({
            'timestamp':  timestamp,
            'fault':      CLASS_NAMES[pred_class],
            'confidence': round(confidence * 100, 1),
            'color':      FAULT_COLORS[pred_class],
            'icon':       FAULT_ICONS_HTML[pred_class],
        })

    return jsonify(record)

@app.route('/predict', methods=['POST'])
def predict():
    """Accepts JSON sensor data, returns prediction."""
    data = request.json
    try:
        reading = {f: float(data.get(f, 0)) for f in FEATURES}
        pred_class, probs, confidence = run_prediction(reading)
        return jsonify({
            'success':    True,
            'class_id':   pred_class,
            'class_name': CLASS_NAMES[pred_class],
            'confidence': round(confidence * 100, 1),
            'probabilities': {CLASS_NAMES[i]: round(probs[i] * 100, 1) for i in range(4)},
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/history', methods=['GET'])
def get_history():
    """Returns last N readings from history."""
    n = min(int(request.args.get('n', 60)), MAX_HISTORY)
    h = list(history)[-n:]
    return jsonify({'data': h, 'count': len(h)})

@app.route('/alerts', methods=['GET'])
def get_alerts():
    """Returns recent fault alerts."""
    return jsonify({'alerts': list(alert_log)})

@app.route('/stats', methods=['GET'])
def get_stats():
    """Returns system-wide statistics."""
    uptime_seconds = int(time.time() - start_time)
    hours   = uptime_seconds // 3600
    minutes = (uptime_seconds % 3600) // 60
    seconds = uptime_seconds % 60
    health  = max(0, 100 - (total_faults / max(total_readings, 1)) * 100 * 3)

    return jsonify({
        'total_readings': total_readings,
        'total_faults':   total_faults,
        'fault_rate':     round(total_faults / max(total_readings, 1) * 100, 1),
        'health_score':   round(health, 1),
        'uptime':         f'{hours:02d}:{minutes:02d}:{seconds:02d}',
        'model_accuracy': round(metadata['accuracy'] * 100, 1),
        'alerts_count':   len(alert_log)
    })

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  IoT Elevator Predictive Maintenance System")
    print("  Flask API Server Starting...")
    print("=" * 60)
    print(f"\n  Dashboard URL: http://127.0.0.1:5000")
    print(f"  Model Accuracy: {metadata['accuracy']*100:.2f}%")
    print(f"  Fault Classes: {', '.join(CLASS_NAMES)}")
    print("\n  Press Ctrl+C to stop\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
