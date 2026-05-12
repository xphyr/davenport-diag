from flask import Flask, render_template, jsonify, request
import onnxruntime as ort
import numpy as np
import pandas as pd
import random
import time
import os
import joblib

# Must match models/train.py isolation forest inputs (order + names).
ISO_FEATURE_NAMES = (
    'Rolling_Total_Trim_B1',
    'Rolling_Total_Trim_B2',
    'Bank_Imbalance',
)

app = Flask(__name__)

# Load models
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)
onnx_path = os.path.join(project_root, 'models/saved/rf_engine_warning.onnx')
iso_path = os.path.join(project_root, 'models/saved/isolation_forest.joblib')

session = None
input_name = None
label_name = None
iso_forest = None

try:
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
except Exception as e:
    print(f"Warning: Could not load ONNX model: {e}")

try:
    iso_forest = joblib.load(iso_path)
except Exception as e:
    print(f"Warning: Could not load isolation forest ({iso_path}): {e}")

# Simulator state
sim_state = {
    'rpm': 800.0,
    'load': 20.0,
    'speed': 0.0,
    'oat': 25.0,
    'throttle': 0.0,
    'brake': 0.0,
    'ext_load': 20.0,
    'buffer': []
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/control', methods=['POST'])
def control():
    data = request.json
    if 'throttle' in data:
        sim_state['throttle'] = float(data['throttle'])
    if 'brake' in data:
        sim_state['brake'] = float(data['brake'])
    if 'ext_load' in data:
        sim_state['ext_load'] = float(data['ext_load'])
    return jsonify({'status': 'ok'})

@app.route('/stream')
def stream():
    # Simple physics simulation
    # Acceleration based on throttle, load, and braking
    power = 4.0 * sim_state['throttle']
    friction = 0.5 + (sim_state['ext_load'] / 20.0) # Base friction + weight
    braking = 8.0 * sim_state['brake']
    
    # Drag increases with speed
    drag = (sim_state['speed'] ** 2) * 0.005
    
    acceleration = power - friction - braking - drag
    sim_state['speed'] += acceleration
    sim_state['speed'] = max(0, min(160, sim_state['speed']))
    
    # RPM logic: Idle is 800. Revs up with throttle and speed.
    idle_rpm = 800
    if sim_state['speed'] < 1:
        sim_state['rpm'] = idle_rpm + (sim_state['throttle'] * 5000)
    else:
        # Simple linear gear proxy
        sim_state['rpm'] = (sim_state['speed'] * 35) + (sim_state['throttle'] * 1500)
    
    sim_state['rpm'] = max(idle_rpm, min(7000, sim_state['rpm']))
    
    # Load is influenced by throttle and the weight dial
    sim_state['load'] = (sim_state['throttle'] * 70) + (sim_state['ext_load'] / 2) + random.uniform(-2, 2)
    sim_state['load'] = max(10, min(100, sim_state['load']))
        
    # Update buffer (keep last 60 samples for ~60 seconds if polling at 1Hz)
    sim_state['buffer'].append({
        'rpm': sim_state['rpm'],
        'load': sim_state['load'],
        'speed': sim_state['speed']
    })
    if len(sim_state['buffer']) > 60:
        sim_state['buffer'].pop(0)
        
    # Simulate fuel trims for the isolation forest
    trim_b1 = random.uniform(-2, 2) + (sim_state['load'] / 10)
    trim_b2 = random.uniform(-2, 2) + (sim_state['load'] / 10)
    if sim_state['load'] > 90:
        trim_b1 += random.uniform(5, 15)
        trim_b2 += random.uniform(5, 15)
    
    imbalance = abs(trim_b1 - trim_b2)
        
    # Calculate rolling features
    rpm_vals = [s['rpm'] for s in sim_state['buffer']]
    load_vals = [s['load'] for s in sim_state['buffer']]
    speed_vals = [s['speed'] for s in sim_state['buffer']]
    
    rpm_mean = sum(rpm_vals) / len(rpm_vals)
    rpm_max = max(rpm_vals)
    load_mean = sum(load_vals) / len(load_vals)
    load_max = max(load_vals)
    speed_mean = sum(speed_vals) / len(speed_vals)
    
    # Advanced Phase 2 Features
    rpm_delta = sim_state['rpm'] - sim_state['buffer'][-2]['rpm'] if len(sim_state['buffer']) > 1 else 0
    load_delta = sim_state['load'] - sim_state['buffer'][-2]['load'] if len(sim_state['buffer']) > 1 else 0
    stress_index = rpm_mean * load_mean
    gear_ratio = sim_state['rpm'] / sim_state['speed'] if sim_state['speed'] > 0 else 0
            
    # Unsupervised Anomaly Score
    if iso_forest:
        # decision_function returns scores where lower is more anomalous
        iso_df = pd.DataFrame(
            [[trim_b1, trim_b2, imbalance]],
            columns=ISO_FEATURE_NAMES,
        )
        anomaly_score = float(iso_forest.decision_function(iso_df)[0])
    else:
        anomaly_score = 0
            
    # Predict using ONNX
    prediction = 0
    if session:
        # Features: RPM_mean, RPM_max, Load_mean, Load_max, Speed_mean, OAT, 
        #           RPM_delta, Load_delta, Stress_Index, Gear_Ratio, Anomaly_Score
        X = np.array([[
            rpm_mean,
            rpm_max,
            load_mean,
            load_max,
            speed_mean,
            sim_state['oat'],
            rpm_delta,
            load_delta,
            stress_index,
            gear_ratio,
            anomaly_score
        ]], dtype=np.float32)
        
        start_time = time.time()
        pred = session.run([label_name], {input_name: X})[0]
        latency = (time.time() - start_time) * 1000
        prediction = int(pred[0])
    else:
        latency = 0.0

    return jsonify({
        'rpm': round(sim_state['rpm'], 2),
        'load': round(sim_state['load'], 2),
        'speed': round(sim_state['speed'], 2),
        'oat': round(sim_state['oat'], 2),
        'prediction': prediction,
        'latency_ms': round(latency, 2),
        'is_imbalanced': imbalance > 10,
        'anomaly_score': round(float(anomaly_score), 4)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
