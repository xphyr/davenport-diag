from flask import Flask, render_template, jsonify
import onnxruntime as ort
import numpy as np
import random
import time

app = Flask(__name__)

# Load the model
onnx_path = '../models/saved/rf_engine_warning.onnx'
try:
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
except Exception as e:
    print(f"Warning: Could not load ONNX model: {e}")
    session = None

# Simulator state
sim_state = {
    'rpm': 2000.0,
    'load': 40.0,
    'speed': 60.0,
    'oat': 25.0,
    'buffer': [] # To calculate rolling stats
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream')
def stream():
    # Random walk the parameters to simulate driving
    sim_state['speed'] += random.uniform(-2, 2)
    sim_state['speed'] = max(0, min(140, sim_state['speed']))
    
    sim_state['rpm'] = (sim_state['speed'] * 30) + random.uniform(-100, 100)
    if sim_state['speed'] < 5:
        sim_state['rpm'] = 800 + random.uniform(-50, 50)
        
    sim_state['load'] += random.uniform(-5, 5)
    sim_state['load'] = max(10, min(100, sim_state['load']))
    
    # Randomly inject an anomaly
    if random.random() < 0.05:
        sim_state['load'] = 98.0
        sim_state['rpm'] = 6000.0
        
    # Update buffer (keep last 60 samples for ~60 seconds if polling at 1Hz)
    sim_state['buffer'].append({
        'rpm': sim_state['rpm'],
        'load': sim_state['load'],
        'speed': sim_state['speed']
    })
    if len(sim_state['buffer']) > 60:
        sim_state['buffer'].pop(0)
        
    # Calculate rolling features
    rpm_vals = [s['rpm'] for s in sim_state['buffer']]
    load_vals = [s['load'] for s in sim_state['buffer']]
    speed_vals = [s['speed'] for s in sim_state['buffer']]
    
    rpm_mean = sum(rpm_vals) / len(rpm_vals)
    rpm_max = max(rpm_vals)
    load_mean = sum(load_vals) / len(load_vals)
    load_max = max(load_vals)
    speed_mean = sum(speed_vals) / len(speed_vals)
            
    # Predict using ONNX
    prediction = 0
    if session:
        # Features: RPM_mean, RPM_max, Load_mean, Load_max, Speed_mean, OAT
        X = np.array([[
            rpm_mean,
            rpm_max,
            load_mean,
            load_max,
            speed_mean,
            sim_state['oat']
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
        'latency_ms': round(latency, 2)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
