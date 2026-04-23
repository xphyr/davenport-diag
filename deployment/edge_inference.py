import onnxruntime as ort
import numpy as np
import time

def run_inference():
    onnx_path = 'models/saved/rf_engine_warning.onnx'
    print(f"Loading ONNX model from {onnx_path}...")
    session = ort.InferenceSession(onnx_path)
    
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
    
    print("Simulating edge inference on dummy data...")
    # Features: 'RPM_rolling_mean', 'RPM_rolling_max', 'Load_rolling_mean', 'Load_rolling_max', 'Speed_rolling_mean', 'OAT_DegC'
    # Dummy data: [RPM_mean, RPM_max, Load_mean, Load_max, Speed_mean, OAT]
    dummy_data = np.array([
        [2000.0, 2200.0, 40.0, 45.0, 60.0, 25.0], # Normal
        [5000.0, 5500.0, 95.0, 98.0, 120.0, 30.0], # Aggressive/High Load
    ], dtype=np.float32)
    
    start_time = time.time()
    pred_onx = session.run([label_name], {input_name: dummy_data})[0]
    end_time = time.time()
    
    latency_ms = (end_time - start_time) * 1000 / len(dummy_data)
    
    print(f"Predictions: {pred_onx}")
    print(f"Average Inference Latency: {latency_ms:.4f} ms per sample")

if __name__ == "__main__":
    run_inference()
