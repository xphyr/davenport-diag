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
    # Features: 'Engine_RPM_RPM', 'Absolute_Load_pct', 'Vehicle_Speed_km_per_h', 'OAT_DegC', 'log_MAF'
    # Dummy data: [RPM, Load, Speed, OAT, log_MAF]
    dummy_data = np.array([
        [2000.0, 40.0, 60.0, 25.0, 3.5], # Normal
        [5000.0, 95.0, 120.0, 30.0, 5.0], # High Load
    ], dtype=np.float32)
    
    start_time = time.time()
    pred_onx = session.run([label_name], {input_name: dummy_data})[0]
    end_time = time.time()
    
    latency_ms = (end_time - start_time) * 1000 / len(dummy_data)
    
    print(f"Predictions: {pred_onx}")
    print(f"Average Inference Latency: {latency_ms:.4f} ms per sample")

if __name__ == "__main__":
    run_inference()
