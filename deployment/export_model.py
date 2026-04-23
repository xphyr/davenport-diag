import joblib
import os
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def export_model():
    model_path = 'models/saved/rf_engine_warning.joblib'
    print(f"Loading trained model from {model_path}...")
    clf = joblib.load(model_path)
    
    # 10 input features for Phase 2
    initial_type = [('float_input', FloatTensorType([None, 10]))]
    
    print("Converting model to ONNX format...")
    onnx_model = convert_sklearn(clf, initial_types=initial_type)
    
    onnx_path = 'models/saved/rf_engine_warning.onnx'
    print(f"Saving ONNX model to {onnx_path}...")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
        
    print("Export complete.")

if __name__ == "__main__":
    export_model()
