import joblib
import os
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxmltools

def export_to_onnx(model_path='models/saved/rf_engine_warning.joblib', onnx_path='models/saved/rf_engine_warning.onnx'):
    print(f"Loading trained model from {model_path}...")
    clf = joblib.load(model_path)
    
    # 11 input features for Phase 3/4
    initial_type = [('float_input', FloatTensorType([None, 11]))]
    
    print("Converting model to ONNX format (using onnxmltools for LightGBM)...")
    # onnxmltools handles LightGBM native objects better than skl2onnx
    onnx_model = onnxmltools.convert_lightgbm(clf, initial_types=initial_type, target_opset=13)
    
    print(f"Saving ONNX model to {onnx_path}...")
    onnxmltools.utils.save_model(onnx_model, onnx_path)
    print("Export complete.")

if __name__ == "__main__":
    export_to_onnx()
