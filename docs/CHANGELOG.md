# Changelog

## [1.0.0] - 2026-04-22
### Added
- Project structure created (`data/`, `models/`, `deployment/`, `notebooks/`, `docs/`, `simulator/`).
- Data preprocessing script (`data/preprocess.py`) to label engine warnings based on sustained rich/lean fuel trims.
- Model training script (`models/train.py`) to train a Random Forest classifier using driving behavior characteristics.
- Deployment script (`deployment/export_model.py`) to convert the trained model to ONNX.
- Edge inference simulation (`deployment/edge_inference.py`) to benchmark the model.
- Flask-based web simulator with a dynamic dashboard to demonstrate real-time edge prediction capabilities.
- Documentation (`ARCHITECTURE.md` and `CHANGELOG.md`).
