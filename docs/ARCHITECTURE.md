# Engine Failure Diagnostics - Architecture

## Overview
This system is an AI-powered edge application designed to predict and warn about engine failures in real-time, based on driving characteristics. The system was developed using the VED (Vehicle Emissions Dataset).

## Components

### 1. Data Ingestion & Preprocessing
- **Source**: VED Refined 20M Engine Records dataset.
- **Preprocessing**: Calculates Total Fuel Trim (STFT + LTFT) over a rolling 30-second window.
- **Feature Engineering**: Generates **Rolling Style Features** (Mean and Max of RPM, Load, and Speed over a 60-second window) to capture temporal driving context.

### 2. Edge AI Model
- **Model Type**: Random Forest Classifier
- **Features**: `RPM_rolling_mean`, `RPM_rolling_max`, `Load_rolling_mean`, `Load_rolling_max`, `Speed_rolling_mean`, `OAT_DegC`.
- **Target**: `Engine_Warning` (Derived from Fuel Trim anomalies).
- **Format**: Converted to ONNX format for high-speed inference on edge devices (e.g., NVIDIA Jetson).

### 3. Web Simulator
- **Backend**: Flask application (`simulator/app.py`). Serves the web interface and handles the ONNX inference session.
- **Real-time Analytics**: Maintains a 60-point rolling buffer in-memory to calculate the style features required by the ONNX model before each prediction.
- **Frontend**: A modern, glassmorphic dashboard (`simulator/templates/index.html`) using HTML/CSS/Vanilla JS.
