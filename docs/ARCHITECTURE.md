# Engine Failure Diagnostics - Architecture

## Overview
This system is an AI-powered edge application designed to predict and warn about engine failures in real-time, based on driving characteristics. The system was developed using the VED (Vehicle Emissions Dataset).

## Components

### 1. Data Ingestion & Preprocessing
- **Source**: VED Refined 20M Engine Records dataset.
- **Preprocessing**: Calculates Total Fuel Trim (STFT + LTFT) over a rolling 30-second window. Identifies periods of sustained rich/lean operation (> 15% absolute total trim) to define a ground-truth "Engine Warning" state.

### 2. Edge AI Model
- **Model Type**: Random Forest Classifier
- **Features**: `Engine_RPM_RPM`, `Absolute_Load_pct`, `Vehicle_Speed_km_per_h`, `OAT_DegC`, `log_MAF`.
- **Target**: `Engine_Warning`
- **Format**: Converted to ONNX format for high-speed inference on edge devices (e.g., NVIDIA Jetson).

### 3. Web Simulator
- **Backend**: Flask application (`simulator/app.py`). Serves the web interface and handles the ONNX inference session.
- **Frontend**: A modern, glassmorphic dashboard (`simulator/templates/index.html`) using HTML/CSS/Vanilla JS. Polling is used to request real-time data from the backend, simulating a continuous stream of edge telemetry.
