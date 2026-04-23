import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import joblib
import os

def run_anomaly_detection(data_path='data/processed_data.parquet', sample_size=100000):
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Sample data for speed if necessary
    if len(df) > sample_size:
        print(f"Sampling {sample_size} records for anomaly detection...")
        df_sample = df.sample(sample_size, random_state=42).reset_index(drop=True)
    else:
        df_sample = df
        
    # Features to look for anomalies in
    # We focus on the fuel trims themselves to find "weird" engine behavior
    anomaly_features = [
        'Rolling_Total_Trim_B1',
        'Rolling_Total_Trim_B2',
        'Bank_Imbalance'
    ]
    
    X = df_sample[anomaly_features].fillna(0)
    
    print("Training Isolation Forest...")
    # contamination=0.05 means we expect about 5% of data to be anomalous
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
    
    # Fit and predict (-1 for anomaly, 1 for normal)
    df_sample['Anomaly_Score'] = iso_forest.fit_predict(X)
    df_sample['Is_Unsupervised_Anomaly'] = (df_sample['Anomaly_Score'] == -1).astype(int)
    
    # Analyze results
    print("\n--- Anomaly Detection Results ---")
    num_anomalies = df_sample['Is_Unsupervised_Anomaly'].sum()
    print(f"Anomalies detected: {num_anomalies} ({num_anomalies/len(df_sample)*100:.2f}%)")
    
    # Check overlap with our rule-based Engine_Warning
    overlap = df_sample[(df_sample['Is_Unsupervised_Anomaly'] == 1) & (df_sample['Engine_Warning'] == 1)]
    print(f"Overlap with rule-based warnings: {len(overlap)} ({len(overlap)/num_anomalies*100:.2f}% of unsupervised anomalies)")
    
    new_anomalies = df_sample[(df_sample['Is_Unsupervised_Anomaly'] == 1) & (df_sample['Engine_Warning'] == 0)]
    print(f"New anomalies identified (not caught by rules): {len(new_anomalies)}")
    
    if len(new_anomalies) > 0:
        print("\nExample of 'New' anomalies (threshold was < 15% but model flagged as weird):")
        print(new_anomalies[anomaly_features].describe())
    
    # Save the model
    os.makedirs('models/saved', exist_ok=True)
    joblib.dump(iso_forest, 'models/saved/isolation_forest.joblib')
    print("\nModel saved to models/saved/isolation_forest.joblib")

if __name__ == "__main__":
    run_anomaly_detection()
