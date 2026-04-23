import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

def load_processed_data(path='data/processed_data.parquet'):
    print(f"Loading {path}...")
    df = pd.read_parquet(path)
    return df

def train_model():
    df = load_processed_data()
    
    # Unsupervised Anomaly Detection: isolation forest on trims
    print("Training Isolation Forest for unsupervised anomaly detection...")
    iso_features = ['Rolling_Total_Trim_B1', 'Rolling_Total_Trim_B2', 'Bank_Imbalance']
    iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    
    # We fit on a subset to keep training time reasonable
    iso_sample = df[iso_features].dropna().sample(min(500000, len(df)), random_state=42)
    iso_forest.fit(iso_sample)
    
    # Add Anomaly Score as a new feature
    print("Generating anomaly scores...")
    df['Anomaly_Score'] = iso_forest.decision_function(df[iso_features].fillna(0))
    
    # Feature Engineering: Use driving characteristics + anomaly score to predict warning
    features = [
        'RPM_rolling_mean', 
        'RPM_rolling_max', 
        'Load_rolling_mean', 
        'Load_rolling_max', 
        'Speed_rolling_mean',
        'OAT_DegC',
        'RPM_delta',
        'Load_delta',
        'Stress_Index',
        'Gear_Ratio_Proxy',
        'Anomaly_Score'
    ]
    target = 'Engine_Warning'
    
    print(f"Using features: {features}")
    
    # Drop rows with NaN in features or target
    df_clean = df[features + [target]].dropna()
    print(f"Data shape after dropping NaNs: {df_clean.shape}")
    
    X = df_clean[features]
    y = df_clean[target]
    
    # Save the isolation forest for deployment
    joblib.dump(iso_forest, 'models/saved/isolation_forest.joblib')
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    print("Training LightGBM Classifier...")
    
    # Using LightGBM for better precision and fast inference
    clf = LGBMClassifier(
        n_estimators=100,
        max_depth=7,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    clf.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    os.makedirs('models/saved', exist_ok=True)
    model_path = 'models/saved/rf_engine_warning.joblib'
    print(f"Saving model to {model_path}...")
    joblib.dump(clf, model_path)
    print("Done.")

if __name__ == "__main__":
    train_model()
