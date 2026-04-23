import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

def load_processed_data(path='data/processed_data.parquet'):
    print(f"Loading {path}...")
    df = pd.read_parquet(path)
    return df

def train_model():
    df = load_processed_data()
    
    # Feature Engineering: Use driving characteristics to predict the warning state
    features = [
        'RPM_rolling_mean', 
        'RPM_rolling_max', 
        'Load_rolling_mean', 
        'Load_rolling_max', 
        'Speed_rolling_mean',
        'OAT_DegC'
    ]
    target = 'Engine_Warning'
    
    print(f"Using features: {features}")
    
    # Drop rows with NaN in features or target
    df_clean = df[features + [target]].dropna()
    print(f"Data shape after dropping NaNs: {df_clean.shape}")
    
    X = df_clean[features]
    y = df_clean[target]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    print("Training Random Forest Classifier...")
    
    # Using a shallow Random Forest for fast edge inference and training
    clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
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
