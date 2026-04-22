import pandas as pd
import glob
import sys

def inspect():
    files = glob.glob('data/content/ved_phase_1/*.parquet')
    if not files:
        print("No parquet files found.")
        sys.exit(1)
        
    file_path = files[0]
    print(f"Reading file: {file_path}")
    df = pd.read_parquet(file_path)
    
    print("\n--- Columns ---")
    for col in df.columns:
        print(col)
        
    print("\n--- Data Head (5 rows) ---")
    print(df.head())
    
    # Check for STFT / LTFT
    fuel_cols = [c for c in df.columns if 'trim' in c.lower() or 'stft' in c.lower() or 'ltft' in c.lower()]
    print(f"\n--- Fuel Trim Columns found: {fuel_cols} ---")
    
    if fuel_cols:
        print(df[fuel_cols].describe())
    else:
        print("No Fuel Trim columns matched the simple search.")

if __name__ == "__main__":
    inspect()
