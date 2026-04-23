import pandas as pd
import glob
import os
import argparse

def load_data(data_dir='data/content/ved_phase_1', max_files=2):
    """Load a subset of parquet files for development/testing."""
    files = glob.glob(os.path.join(data_dir, '*.parquet'))
    if max_files:
        files = files[:max_files]
        
    dfs = []
    for f in files:
        print(f"Loading {f}...")
        df = pd.read_parquet(f)
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

def preprocess(df):
    """
    Clean data and calculate fuel trim metrics.
    Engine running rich: Negative fuel trims (ECU is pulling fuel).
    Engine running lean: Positive fuel trims (ECU is adding fuel).
    """
    print(f"Original shape: {df.shape}")
    
    # Sort by vehicle, trip, and time
    df = df.sort_values(by=['VehId', 'Trip', 'Timestampms']).reset_index(drop=True)
    
    # Calculate Total Fuel Trim for Bank 1 (and Bank 2 if available, though Bank 1 is primary)
    df['Total_Fuel_Trim_Bank_1'] = df['Short_Term_Fuel_Trim_Bank_1_pct'] + df['Long_Term_Fuel_Trim_Bank_1_pct']
    
    # Calculate a rolling average of Total Fuel Trim to detect "extended periods"
    # Assuming data is recorded frequently, we use a rolling window per trip
    # E.g., 60 seconds rolling window. Timestampms is in milliseconds.
    
    # Let's create a simpler rolling mean by row count first, or by timestamp.
    # We will group by Trip and apply rolling
    
    # Set Timestampms as index to use rolling by time
    df['Time_Sec'] = pd.to_timedelta(df['Timestampms'], unit='ms')
    df = df.set_index('Time_Sec')
    
    print("Calculating rolling trims...")
    # 30 second rolling window
    rolling_trim = df.groupby(['VehId', 'Trip'])['Total_Fuel_Trim_Bank_1'].rolling('30s').mean()
    # rolling() with time window keeps the group keys in the index.
    # we can drop them and map back to the dataframe
    df['Rolling_Total_Trim_B1'] = rolling_trim.reset_index(level=[0,1], drop=True)
    
    print("Calculating rolling features (60s window)...")
    # 60 second rolling features for driving style
    grouped = df.groupby(['VehId', 'Trip'])
    
    df['RPM_rolling_mean'] = grouped['Engine_RPM_RPM'].rolling('60s').mean().reset_index(level=[0,1], drop=True)
    df['RPM_rolling_max'] = grouped['Engine_RPM_RPM'].rolling('60s').max().reset_index(level=[0,1], drop=True)
    
    df['Load_rolling_mean'] = grouped['Absolute_Load_pct'].rolling('60s').mean().reset_index(level=[0,1], drop=True)
    df['Load_rolling_max'] = grouped['Absolute_Load_pct'].rolling('60s').max().reset_index(level=[0,1], drop=True)
    
    df['Speed_rolling_mean'] = grouped['Vehicle_Speed_km_per_h'].rolling('60s').mean().reset_index(level=[0,1], drop=True)
    
    df = df.reset_index()
    
    # Define "Failure/Warning" condition:
    # Industry standard for check engine light is usually > +20% (lean) or < -20% (rich)
    # We'll set a warning threshold at +/- 15% for extended periods
    TRIM_THRESHOLD = 15.0
    
    df['Engine_Warning'] = 0
    df.loc[df['Rolling_Total_Trim_B1'] > TRIM_THRESHOLD, 'Engine_Warning'] = 1 # Lean
    df.loc[df['Rolling_Total_Trim_B1'] < -TRIM_THRESHOLD, 'Engine_Warning'] = 1 # Rich
    
    print(f"Number of warning states identified: {df['Engine_Warning'].sum()} out of {len(df)} records")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Preprocess VED dataset with rolling window features.')
    parser.add_argument('--max_files', type=int, default=2, help='Maximum number of files to load (default: 2)')
    args = parser.parse_args()

    df = load_data(max_files=args.max_files)
    df_clean = preprocess(df)
    
    out_path = 'data/processed_data.parquet'
    print(f"Saving processed data to {out_path}...")
    df_clean.to_parquet(out_path, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
