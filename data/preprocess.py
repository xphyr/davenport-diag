import pandas as pd
import glob
import os
import argparse
import numpy as np

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
    
    # Calculate Total Fuel Trim for Bank 1 and Bank 2
    df['Total_Fuel_Trim_Bank_1'] = df['Short_Term_Fuel_Trim_Bank_1_pct'] + df['Long_Term_Fuel_Trim_Bank_1_pct']
    df['Total_Fuel_Trim_Bank_2'] = df['Short_Term_Fuel_Trim_Bank_2_pct'] + df['Long_Term_Fuel_Trim_Bank_2_pct']
    
    # Set Timestampms as index to use rolling by time
    df['Time_Sec'] = pd.to_timedelta(df['Timestampms'], unit='ms')
    df = df.set_index('Time_Sec')
    
    print("Calculating rolling trims and imbalances...")
    grouped = df.groupby(['VehId', 'Trip'])
    
    # 30 second rolling window for trims
    df['Rolling_Total_Trim_B1'] = grouped['Total_Fuel_Trim_Bank_1'].rolling('30s').mean().reset_index(level=[0,1], drop=True)
    df['Rolling_Total_Trim_B2'] = grouped['Total_Fuel_Trim_Bank_2'].rolling('30s').mean().reset_index(level=[0,1], drop=True)
    
    # Bank Imbalance: Difference between Bank 1 and Bank 2
    df['Bank_Imbalance'] = (df['Rolling_Total_Trim_B1'] - df['Rolling_Total_Trim_B2']).abs()
    
    print("Calculating advanced driving features...")
    # 60 second rolling features for driving style
    df['RPM_rolling_mean'] = grouped['Engine_RPM_RPM'].rolling('60s').mean().reset_index(level=[0,1], drop=True)
    df['RPM_rolling_max'] = grouped['Engine_RPM_RPM'].rolling('60s').max().reset_index(level=[0,1], drop=True)
    df['Load_rolling_mean'] = grouped['Absolute_Load_pct'].rolling('60s').mean().reset_index(level=[0,1], drop=True)
    df['Load_rolling_max'] = grouped['Absolute_Load_pct'].rolling('60s').max().reset_index(level=[0,1], drop=True)
    df['Speed_rolling_mean'] = grouped['Vehicle_Speed_km_per_h'].rolling('60s').mean().reset_index(level=[0,1], drop=True)
    
    # Instantaneous Deltas (within trip)
    df['RPM_delta'] = grouped['Engine_RPM_RPM'].diff().fillna(0)
    df['Load_delta'] = grouped['Absolute_Load_pct'].diff().fillna(0)
    
    # Derived Features
    df['Stress_Index'] = df['RPM_rolling_mean'] * df['Load_rolling_mean']
    
    # Gear Ratio Proxy: RPM / Speed (Avoid division by zero)
    df['Gear_Ratio_Proxy'] = df['Engine_RPM_RPM'] / df['Vehicle_Speed_km_per_h'].replace(0, np.nan)
    df['Gear_Ratio_Proxy'] = df['Gear_Ratio_Proxy'].fillna(0)
    
    df = df.reset_index()
    
    # Define "Failure/Warning" condition:
    # 1. Individual bank trim > 15%
    # 2. Bank imbalance > 10% (suggests bank-specific component failure)
    TRIM_THRESHOLD = 15.0
    IMBALANCE_THRESHOLD = 10.0
    
    df['Engine_Warning'] = 0
    # Bank 1 anomalies
    df.loc[df['Rolling_Total_Trim_B1'] > TRIM_THRESHOLD, 'Engine_Warning'] = 1
    df.loc[df['Rolling_Total_Trim_B1'] < -TRIM_THRESHOLD, 'Engine_Warning'] = 1
    # Bank 2 anomalies
    df.loc[df['Rolling_Total_Trim_B2'] > TRIM_THRESHOLD, 'Engine_Warning'] = 1
    df.loc[df['Rolling_Total_Trim_B2'] < -TRIM_THRESHOLD, 'Engine_Warning'] = 1
    # Bank Imbalance anomalies
    df.loc[df['Bank_Imbalance'] > IMBALANCE_THRESHOLD, 'Engine_Warning'] = 1
    
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
