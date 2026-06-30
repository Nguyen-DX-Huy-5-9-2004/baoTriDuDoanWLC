import pandas as pd
import numpy as np
import os

base_dir = r"C:\Users\huynd1\Downloads\tBTDD"
processed_dir = os.path.join(base_dir, "data", "processed")

files = ["laser_final_features.csv", "press_brake_final_features.csv", 
         "punching_final_features.csv", "welding_robot_final_features.csv"]

for f in files:
    path = os.path.join(processed_dir, f)
    if not os.path.exists(path):
        print(f"❌ Không tìm thấy: {f}")
        continue
        
    df = pd.read_csv(path)
    print(f"\n{'='*90}")
    print(f"FILE: {f} | Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Timestamp
    time_cols = [c for c in df.columns if 'time' in c.lower() or 'timestamp' in c.lower()]
    if time_cols:
        time_col = time_cols[0]
        df[time_col] = pd.to_datetime(df[time_col])
        print(f"Timestamp range: {df[time_col].min()} -> {df[time_col].max()}")
        print(f"Median gap: {df[time_col].diff().median()}")
    
    # Target columns
    target_cols = [c for c in df.columns if any(k in c.lower() for k in ['failure', 'wear', 'valve', 'leak', 'target', 'risk'])]
    print(f"Potential target columns: {target_cols}")
    
    for t in target_cols:
        print(f"  {t}: {df[t].value_counts().to_dict()}")
    
    # Top correlation
    numeric = df.select_dtypes(include=[np.number])
    if target_cols:
        first_target = target_cols[0]
        if first_target in numeric.columns:
            corr = numeric.corr()[first_target].abs().sort_values(ascending=False)
            print("Top 8 correlation with first target:")
            print(corr.head(8))
    
    print(f"{'='*90}")