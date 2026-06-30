import pandas as pd
import numpy as np
import os
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

class WeldcomFinalCleanPipeline:
    def __init__(self, base_dir):
        self.processed_dir = os.path.join(base_dir, "data", "processed")
        self.output_dir = os.path.join(base_dir, "data", "processed_final_xgboost_ready")
        os.makedirs(self.output_dir, exist_ok=True)

    def fix_and_save(self, machine_name, file_name):
        print(f"\n🔧 Đang xử lý {machine_name}...")
        df = pd.read_csv(os.path.join(self.processed_dir, file_name))
        
        # Timestamp & sort
        if 'timestamp' not in df.columns:
            freq_map = {"laser": "1h", "punching": "15min", "press_brake": "1min", "robot": "15min"}
            freq = freq_map.get(machine_name, "1min")
            df['timestamp'] = pd.date_range('2024-01-01', periods=len(df), freq=freq)
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        # Target
        if machine_name == "laser":
            df['target_risk'] = ((df.get('target_lens_failure', 0) > 0) | (df.get('target_gantry_failure', 0) > 0)).astype(int)
        elif machine_name == "press_brake":
            df['target_risk'] = ((df['target_valve_condition'] != 100) | 
                               (df.get('target_pump_leakage', 0) > 0) | 
                               (df.get('target_accumulator_state', 130) != 130)).astype(int)
        elif machine_name == "punching":
            df['target_risk'] = pd.to_numeric(df.get('target_wear', 0), errors='coerce').fillna(0).astype(int)
        elif machine_name == "robot":
            df['target_risk'] = pd.to_numeric(df.get('target_failure_24h', 0), errors='coerce').fillna(0).astype(int)
        
        # Rolling
        feature_map = {
            "laser": ['lens_temperature_c', 'laser_source_temp_c', 'xy_axis_torque_nm'],
            "press_brake": ['ps1_mean', 'ps2_mean', 'ts1_mean', 'fs1_mean'],
            "punching": ['CF_Feature_1', 'CF_Feature_2', 'Vib_Feature_1', 'Vib_Feature_4'],
            "robot": ['robot_vibration_g', 'servo_motor_temp_c', 'robot_torque_proxy_amp']
        }
        for col in feature_map.get(machine_name, []):
            if col in df.columns:
                df[f'{col}_roll_std_6'] = df[col].rolling(6, min_periods=3).std().fillna(0)
        
        # Drop leakage
        leakage = ['days_since', 'hours_since', 'tool_hits_count', 'z_axis_tracking_error_proxy']
        df = df.drop(columns=[c for c in df.columns if any(k in c for k in leakage)], errors='ignore')
        
        df.to_csv(os.path.join(self.output_dir, f"{machine_name}_xgboost_ready.csv"), index=False)
        print(f"✅ {machine_name} → Shape {df.shape} | Target {df['target_risk'].value_counts().to_dict()}")

    def run(self):
        machines = {
            "laser": "laser_final_features.csv",
            "press_brake": "press_brake_final_features.csv",
            "punching": "punching_final_features.csv",
            "robot": "welding_robot_final_features.csv"
        }
        for m, fname in machines.items():
            self.fix_and_save(m, fname)
        print(f"\n🎉 HOÀN THÀNH vFinal-8!")

if __name__ == "__main__":
    BASE = r"C:\Users\huynd1\Downloads\tBTDD"
    WeldcomFinalCleanPipeline(BASE).run()