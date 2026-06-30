import os
import sys
import time
import random
import json
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "microsoft-azure-predictive-maintenance")
DB_URL = "postgresql://postgres:s@localhost:5432/predictive_maintenance"


def start_dynamic_factory_feed():
    print("=" * 80)
    print("KHỞI ĐỘNG TRẠM IOT - MÔ PHỎNG THỜI GIAN THỰC HIỆN TẠI")
    print("=" * 80)

    print("\nTÙY CHỈNH TỐC ĐỘ MÔ PHỎNG DEMO:")
    print("1. Chạy như thực tế (1 giờ dữ liệu = Đợi 1 giờ thực)")
    print("2. Demo Nhanh       (1 giờ dữ liệu = Đợi 2 giây)")
    print("3. Demo Siêu Tốc    (1 giờ dữ liệu = Đợi 0.2 giây)")

    choice = input("chọn tốc độ (1/2/3) [Mặc định: 2]: ").strip()
    if choice == '1':
        sleep_time = 3600
    elif choice == '3':
        sleep_time = 0.2
    else:
        sleep_time = 2.0

    engine = create_engine(DB_URL)

    # Load training data
    failures_df = pd.read_csv(os.path.join(DATA_DIR, "PdM_failures.csv"), parse_dates=['datetime'])
    telemetry_df = pd.read_csv(os.path.join(DATA_DIR, "PdM_telemetry.csv"), parse_dates=['datetime'])

    # Get list of machines from DB
    with engine.connect() as conn:
        machines_df = pd.read_sql("SELECT machine_id FROM machines ORDER BY machine_id", conn)
        machine_ids = machines_df['machine_id'].tolist()

    if not machine_ids:
        print("Lỗi: Không tìm thấy máy nào trong database!")
        return

    print(f"Sử dụng bảng telemetry_stream từ TimescaleDB với {len(machine_ids)} máy")

    # Step 1: Pick a random failure event
    random_fail_event = failures_df.sample(1).iloc[0]
    original_fail_machine_id = int(random_fail_event['machineID'])
    original_fail_time = random_fail_event['datetime']
    failure_component = random_fail_event['failure']

    # Step 2: Assign to a real machine in our DB
    target_machine_id = random.choice(machine_ids)

    # Step 3: Get data from 50h before failure up to failure (for the failing machine)
    start_time_old = original_fail_time - pd.Timedelta(hours=60)
    fail_machine_telemetry = telemetry_df[
        (telemetry_df['machineID'] == original_fail_machine_id) &
        (telemetry_df['datetime'] >= start_time_old) &
        (telemetry_df['datetime'] <= original_fail_time)
    ].copy()

    # Step 4: Shift timestamps to current time
    now = pd.Timestamp(datetime.now()).floor('h')
    time_shift = now - original_fail_time
    fail_machine_telemetry['original_datetime'] = fail_machine_telemetry['datetime']
    fail_machine_telemetry['datetime'] = fail_machine_telemetry['datetime'] + time_shift
    fail_time_new = original_fail_time + time_shift

    # Step 5: Prepare data for normal machines
    normal_machine_ids = [mid for mid in machine_ids if mid != target_machine_id]
    
    # For each normal machine, pick a random 48h period from a random healthy machine
    normal_machine_data = {}
    all_normal_machines_in_dataset = telemetry_df['machineID'].unique()
    
    for normal_mid in normal_machine_ids:
        # Pick a random healthy machine from dataset (not the failing one)
        healthy_mid = random.choice([m for m in all_normal_machines_in_dataset if m != original_fail_machine_id])
        
        # Pick a random 50h window from this healthy machine
        healthy_times = telemetry_df[telemetry_df['machineID'] == healthy_mid]['datetime'].sort_values()
        if len(healthy_times) < 51:  # Need at least 51 points for 50h
            continue
            
        start_idx = random.randint(0, len(healthy_times) - 51)
        normal_start_time = healthy_times.iloc[start_idx]
        normal_end_time = healthy_times.iloc[start_idx + 50]
        
        normal_telemetry = telemetry_df[
            (telemetry_df['machineID'] == healthy_mid) & 
            (telemetry_df['datetime'] >= normal_start_time) & 
            (telemetry_df['datetime'] <= normal_end_time)
        ].copy()
        
        normal_telemetry['original_datetime'] = normal_telemetry['datetime']
        normal_telemetry['datetime'] = normal_telemetry['datetime'] + time_shift
        
        normal_machine_data[normal_mid] = normal_telemetry

    print("\n" + "*" * 70)
    print(f"KỊCH BẢN CA LÀM VIỆC (QUY MÔ DEMO: {len(machine_ids)} MÁY):")
    print(f" 🚨 Lỗi cao (Hỏng thật): Máy #{target_machine_id} (Hỏng lúc {fail_time_new.strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"     Thành phần hỏng: {failure_component}")
    print(f"     Dữ liệu gốc từ: Máy #{original_fail_machine_id} (ngày {original_fail_time.strftime('%Y-%m-%d %H:%M:%S')})")
    print(f" ✅ Khỏe mạnh (Ổn định): {len(normal_machine_ids)} Máy còn lại")
    print(f" ⚡ Tốc độ mô phỏng: {'Thực tế' if sleep_time == 3600 else 'Nhanh' if sleep_time == 2.0 else 'Siêu tốc'}")
    print("*" * 70 + "\n")

    # Step 6: Prepare data for each timestamp (group by datetime)
    fail_machine_grouped = fail_machine_telemetry.sort_values('datetime').groupby('datetime')

    # Tính tổng số bước để chia tỷ lệ suy thoái
    total_steps = len(fail_machine_grouped)

    for idx, (current_dt, fail_machine_data) in enumerate(fail_machine_grouped):
        telemetry_records = []

        # TÍNH TOÁN HỆ SỐ SUY THOÁI TỪ TỪ (DRIFT MULTIPLIER)
        # progress chạy từ 0.0 đến 1.0. Dùng mũ 2 để tạo đường cong cong dần lên (Lúc đầu hỏng chậm, về cuối hỏng nhanh)
        progress = idx / max(1, (total_steps - 1))
        drift_multiplier = progress ** 2 

        # 1. ADD DATA FOR FAILING MACHINE (Ép lỗi tăng dần từng chút một)
        if not fail_machine_data.empty:
            row = fail_machine_data.iloc[0]
            
            # Cài đặt biên độ lỗi tối đa
            max_vibration_add = 20.0
            max_pressure_add = 30.0
            max_rotate_drop = 40.0

            sensor_payload = {
                # Volt dao động nhẹ tự nhiên
                'volt': float(row['volt']) + random.gauss(0, 2.5),
                
                # Rotate giảm dần từ từ + nhiễu
                'rotate': float(row['rotate']) - (drift_multiplier * max_rotate_drop) + random.gauss(0, 3.5),
                
                # Pressure & Vibration tăng dần từ từ + nhiễu
                'pressure': float(row['pressure']) + (drift_multiplier * max_pressure_add) + random.gauss(0, 3.5),
                'vibration': float(row['vibration']) + (drift_multiplier * max_vibration_add) + random.gauss(0, 2.0),
                
                'original_datetime': row['original_datetime'].isoformat(),
                'original_machine_id': original_fail_machine_id
            }
            telemetry_records.append({
                'time': current_dt,
                'machine_id': target_machine_id,
                'sensor_payload': sensor_payload
            })

        # 2. ADD DATA FOR NORMAL MACHINES (Bơm nhiễu dao động tự nhiên)
        for normal_mid in normal_machine_ids:
            if normal_mid not in normal_machine_data:
                continue
                
            normal_data = normal_machine_data[normal_mid]
            if idx < len(normal_data):
                normal_row = normal_data.iloc[idx]
                
                # Dùng random.gauss(mean=0, std) để tạo dao động tự nhiên cho máy khỏe
                sensor_payload = {
                    'volt': float(normal_row['volt']) + random.gauss(0, 1.0),
                    'rotate': float(normal_row['rotate']) + random.gauss(0, 1.5),
                    'pressure': float(normal_row['pressure']) + random.gauss(0, 1.5),
                    'vibration': float(normal_row['vibration']) + random.gauss(0, 0.5),
                    'original_datetime': normal_row['original_datetime'].isoformat(),
                    'original_machine_id': int(normal_row['machineID'])
                }
                telemetry_records.append({
                    'time': current_dt,
                    'machine_id': normal_mid,
                    'sensor_payload': sensor_payload
                })

        # Convert sensor_payload dict to JSON string for PostgreSQL
        for record in telemetry_records:
            record['sensor_payload'] = json.dumps(record['sensor_payload'])

        # Insert into telemetry_stream
        df_telemetry = pd.DataFrame(telemetry_records)
        df_telemetry.to_sql('telemetry_stream', engine, if_exists='append', index=False)

        # Update machine live metrics
        for rec in telemetry_records:
            load_pct = random.uniform(30, 95)
            op_level = 1 if load_pct < 50 else (2 if load_pct < 80 else 3)
            
            with engine.connect() as conn:
                result = conn.execute(text("""
                    UPDATE machine_live_metrics
                    SET load_percentage = :load, operation_level = :level, updated_at = NOW()
                    WHERE machine_id = :mid
                """), {'mid': rec['machine_id'], 'load': load_pct, 'level': op_level})

                if result.rowcount == 0:
                    conn.execute(text("""
                        INSERT INTO machine_live_metrics (machine_id, load_percentage, operation_level, updated_at)
                        VALUES (:mid, :load, :level, NOW())
                    """), {'mid': rec['machine_id'], 'load': load_pct, 'level': op_level})
                conn.commit()

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Đã đồng bộ mốc {current_dt.strftime('%Y-%m-%d %H:%M')} | Tiến trình: {int(progress*100)}%")
        time.sleep(sleep_time)

    print(f"\nCA LÀM VIỆC KẾT THÚC! MÁY #{target_machine_id} ĐÃ HỎNG VÀO LÚC {fail_time_new.strftime('%Y-%m-%d %H:%M:%S')}!")


if __name__ == "__main__":
    start_dynamic_factory_feed()
