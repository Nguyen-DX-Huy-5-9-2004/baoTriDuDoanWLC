import pandas as pd
from sqlalchemy import create_engine, text
import time

# 1. Cấu hình kết nối (Thay 'admin' bằng mật khẩu postgres thực tế của bạn)
DB_URI = "postgresql://postgres:s@localhost:5432/pe_maintenance"
engine = create_engine(DB_URI)

# 2. Đường dẫn thư mục dữ liệu gốc của bạn
DATA_DIR = r"C:\Users\huynd1\Downloads\tBTDD\datasets"

def load_ai4i_data():
    print("Đang xử lý tập dữ liệu chẩn đoán AI4I...")
    file_path = f"{DATA_DIR}\\predictive-maintenance-dataset-ai4i-2020\\ai4i2020.csv"
    df = pd.read_csv(file_path)
    
    # Chuẩn hóa tên cột để tương thích tốt với PostgreSQL (viết thường, bỏ khoảng trắng)
    df.columns = [c.lower().replace(' ', '_').replace('[', '').replace(']', '') for c in df.columns]
    
    # Bơm thẳng vào DB, nếu bảng đã có thì ghi đè (replace)
    df.to_sql('mechanical_features', engine, if_exists='replace', index=False)
    print(f"Đã nạp thành công {len(df)} dòng vào bảng 'mechanical_features'.")

def load_iot_telemetry():
    print("\nĐang xử lý tập dữ liệu IoT Telemetry (Sẽ mất chút thời gian)...")
    file_path = f"{DATA_DIR}\\iot-integrated-predictive-maintenance-dataset\\predictive_maintenance_dataset.csv"
    df = pd.read_csv(file_path)
    
    # Ép kiểu cột thời gian cho chuẩn xác
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Nạp vào PostgreSQL
    df.to_sql('machine_telemetry', engine, if_exists='replace', index=False)
    
    # Kích hoạt siêu năng lực của TimescaleDB: Biến bảng thường thành Hypertable
    try:
        with engine.connect() as conn:
            # Lệnh biến bảng thành hypertable, tự migrate data đã có
            query = text("SELECT create_hypertable('machine_telemetry', 'timestamp', migrate_data := true);")
            conn.execute(query)
            conn.commit()
        print(f"Đã nạp {len(df)} dòng và cấu hình Hypertable cho 'machine_telemetry' thành công!")
    except Exception as e:
        print(f"Hypertable có thể đã được tạo từ trước. Chi tiết: {e}")

if __name__ == "__main__":
    start_time = time.time()
    
    print("=== BẮT ĐẦU NẠP DỮ LIỆU VÀO DATABASE ===")
    load_ai4i_data()
    load_iot_telemetry()
    
    end_time = time.time()
    print(f"\nHoàn tất toàn bộ trong {round(end_time - start_time, 2)} giây!")