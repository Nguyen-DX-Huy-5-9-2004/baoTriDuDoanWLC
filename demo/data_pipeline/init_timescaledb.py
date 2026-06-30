import os
import sys
import pandas as pd
from sqlalchemy import create_engine, text
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

sys.path.append(PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "microsoft-azure-predictive-maintenance")

DB_USER = "postgres"
DB_PASS = "s"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "predictive_maintenance" #database trên pgAdmin

def setup_and_seed_database():
    print("="*60)
    print(" BẮT ĐẦU NẠP DỮ LIỆU CẢM BIẾN LÊN TIMESCALEDB")
    print("="*60)
    
    # 1. Tạo kết nối
    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    
    # 2. Đọc dữ liệu CSV
    print(f"[1/4] Đang đọc file CSV từ: {DATA_DIR}/PdM_telemetry.csv...")
    df = pd.read_csv(f"{DATA_DIR}/PdM_telemetry.csv", parse_dates=['datetime'])
    
    # 3. Đẩy dữ liệu vào PostgreSQL (Chia batch nhỏ để tránh tràn RAM máy tính)
    print(f"[2/4] Đang bơm {len(df)} dòng dữ liệu vào Database...")
    start_time = time.time()
    
    # Đẩy dữ liệu vào bảng 'telemetry'
    df.to_sql('telemetry', engine, if_exists='replace', index=False, chunksize=10000)
    
    print(f" -> Nạp xong dữ liệu thô mất {time.time() - start_time:.2f} giây.")

    # 4. Biến bảng thành Hypertable của TimescaleDB
    print("[3/4] Đang cấu hình TimescaleDB Hypertable để tối ưu truy vấn thời gian...")
    with engine.connect() as conn:
        # Cấp quyền và tạo extension nếu chưa có
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
        conn.commit()
        
        # Đảm bảo cột datetime là kiểu TIMESTAMP
        conn.execute(text("ALTER TABLE telemetry ALTER COLUMN datetime TYPE TIMESTAMP;"))
        conn.commit()
        
        # Chuyển thành Hypertable
        try:
            # [SỬA LỖI] Thêm migrate_data => TRUE để TimescaleDB cấu trúc lại lượng data vừa nạp
            conn.execute(text("SELECT create_hypertable('telemetry', 'datetime', if_not_exists => TRUE, migrate_data => TRUE);"))
            conn.commit()
            print(" -> Đã kích hoạt Hypertable thành công!")
        except Exception as e:
            # [SỬA LỖI] Rollback transaction nếu bị lỗi để không chặn các lệnh SQL phía sau
            conn.rollback() 
            print(f" -> Lưu ý: {e}")
            
        # Tạo index trên machineID để query cực nhanh
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_machine_time ON telemetry (\"machineID\", datetime DESC);"))
        conn.commit()

    print("[4/4] TÍCH HỢP TIMESCALEDB THÀNH CÔNG!")
    print("Bây giờ hệ thống AI đã có thể truy vấn trực tiếp từ Database.")

if __name__ == "__main__":
    setup_and_seed_database()