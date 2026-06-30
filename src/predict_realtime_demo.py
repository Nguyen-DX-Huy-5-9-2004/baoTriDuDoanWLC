#test chất lượng mô hình sau train. Chuyển sang thực tế cần thay đổi quy cách lấy data, thiết lập pipelen và chuẩn hóa
import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
from datetime import timedelta
from sqlalchemy import create_engine, text

warnings.filterwarnings('ignore')

# Xử lý đường dẫn linh hoạt
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)
CORRECT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "microsoft-azure-predictive-maintenance")

try:
    from models.config import GRU_CONFIG, XGB_CONFIG, SEQ_LENGTH
except ModuleNotFoundError:
    from config import GRU_CONFIG, XGB_CONFIG, SEQ_LENGTH # type: ignore
from models.gru_extractor import GRURiskExtractor

MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "saved_models")

DB_URL = "postgresql://postgres:s@localhost:5432/predictive_maintenance"

class RealtimeMonitor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.components = ['comp1', 'comp2', 'comp3', 'comp4']
        self._load_models()

    def _load_models(self):
        self.gru_model = GRURiskExtractor(config=GRU_CONFIG).to(self.device)
        gru_path = os.path.join(MODEL_DIR, "gru_latest_model.pth")
        self.gru_model.load_state_dict(torch.load(gru_path, map_location=self.device))
        self.gru_model.eval()

        self.xgb_models = {}
        for comp in self.components:
            xgb_path = os.path.join(MODEL_DIR, f"xgb_model_{comp}.json")
            model = xgb.XGBClassifier()
            model.load_model(xgb_path)
            self.xgb_models[comp] = model

    def predict_live_status(self, machine_id, telemetry_24h_scaled, current_erp_state):
        x_seq = torch.tensor(telemetry_24h_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            memory_vector = self.gru_model.extract_features(x_seq).cpu().numpy()
            
        hybrid_features = np.hstack((memory_vector, current_erp_state))
        
        alerts = {}
        for comp in self.components:
            risk_prob = self.xgb_models[comp].predict_proba(hybrid_features)[0][1]
            alerts[comp] = risk_prob * 100
        return alerts, hybrid_features


def fetch_sensor_data_from_db(machine_id, current_time):
    """
    Truy vấn trực tiếp TimescaleDB để lấy 24h dữ liệu cảm biến LÙI VỀ TRƯỚC từ current_time
    """
    engine = create_engine(DB_URL)
    
    # Câu lệnh SQL đặc trưng của TimescaleDB trong môi trường Production
    query = f"""
    SELECT volt, rotate, pressure, vibration 
    FROM telemetry 
    WHERE "machineID" = {machine_id} AND datetime <= '{current_time}'
    ORDER BY datetime DESC
    LIMIT 24;
    """
    with engine.connect() as conn:
        df_24h = pd.read_sql(text(query), conn)
        
    # Lật ngược dataframe vì SQL lấy DESC (hiện tại lùi về quá khứ)
    # Nhưng GRU cần đọc thuận chiều thời gian (quá khứ tới hiện tại)
    df_24h = df_24h.iloc[::-1].reset_index(drop=True)
    return df_24h.values


def simulate_production_timeline(target_machine_id=2):
    """
    Mô phỏng đồng hồ hệ thống chạy liên tục.
    Tua thời gian về trước khi máy hỏng 48 tiếng và bắt đầu chạy thực tế.
    """
    print("\n" + "="*80)
    print("KHỞI ĐỘNG HỆ THỐNG GIÁM SÁT THỜI GIAN THỰC (LẤY DATA TỪ TIMESCALEDB)")
    print("="*80)
    
    # 1. Khởi tạo Engine và Scaler (Chỉ làm 1 lần lúc bật hệ thống)
    monitor = RealtimeMonitor()
    
    print("[HỆ THỐNG] Đang tải cấu hình Chuẩn hóa dữ liệu...")
    from demo.data_pipeline.time_series_generator import TimeSeriesPreprocessor
    telemetry_raw = pd.read_csv(f"{CORRECT_DATA_DIR}/PdM_telemetry.csv", parse_dates=['datetime'])
    ts_preprocessor = TimeSeriesPreprocessor()
    ts_preprocessor.fit_transform_telemetry(telemetry_raw)
    means = ts_preprocessor.scaler.mean_
    stds = ts_preprocessor.scaler.scale_
    
    # 2. Lấy dữ liệu ERP tĩnh (Chỉ cần load 1 lần, sau đó sẽ truy cập theo thời gian thực)
    print("[HỆ THỐNG] Đang kết nối với module ERP...")
    from demo.data_pipeline.erp_feature_engineering import ERPFeatureEngineer
    erp_engineer = ERPFeatureEngineer(CORRECT_DATA_DIR)
    erp_engineer.load_data()
    telemetry_dates = telemetry_raw[telemetry_raw['machineID'] == target_machine_id][['datetime', 'machineID']].drop_duplicates()
    
    # Ẩn log
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w', encoding='utf-8')
    erp_matrix = erp_engineer.execute_pipeline(telemetry_dates)
    sys.stdout = old_stdout
    
    # 3. THIẾT LẬP CỖ MÁY THỜI GIAN
    failing_records = erp_matrix[erp_matrix['risk_comp2'] == 1]
    if len(failing_records) == 0:
        return
        
    # Lấy mốc thời gian máy bốc khói (Giờ G)
    crash_time = failing_records.iloc[-1]['datetime']
    
    # Cài đặt đồng hồ quay về trước giờ G 48 tiếng
    current_simulated_time = crash_time - timedelta(hours=48)
    
    print(f"\nCẢNH BÁO MÔ PHỎNG: Máy số {target_machine_id} sẽ hỏng COMP2 vào lúc: {crash_time}")
    print("Tua thời gian về 48 giờ trước. Bắt đầu giám sát liên tục...\n")
    print("-"*80)
    
    # 4. VÒNG LẶP THỜI GIAN THỰC (Mỗi vòng lặp mô phỏng 1 giờ trôi qua)
    while current_simulated_time <= crash_time:
        # Giao diện đồng hồ
        print(f"THỜI GIAN THỰC TẾ: {current_simulated_time} | Đang truy vấn CSDL...", end="\r")
        
        # BƯỚC A: Lấy 24h cảm biến sát nhất từ Database
        raw_sensor_data = fetch_sensor_data_from_db(target_machine_id, current_simulated_time)
        if len(raw_sensor_data) < 24:
            current_simulated_time += timedelta(hours=1)
            continue
            
        # BƯỚC B: Chuẩn hóa tín hiệu
        scaled_sensor_data = (raw_sensor_data - means) / stds
        
        # BƯỚC C: Lấy trạng thái ERP tại đúng thời điểm đó
        current_erp_record = erp_matrix[erp_matrix['datetime'] == current_simulated_time]
        if len(current_erp_record) == 0:
            current_simulated_time += timedelta(hours=1)
            continue
            
        drop_cols = ['datetime', 'machineID', 'model', 'risk_comp1', 'risk_comp2', 'risk_comp3', 'risk_comp4']
        erp_state = current_erp_record.drop(labels=drop_cols, axis=1).values.astype(float).reshape(1, -1)
        
        # BƯỚC D: AI Phán quyết
        alerts, _ = monitor.predict_live_status(target_machine_id, scaled_sensor_data, erp_state)
        
        prob_comp2 = alerts['comp2']
        
        # Xóa dòng in "Đang truy vấn" và in kết quả chính thức
        sys.stdout.write('\033[2K\033[1G') 
        
        if prob_comp2 > 80:
            status = f" BÁO ĐỘNG ĐỎ ({prob_comp2:.1f}%)"
        elif prob_comp2 > 50:
            status = f" CẢNH BÁO ({prob_comp2:.1f}%)"
        elif prob_comp2 > 20:
            status = f" CHÚ Ý ({prob_comp2:.1f}%)"
        else:
            status = f" BÌNH THƯỜNG ({prob_comp2:.1f}%)"
            
        print(f"LÚC {current_simulated_time} | Trạng thái Cụm 2: {status}")
        
        # Tăng thời gian lên 1 tiếng cho vòng lặp sau
        current_simulated_time += timedelta(hours=1)
        
        # Nghỉ 1 giây ngoài đời thực để người xem kịp quan sát sự thay đổi
        time.sleep(1)

    print("-"*80)
    print(f" BÙM! MÁY ĐÃ HỎNG VÀO LÚC {crash_time}.")
    print("Nếu kỹ sư trực màn hình này, họ đã có thể dừng máy từ hơn 20 tiếng trước đó!")

if __name__ == "__main__":
    simulate_production_timeline(target_machine_id=2)