#python src/explain_live_alert.py
import os
import sys
import shap
import torch
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
from sqlalchemy import create_engine, text

warnings.filterwarnings('ignore')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)
CORRECT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "microsoft-azure-predictive-maintenance")

try:
    from models.config import GRU_CONFIG
except ModuleNotFoundError:
    from config import GRU_CONFIG # type: ignore

from models.gru_extractor import GRURiskExtractor

MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "saved_models")
DB_URL = "postgresql://postgres:s@localhost:5432/predictive_maintenance"

def load_feature_names():
    gru_features = [f"GRU_Node_{i}" for i in range(GRU_CONFIG['hidden_size'])]
    erp_features = [
        'error1_24h', 'error2_24h', 'error3_24h', 'error4_24h', 'error5_24h',
        'days_since_comp1', 'days_since_comp2', 'days_since_comp3', 'days_since_comp4',
        'machine_age'
    ]
    return gru_features + erp_features

def explain_live_machine(machine_id):
    print(f"\nĐANG TRỤC XUẤT HỒ SƠ BỆNH ÁN THỜI GIAN THỰC CỦA MÁY #{machine_id}...")
    
    # 1. Quét CSDL lấy 24h gần nhất
    engine = create_engine(DB_URL)
    query = f"""
    SELECT volt, rotate, pressure, vibration, datetime, original_datetime 
    FROM telemetry_live 
    WHERE "machineID" = {machine_id}
    ORDER BY datetime DESC
    LIMIT 24;
    """
    with engine.connect() as conn:
        df_24h = pd.read_sql(text(query), conn)
        
    if len(df_24h) < 24:
        print("Máy chưa tích lũy đủ 24 giờ dữ liệu để phân tích (Hoặc bạn nhập sai ID)!")
        return
        
    df_24h = df_24h.iloc[::-1].reset_index(drop=True)
    raw_sensor = df_24h[['volt', 'rotate', 'pressure', 'vibration']].values
    latest_time = df_24h['datetime'].iloc[-1]
    orig_time = df_24h['original_datetime'].iloc[-1]
    
    print(f"Mốc thời gian chẩn đoán: {latest_time} (Dữ liệu gốc: {orig_time})")

    print("Đang nạp hệ thống tiền xử lý ERP (Mất khoảng 3 giây)...")
    from demo.data_pipeline.time_series_generator import TimeSeriesPreprocessor
    from demo.data_pipeline.erp_feature_engineering import ERPFeatureEngineer
    
    # Ép kiểu stdout để tắt log rác
    class DummyOutput:
        def write(self, x): pass
        def flush(self): pass
        
    old_stdout = sys.stdout
    sys.stdout = DummyOutput()
    try:
        telemetry_raw = pd.read_csv(f"{CORRECT_DATA_DIR}/PdM_telemetry.csv", parse_dates=['datetime'])
        ts_preprocessor = TimeSeriesPreprocessor()
        ts_preprocessor.fit_transform_telemetry(telemetry_raw)
        means, stds = ts_preprocessor.scaler.mean_, ts_preprocessor.scaler.scale_
        
        erp_engineer = ERPFeatureEngineer(CORRECT_DATA_DIR)
        erp_engineer.load_data()
        telemetry_dates = telemetry_raw[['datetime', 'machineID']].drop_duplicates()
        erp_matrix = erp_engineer.execute_pipeline(telemetry_dates)
    finally:
        sys.stdout = old_stdout 
        
    scaled_sensor = (raw_sensor - means) / stds
    
    # 2. Lấy dữ liệu tĩnh tương ứng
    current_erp_records = erp_matrix[(erp_matrix['datetime'] == orig_time) & (erp_matrix['machineID'] == machine_id)]
    if len(current_erp_records) == 0:
        print("Không tìm thấy hồ sơ ERP tĩnh cho mốc thời gian này.")
        return
    current_erp_record = current_erp_records.iloc[-1:]
    drop_cols = ['datetime', 'machineID', 'model', 'risk_comp1', 'risk_comp2', 'risk_comp3', 'risk_comp4']
    erp_state = current_erp_record.drop(labels=drop_cols, axis=1).values.astype(float).reshape(1, -1)

    # 3. Phân tích qua GRU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gru_model = GRURiskExtractor(config=GRU_CONFIG).to(device)
    gru_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "gru_latest_model.pth"), map_location=device))
    gru_model.eval()

    x_seq = torch.tensor(scaled_sensor, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        memory_vector = gru_model.extract_features(x_seq).cpu().numpy()
        
    hybrid_features = np.hstack((memory_vector, erp_state))
    
    # 4. Quét SHAP (X-Ray) cho các linh kiện có rủi ro
    print("ĐANG QUÉT X-RAY TÌM NGUYÊN NHÂN LÕI...\n")
    feature_names = load_feature_names()
    
    found_risk = False
    for comp in ['comp1', 'comp2', 'comp3', 'comp4']:
        xgb_path = os.path.join(MODEL_DIR, f"xgb_model_{comp}.json")
        model = xgb.XGBClassifier()
        model.load_model(xgb_path)
        
        risk_prob = model.predict_proba(hybrid_features)[0][1] * 100
        
        # Chỉ hiển thị giải trình nếu linh kiện đó đang bị cảnh báo (>15%)
        if risk_prob > 15.0:
            found_risk = True
            print("="*80)
            print(f"BÁO CÁO GIẢI TRÌNH: RỦI RO {comp.upper()} = {risk_prob:.1f}%".center(80))
            print("="*80)
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(hybrid_features)
            
            total_impact = np.sum(np.abs(shap_values[0]))
            contributions = list(zip(feature_names, shap_values[0], hybrid_features[0]))
            contributions.sort(key=lambda x: np.abs(x[1]), reverse=True)
            
            print(f"{'MỨC ĐÓNG GÓP'.ljust(15)} | {'CHỈ SỐ / CẢM BIẾN'.ljust(18)} | {'GIÁ TRỊ ĐO ĐƯỢC'.ljust(15)} | LÝ DO CHI TIẾT")
            print("-" * 80)
            
            for feat_name, shap_val, actual_val in contributions[:5]:
                impact_pct = (np.abs(shap_val) / total_impact) * 100
                sign = "+" if shap_val > 0 else "-"
                impact_str = f"[{sign}{impact_pct:.1f}%]"
                
                if "days_since" in feat_name:
                    feat_desc = f"Linh kiện đã chạy {actual_val:.1f} ngày"
                    val_str = f"{actual_val:.1f} ngày"
                elif "error" in feat_name:
                    err_id = feat_name.split('_')[0]
                    feat_desc = f"Lỗi {err_id.upper()} xuất hiện"
                    val_str = f"{actual_val:.0f} lần"
                elif "GRU_Node" in feat_name:
                    node_id = feat_name.split('_')[-1]
                    feat_desc = f"Điểm uốn sóng {node_id} (Hao mòn cơ khí)"
                    val_str = f"{actual_val:.3f}"
                elif "age" in feat_name:
                    feat_desc = f"Khung máy lão hóa"
                    val_str = f"{actual_val:.0f} năm"
                else:
                    feat_desc = f"Bất thường"
                    val_str = f"{actual_val:.2f}"
                    
                print(f"{impact_str.ljust(15)} | {feat_name.ljust(18)} | {val_str.ljust(15)} | {feat_desc}")
            print("\n")
            
    if not found_risk:
        print(f"Máy #{machine_id} hiện tại đang hoạt động RẤT ỔN ĐỊNH. Tất cả linh kiện đều dưới 15% rủi ro.")

if __name__ == "__main__":
    try:
        m_id = input("🔍 Nhập ID Máy cần chẩn đoán (VD: 7): ").strip()
        explain_live_machine(int(m_id))
    except ValueError:
        print("Vui lòng nhập một con số hợp lệ!")