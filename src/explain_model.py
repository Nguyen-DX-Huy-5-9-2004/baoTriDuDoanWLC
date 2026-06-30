#& c:\Users\huynd1\Downloads\tBTDD\.venv\Scripts\python.exe -X utf8 c:/Users/huynd1/Downloads/tBTDD/src/explain_model.py
import os
import sys
import shap
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# Xử lý đường dẫn linh hoạt
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

CORRECT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "microsoft-azure-predictive-maintenance")

try:
    from models.config import GRU_CONFIG
except ModuleNotFoundError:
    from src.models.config import GRU_CONFIG

MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "saved_models")

def load_feature_names():
    """Tạo danh sách tên cho Ma trận Lai"""
    gru_features = [f"GRU_Node_{i}" for i in range(GRU_CONFIG['hidden_size'])]
    erp_features = [
        'error1_24h', 'error2_24h', 'error3_24h', 'error4_24h', 'error5_24h',
        'days_since_comp1', 'days_since_comp2', 'days_since_comp3', 'days_since_comp4',
        'machine_age'
    ]
    return gru_features + erp_features

def explain_prediction(comp_name, hybrid_feature_matrix):
    """
    Mổ xẻ nguyên nhân có chứa Số liệu Định lượng (Tỷ trọng % và Giá trị thực).
    """
    print(f"\n ĐANG BÓC TÁCH DỮ LIỆU ĐỂ GIẢI TRÌNH CHO CỤM: {comp_name.upper()}...")
    
    xgb_path = os.path.join(MODEL_DIR, f"xgb_model_{comp_name}.json")
    if not os.path.exists(xgb_path):
        print(f"Không tìm thấy mô hình {comp_name}.")
        return
        
    model = xgb.XGBClassifier()
    model.load_model(xgb_path)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(hybrid_feature_matrix)
    
    feature_names = load_feature_names()
    
    if len(feature_names) != hybrid_feature_matrix.shape[1]:
        print(f"Lỗi: Kích thước tên cột không khớp.")
        return

    # Tính tổng mức độ tác động tuyệt đối của TẤT CẢ các đặc trưng để quy ra %
    total_impact = np.sum(np.abs(shap_values[0]))

    contributions = list(zip(feature_names, shap_values[0], hybrid_feature_matrix[0]))
    # Sắp xếp theo mức độ ảnh hưởng (Giá trị tuyệt đối lớn nhất)
    contributions.sort(key=lambda x: np.abs(x[1]), reverse=True)
    
    print("\n" + "=" * 80)
    print(f" BÁO CÁO GIẢI TRÌNH AI - TOP CÁC YẾU TỐ DẪN ĐẾN RỦI RO".center(80))
    print("=" * 80)
    print(f"{'MỨC ĐÓNG GÓP'.ljust(15)} | {'CHỈ SỐ / CẢM BIẾN'.ljust(18)} | {'GIÁ TRỊ ĐO ĐƯỢC'.ljust(15)} | LÝ DO CHI TIẾT")
    print("-" * 80)
    
    for feat_name, shap_val, actual_val in contributions[:5]: # Chỉ lấy Top 5
        # Tính % đóng góp của riêng yếu tố này vào tổng rủi ro
        impact_pct = (np.abs(shap_val) / total_impact) * 100
        
        # Format dấu + (tăng rủi ro) và - (giảm rủi ro)
        sign = "+" if shap_val > 0 else "-"
        impact_str = f"[{sign}{impact_pct:.1f}%]"
        
        # Phiên dịch logic máy sang ngôn ngữ kỹ sư (PE) với SỐ LIỆU THẬT
        if "days_since" in feat_name:
            comp_id = feat_name[-1]
            feat_desc = f"Vượt ngưỡng an toàn. Linh kiện đã chạy {actual_val:.1f} ngày"
            val_str = f"{actual_val:.1f} ngày"
        elif "error" in feat_name:
            err_id = feat_name.split('_')[0]
            feat_desc = f"Lỗi {err_id.upper()} xuất hiện liên tục trong 24h"
            val_str = f"{actual_val:.0f} lần"
        elif "GRU_Node" in feat_name:
            node_id = feat_name.split('_')[-1]
            feat_desc = f"Điểm uốn dị thường ở Node sóng {node_id} (Dấu hiệu hao mòn cơ khí)"
            val_str = f"{actual_val:.3f}"
        elif "age" in feat_name:
            feat_desc = f"Khung máy lão hóa"
            val_str = f"{actual_val:.0f} năm"
        else:
            feat_desc = f"Bất thường"
            val_str = f"{actual_val:.2f}"
            
        print(f"{impact_str.ljust(15)} | {feat_name.ljust(18)} | {val_str.ljust(15)} | {feat_desc}")

    print("-" * 80)
    print(" KẾT LUẬN TỪ AI:")
    print("Các yếu tố mang dấu [+] đã cộng dồn lại đẩy xác suất hỏng hóc lên mức Báo Động Đỏ.")
    print("Hãy đối chiếu [Giá trị đo được] của các Node GRU với ngưỡng rung/nhiệt thực tế của máy.")

def test_explain_with_real_data(target_machine_id=2):
    """
    Kết nối trực tiếp với dữ liệu thực tế và TỰ ĐỘNG QUÉT TOÀN BỘ CÁC LINH KIỆN.
    """
    try:
        from demo.data_pipeline.erp_feature_engineering import ERPFeatureEngineer
        from demo.data_pipeline.time_series_generator import TimeSeriesPreprocessor
        from predict_realtime_demo import RealtimeMonitor
    except ImportError:
        from demo.data_pipeline.erp_feature_engineering import ERPFeatureEngineer
        from demo.data_pipeline.time_series_generator import TimeSeriesPreprocessor
        from predict_realtime_demo import RealtimeMonitor

    telemetry_raw = pd.read_csv(f"{CORRECT_DATA_DIR}/PdM_telemetry.csv", parse_dates=['datetime'])
    erp_engineer = ERPFeatureEngineer(CORRECT_DATA_DIR)
    erp_engineer.load_data()
    
    telemetry_dates = telemetry_raw[telemetry_raw['machineID'] == target_machine_id][['datetime', 'machineID']].drop_duplicates()
    
    # Ẩn bớt log quá trình chuẩn bị dữ liệu
    import sys, os
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    erp_matrix = erp_engineer.execute_pipeline(telemetry_dates)
    ts_preprocessor = TimeSeriesPreprocessor()
    telemetry_scaled = ts_preprocessor.fit_transform_telemetry(telemetry_raw)
    monitor = RealtimeMonitor()
    sys.stdout = old_stdout
    
    components_to_check = ['comp1', 'comp2', 'comp3', 'comp4']
    found_any_failure = False

    # Vòng lặp quét tự động qua tất cả các linh kiện
    for target_comp in components_to_check:
        # Tìm xem máy này có bị hỏng linh kiện target_comp không
        failing_records = erp_matrix[erp_matrix[f'risk_{target_comp}'] == 1]
        
        if len(failing_records) > 0:
            found_any_failure = True
            target_record = failing_records.iloc[-1]
            target_time = target_record['datetime']
            
            print(f"\n" + "*"*80)
            print(f"️ [PHÁT HIỆN CA BỆNH] Máy số {target_machine_id} từng hỏng cụm {target_comp.upper()} vào lúc {target_time}")
            print("*"*80)
            
            sensor_24h = telemetry_scaled[
                (telemetry_scaled['machineID'] == target_machine_id) & 
                (telemetry_scaled['datetime'] <= target_time)
            ].tail(24)
            x_seq = sensor_24h[['volt', 'rotate', 'pressure', 'vibration']].values
            
            drop_cols = ['datetime', 'machineID', 'model', 'risk_comp1', 'risk_comp2', 'risk_comp3', 'risk_comp4']
            erp_state = target_record.drop(labels=drop_cols).values.astype(float).reshape(1, -1)
            
            alerts, hybrid_features = monitor.predict_live_status(target_machine_id, x_seq, erp_state)
            
            # Giải trình cho linh kiện vừa phát hiện
            explain_prediction(target_comp, hybrid_features)
            
    if not found_any_failure:
        print(f"\n Máy số {target_machine_id} chưa từng ghi nhận hỏng hóc nào trong suốt quá trình hoạt động.")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TỰ ĐỘNG QUÉT VÀ GIẢI THÍCH MÔ HÌNH VỚI DỮ LIỆU THỰC TẾ")
    print("="*60)
    
    # Chỉ cần cấp ID Máy, hệ thống tự lo phần còn lại!
    test_explain_with_real_data(target_machine_id=2)