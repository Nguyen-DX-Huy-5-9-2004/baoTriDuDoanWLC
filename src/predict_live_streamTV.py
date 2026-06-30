#streamlit run src/predict_live_streamTV.py
import os
import sys
import time
import torch
import shap
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
from sqlalchemy import create_engine, text
import streamlit as st

warnings.filterwarnings('ignore')

# --- CẤU HÌNH ĐƯỜNG DẪN ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)
CORRECT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "microsoft-azure-predictive-maintenance")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "saved_models")
DB_URL = "postgresql://postgres:s@localhost:5432/predictive_maintenance"

try:
    from models.config import GRU_CONFIG, XGB_CONFIG
except ModuleNotFoundError:
    from config import GRU_CONFIG, XGB_CONFIG # type: ignore
from models.gru_extractor import GRURiskExtractor

#Khởi tạo Engine Database duy nhất 1 lần ở mức toàn cục để tránh cạn kiệt Connection Pool của PostgreSQL
db_engine = create_engine(DB_URL)

# --- CẤU HÌNH GIAO DIỆN STREAMLIT CHO TV ---
st.set_page_config(page_title="AI Predictive Maintenance TV", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    /* Reset toàn bộ Streamlit mặc định để không gây ảnh hưởng */
    html, body, [class*="css"], .stApp {
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
        background-color: #000000 !important;
    }
    header, footer, #MainMenu { visibility: hidden !important; display: none !important; }

    .tv-dashboard-wrapper {
        position: fixed; /* Nhấc giao diện ra khỏi luồng mặc định */
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background-color: #121212;
        z-index: 999999; /* Đè lên tất cả mọi thứ */
        padding: 2vh 2vw; /* Lề an toàn cho Tivi */
        box-sizing: border-box;
        display: flex; /* Dùng Flexbox để chia không gian */
        flex-direction: column;
    }

    /* Thanh Top Banner SCADA (Cố định chiều cao 10vh) */
    .top-banner {
        display: flex; justify-content: space-between; align-items: center;
        background-color: #1a1a1a; border-radius: 1vh;
        border-bottom: 0.4vh solid #00c0f2;
        height: 10vh; padding: 0 2vw; margin-bottom: 2vh;
        box-shadow: 0.2vh 0.2vh 1vh rgba(0,0,0,0.5);
        flex-shrink: 0; /* KHÔNG bao giờ bị ép nhỏ lại */
    }
    .banner-title { font-size: 3.5vh; font-weight: 900; color: #FFFFFF; white-space: nowrap; }
    .kpi-group { display: flex; gap: 1vw; align-items: center; }
    .kpi-item { font-size: 2.2vh; font-weight: bold; padding: 1vh 1vw; border-radius: 0.5vh; white-space: nowrap; }
    .kpi-time { color: #00c0f2; }
    .kpi-safe { color: #28a745; background: rgba(40, 167, 69, 0.15); border: 0.2vh solid #28a745; }
    .kpi-warn { color: #ffc107; background: rgba(255, 193, 7, 0.15); border: 0.2vh solid #ffc107; }
    .kpi-danger { color: #ff4b4b; background: rgba(255, 75, 75, 0.15); border: 0.2vh solid #ff4b4b; animation: text-blink 1s infinite; }
    
    /* Lưới CSS Grid (TỰ ĐỘNG co giãn để LẤP ĐẦY khoảng trống phía dưới) */
    .machine-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        grid-template-rows: repeat(3, 1fr);
        grid-gap: 1.5vh 1vw;
        flex-grow: 1; /* QUAN TRỌNG: Lấp đầy toàn bộ khoảng trống còn lại */
        width: 100%;
    }
    
    /* Thiết kế Thẻ Máy (Machine Card) */
    .machine-card {
        border-radius: 1vh; padding: 1vh 0.5vw;
        display: flex; flex-direction: column; justify-content: flex-start;
        color: white; background-color: #1e1e1e;
        box-shadow: 0.3vh 0.3vh 0.8vh rgba(0,0,0,0.5);
        overflow: hidden; /* Cấm nội dung trào ra ngoài thẻ */
    }
    .title-text { font-size: 2.8vh; font-weight: 900; margin-bottom: 1.5vh; text-align: center; border-bottom: 0.2vh solid #333; padding-bottom: 0.5vh; white-space: nowrap;}
    
    .comp-row { 
        display: flex; justify-content: space-between; 
        margin-bottom: 1vh; font-size: 2.2vh; font-weight: bold; font-family: monospace;
    }
    
    /* Giao diện "Lật thẻ" khi có lỗi (Danger Mode) */
    .danger-mode { display: flex; flex-direction: column; height: 100%; justify-content: center; }
    .worst-comp-title { color: #ff4b4b; font-size: 2.6vh; font-weight: 900; text-align: center; margin-bottom: 1vh; animation: text-blink 1s infinite;}
    .shap-box { background: rgba(255,255,255,0.1); border-left: 0.5vh solid #ffc107; padding: 1vh; border-radius: 0.5vh; }
    .shap-reason { color: #ffffff; display: block; margin-bottom: 0.5vh; font-size: 1.8vh; line-height: 1.3; white-space: normal;}
    
    /* Màu sắc trạng thái thẻ */
    .status-normal { border: 0.3vh solid #28a745; }
    .status-warning { border: 0.3vh solid #ffc107; background-color: #2a2000; }
    .status-danger { 
        border: 0.5vh solid #ff4b4b; 
        background-color: #3a0000; 
        animation: card-blink 1.5s infinite; 
    }
    
    /* Hiệu ứng nhấp nháy */
    @keyframes card-blink {
        0%, 100% { box-shadow: 0 0 1vh #ff4b4b; }
        50% { box-shadow: 0 0 3vh #ff4b4b; border-color: #ff7676; }
    }
    @keyframes text-blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    </style>
""", unsafe_allow_html=True)

# --- CLASS & HÀM TIỆN ÍCH ---
class RealtimeMonitor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.components = ['comp1', 'comp2', 'comp3', 'comp4']
        self._load_models()

    def _load_models(self):
        self.gru_model = GRURiskExtractor(config=GRU_CONFIG).to(self.device)
        self.gru_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "gru_latest_model.pth"), map_location=self.device))
        self.gru_model.eval()

        self.xgb_models = {}
        for comp in self.components:
            model = xgb.XGBClassifier()
            model.load_model(os.path.join(MODEL_DIR, f"xgb_model_{comp}.json"))
            self.xgb_models[comp] = model

    def predict_live_status(self, telemetry_24h_scaled, current_erp_state):
        x_seq = torch.tensor(telemetry_24h_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            memory_vector = self.gru_model.extract_features(x_seq).cpu().numpy()
        hybrid_features = np.hstack((memory_vector, current_erp_state))
        
        alerts = {}
        for comp in self.components:
            alerts[comp] = self.xgb_models[comp].predict_proba(hybrid_features)[0][1] * 100
        return alerts, hybrid_features

def get_latest_timestamp():
    try:
        with db_engine.connect() as conn:
            result = conn.execute(text("SELECT MAX(datetime) FROM telemetry_live;")).scalar()
        return pd.to_datetime(result) if result else None
    except Exception:
        return None

def fetch_live_sensor_data(machine_id, target_time):
    query = f"""
    SELECT volt, rotate, pressure, vibration, original_datetime 
    FROM telemetry_live 
    WHERE "machineID" = {machine_id} AND datetime <= '{target_time}'
    ORDER BY datetime DESC
    LIMIT 24;
    """
    with db_engine.connect() as conn:
        df_24h = pd.read_sql(text(query), conn)
        
    if len(df_24h) < 24: return None, None
    df_24h = df_24h.iloc[::-1].reset_index(drop=True)
    orig_time = df_24h['original_datetime'].iloc[-1]
    sensor_values = df_24h[['volt', 'rotate', 'pressure', 'vibration']].values
    return sensor_values, orig_time

def load_feature_names():
    gru_features = [f"GRU_Node_{i}" for i in range(GRU_CONFIG['hidden_size'])]
    erp_features = ['error1', 'error2', 'error3', 'error4', 'error5', 'days_comp1', 'days_comp2', 'days_comp3', 'days_comp4', 'age']
    return gru_features + erp_features

# --- X-RAY (SHAP) CHO TIVI ---
def get_shap_explanation(comp_name, hybrid_features, monitor_instance):
    model = monitor_instance.xgb_models[comp_name]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(hybrid_features)
    
    feature_names = load_feature_names()
    contributions = list(zip(feature_names, shap_values[0], hybrid_features[0]))
    contributions.sort(key=lambda x: np.abs(x[1]), reverse=True)
    
    reasons = []
    for feat_name, shap_val, actual_val in contributions[:2]:
        if shap_val <= 0: continue 
        
        if "days_comp" in feat_name:
            reasons.append(f"⏱️Quá hạn ({actual_val:.0f} ngày)")
        elif "error" in feat_name:
            err_id = feat_name.split('_')[0].upper()
            reasons.append(f"⚠️Lỗi {err_id} (x{actual_val:.0f})")
        elif "GRU_Node" in feat_name:
            reasons.append(f"📈Dao động cảm biến bất thường")
        elif "age" in feat_name:
            reasons.append(f"🏭Tuổi thọ cao ({actual_val:.0f} năm)")
            
    if not reasons: reasons.append("Đa thông số rủi ro")
    return reasons

# --- KHỞI TẠO HỆ THỐNG ---
@st.cache_resource(show_spinner="Đang nạp não bộ AI...")
def initialize_system():
    monitor = RealtimeMonitor()
    
    from demo.data_pipeline.time_series_generator import TimeSeriesPreprocessor
    telemetry_raw = pd.read_csv(f"{CORRECT_DATA_DIR}/PdM_telemetry.csv", parse_dates=['datetime'])
    ts_preprocessor = TimeSeriesPreprocessor()
    ts_preprocessor.fit_transform_telemetry(telemetry_raw)
    means, stds = ts_preprocessor.scaler.mean_, ts_preprocessor.scaler.scale_
    
    from demo.data_pipeline.erp_feature_engineering import ERPFeatureEngineer
    erp_engineer = ERPFeatureEngineer(CORRECT_DATA_DIR)
    
    class DummyOutput:
        def write(self, x): pass
        def flush(self): pass
        
    old_stdout = sys.stdout
    sys.stdout = DummyOutput()
    try:
        erp_engineer.load_data()
        telemetry_dates = telemetry_raw[['datetime', 'machineID']].drop_duplicates()
        erp_matrix = erp_engineer.execute_pipeline(telemetry_dates)
    finally:
        sys.stdout = old_stdout 
        
    return monitor, means, stds, erp_matrix

def get_status_style(prob):
    if prob > 80: return "#ff4b4b", "🚨", "status-danger"
    elif prob > 40: return "#ffc107", "⚠️", "status-warning"
    elif prob > 15: return "#00c0f2", "👀", "status-normal"
    else: return "#28a745", "✅", "status-normal"

def generate_machine_card_html(m_id, alerts, explanations=None, worst_comp=None):
    max_risk = max(alerts.values())
    main_color, _, css_class = get_status_style(max_risk)
    
    html = f"<div class='machine-card {css_class}'>"
    html += f"<div class='title-text'>MÁY #{m_id:02d}</div>" 
    
    if max_risk > 80 and explanations:
        html += "<div class='danger-mode'>"
        html += f"<div class='worst-comp-title'>LỖI {worst_comp.upper()} ({max_risk:.1f}%)</div>"
        html += "<div class='shap-box'>"
        html += "<span style='color:#ffc107; font-weight:900; font-size:2vh; display:block; margin-bottom:0.5vh;'>NGUYÊN NHÂN:</span>"
        for reason in explanations:
            html += f"<span class='shap-reason'>- {reason}</span>"
        html += "</div></div>"
    else:
        for comp, risk in alerts.items():
            color, _, _ = get_status_style(risk)
            comp_name = comp.upper()
            html += f"<div class='comp-row'> <span>{comp_name}:</span><span style='color: {color};'>{risk:5.1f}%</span></div>"
            
    html += "</div>"
    return html

def generate_top_banner_html(latest_time, total, safe, warn, danger):
    time_str = latest_time.strftime('%Y-%m-%d %H:%M') if latest_time else "Đang kết nối..."
    html = f"""
    <div class='top-banner'>
        <div style="display: flex; flex-direction: column; justify-content: center;">
            <div class='banner-title'>TRUNG TÂM GIÁM SÁT AI (PdM 4.0)</div>
            <div style="color: #aaaaaa; font-size: 1.8vh; margin-top: 0.5vh; font-weight: bold;"> Khả năng xảy ra sự cố trên máy tự động trong vòng 48h tiếp theo </div>
        </div>
        <div class='kpi-group'>
            <div class='kpi-item kpi-time'>🕒 {time_str}</div>
            <div class='kpi-item kpi-safe'>✅ ỔN ĐỊNH: {safe}/{total}</div>
            <div class='kpi-item kpi-warn'>⚠️ CẢNH BÁO: {warn}</div>
            <div class='kpi-item kpi-danger'>🚨 BÁO ĐỘNG: {danger}</div>
        </div>
    </div>
    """
    return html

# --- VÒNG LẶP CHÍNH ---
def main():
    monitor, means, stds, erp_matrix = initialize_system()
    
    ui_placeholder = st.empty()
    last_processed_time = None
    
    while True:
        latest_time = get_latest_timestamp()
        
        if latest_time is None or latest_time == last_processed_time:
            time.sleep(1)
            continue
            
        last_processed_time = latest_time
        machine_data_dict = {}
        machine_hybrid_feats = {}
        
        try:
            with db_engine.connect() as conn:
                query = text(f"SELECT DISTINCT \"machineID\" FROM telemetry_live WHERE datetime = '{latest_time}'")
                machines_active = pd.read_sql(query, conn)['machineID'].tolist()
            
            for m_id in machines_active:
                sensor_data, orig_time = fetch_live_sensor_data(m_id, latest_time)
                if sensor_data is None: continue
                    
                scaled_sensor_data = (sensor_data - means) / stds
                
                current_erp_records = erp_matrix[(erp_matrix['datetime'] == orig_time) & (erp_matrix['machineID'] == m_id)]
                if len(current_erp_records) == 0: continue
                current_erp_record = current_erp_records.iloc[-1:]
                    
                drop_cols = ['datetime', 'machineID', 'model', 'risk_comp1', 'risk_comp2', 'risk_comp3', 'risk_comp4']
                erp_state = current_erp_record.drop(labels=drop_cols, axis=1).values.astype(float).reshape(1, -1)
                
                alerts, hybrid_features = monitor.predict_live_status(scaled_sensor_data, erp_state)
                machine_data_dict[m_id] = alerts
                machine_hybrid_feats[m_id] = hybrid_features
                
        except Exception as e:
            time.sleep(2)
            continue
            
        total_machines = len(machine_data_dict)
        if total_machines == 0:
            ui_placeholder.info("Đang chờ dữ liệu...")
            time.sleep(1)
            continue

        danger_count = sum(1 for alerts in machine_data_dict.values() if max(alerts.values()) > 80)
        warning_count = sum(1 for alerts in machine_data_dict.values() if 40 < max(alerts.values()) <= 80)
        safe_count = total_machines - danger_count - warning_count
        
        full_html = "<div class='tv-dashboard-wrapper'>"

        full_html += generate_top_banner_html(latest_time, total_machines, safe_count, warning_count, danger_count)
        
        full_html += "<div class='machine-grid'>"
        
        for m_id, alerts in sorted(machine_data_dict.items()):
            max_risk = max(alerts.values())
            explanations = None
            worst_comp = None
            
            if max_risk > 80:
                worst_comp = max(alerts, key=alerts.get)
                hybrid_features = machine_hybrid_feats[m_id]
                explanations = get_shap_explanation(worst_comp, hybrid_features, monitor)
            
            full_html += generate_machine_card_html(m_id, alerts, explanations, worst_comp)
            
        full_html += "</div>" 
        full_html += "</div>" 
    
        ui_placeholder.markdown(full_html, unsafe_allow_html=True)
        
        time.sleep(1)

if __name__ == "__main__":
    main()