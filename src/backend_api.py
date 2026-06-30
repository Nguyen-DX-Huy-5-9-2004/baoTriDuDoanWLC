#C:\Users\huynd1\Downloads\tBTDD\src\backend_api.py
import os
import sys
import torch
import shap
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
import threading
import subprocess
import random
import json
import logging
from sqlalchemy import create_engine, text
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# --- CẤU HÌNH ĐƯỜNG DẪN ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)
CORRECT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "microsoft-azure-predictive-maintenance")
MODEL_DIR = os.path.join(PROJECT_ROOT, "demo", "saved_models")
DB_URL = "postgresql://postgres:s@localhost:5432/predictive_maintenance"

try:
    from models.config import GRU_CONFIG
except ModuleNotFoundError:
    from config import GRU_CONFIG
from models.gru_extractor import GRURiskExtractor

db_engine = create_engine(DB_URL)
app = FastAPI(title="PdM 4.0 Backend API")

# --- GLOBAL VARIABLES FOR FEED CONTROL ---
feed_thread = None
feed_process = None

# --- GLOBAL VARIABLES FOR RISK TRACKING ---
# Lưu trữ giá trị rủi ro trước đó
previous_risk_state = {}  # {machine_id: {"max_risk": float, "trend": str}}
RISK_WARNING_THRESHOLD = 40  # Ngưỡng bắt đầu cảnh báo
RISK_DANGER_THRESHOLD = 85   # Ngưỡng báo động
RISK_TREND_WINDOW = 3        # Số chu kỳ để xác định xu hướng

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CLASS & HÀM TIỆN ÍCH (AI LOGIC) ---
class RealtimeMonitor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.components = ['comp1', 'comp2', 'comp3', 'comp4']
        self._load_models()

    def _load_models(self):
        try:
            self.gru_model = GRURiskExtractor(config=GRU_CONFIG).to(self.device)
            self.gru_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "gru_latest_model.pth"), map_location=self.device))
            self.gru_model.eval()
            self.xgb_models = {}
            for comp in self.components:
                model = xgb.XGBClassifier()
                model.load_model(os.path.join(MODEL_DIR, f"xgb_model_{comp}.json"))
                self.xgb_models[comp] = model
            self.models_loaded = True
            logger.info("Đã load toàn bộ AI models!")
        except Exception as e:
            logger.error(f"Lỗi khi load AI models: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.models_loaded = False

    def predict_live_status(self, telemetry_24h_scaled, current_erp_state):
        logger.info(f" [AI Input] Telemetry shape: {telemetry_24h_scaled.shape}, ERP state shape: {current_erp_state.shape}")
        if not self.models_loaded:
            alerts = {comp: random.uniform(0, 5) for comp in self.components}
            logger.warning(f"Models not loaded, returning mock: {alerts}")
            return alerts, np.zeros((1, 20))
        
        x_seq = torch.tensor(telemetry_24h_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            memory_vector = self.gru_model.extract_features(x_seq).cpu().numpy()
        hybrid_features = np.hstack((memory_vector, current_erp_state))
        
        alerts = {}
        for comp in self.components:
            # prob = self.xgb_models[comp].predict_proba(hybrid_features)[0][1] * 100
            raw_prob = self.xgb_models[comp].predict_proba(hybrid_features)[0][1]
            # Làm mềm xác suất: Kéo các giá trị cực đoan về phía giữa
            smoothed_prob = (raw_prob ** 0.5) if raw_prob < 0.5 else (1 - (1 - raw_prob) ** 0.5)
            alerts[comp] = float(smoothed_prob * 100)
        
        logger.info(f"[AI Output] Predicted alerts: {alerts}")
        return alerts, hybrid_features

# --- GLOBAL AI MONITOR ---
logger.info("Đang khởi tạo AI Monitor...")
monitor = RealtimeMonitor()

# --- CLEAR OLD DATA ---
logger.info("Đang xóa dữ liệu cũ trong cơ sở dữ liệu...")
try:
    with db_engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE telemetry_stream CASCADE;"))
        conn.execute(text("TRUNCATE TABLE machine_live_metrics CASCADE;"))
        conn.commit()
    logger.info("Đã xóa dữ liệu cũ!")
except Exception as e:
    logger.warning(f"⚠️ Không thể xóa dữ liệu cũ: {e}")

# --- PREPROCESSORS & ERP DATA ---
logger.info("Đang tải preprocessor và ERP data...")
from demo.data_pipeline.time_series_generator import TimeSeriesPreprocessor
telemetry_raw = pd.read_csv(os.path.join(CORRECT_DATA_DIR, "PdM_telemetry.csv"), parse_dates=['datetime'])
ts_preprocessor = TimeSeriesPreprocessor()
ts_preprocessor.fit_transform_telemetry(telemetry_raw)
means, stds = ts_preprocessor.scaler.mean_, ts_preprocessor.scaler.scale_
logger.info(f"-> Preprocessor loaded: means={means}, stds={stds}")

from demo.data_pipeline.erp_feature_engineering import ERPFeatureEngineer
erp_engineer = ERPFeatureEngineer(CORRECT_DATA_DIR)

class DummyOutput:
    def write(self, x): pass
    def flush(self, x): pass
old_stdout = sys.stdout
sys.stdout = DummyOutput()
try:
    erp_engineer.load_data()
    telemetry_dates = telemetry_raw[['datetime', 'machineID']].drop_duplicates()
    erp_matrix = erp_engineer.execute_pipeline(telemetry_dates)
    logger.info(f"->ERP matrix loaded, shape: {erp_matrix.shape}, columns: {erp_matrix.columns.tolist()}")
finally:
    sys.stdout = old_stdout


def get_machines_from_db():
    try:
        with db_engine.connect() as conn:
            query = text("""
                SELECT machine_id, machine_code, machine_type, location, image_url, status
                FROM machines
                ORDER BY machine_id
            """)
            df_machines = pd.read_sql(query, conn)
        return df_machines
    except Exception as e:
        print(f"Lỗi khi lấy danh sách máy: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def get_components_from_db():
    try:
        with db_engine.connect() as conn:
            query = text("""
                SELECT c.comp_id, c.machine_id, c.comp_code, c.comp_name, c.baseline_lifespan_days
                FROM components c
                ORDER BY c.machine_id, c.comp_code
            """)
            df_components = pd.read_sql(query, conn)
        return df_components
    except Exception as e:
        print(f"Lỗi khi lấy danh sách linh kiện: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def get_machine_live_metrics():
    try:
        with db_engine.connect() as conn:
            query = text("""
                SELECT DISTINCT ON (machine_id) 
                    machine_id, load_percentage, operation_level
                FROM machine_live_metrics
                ORDER BY machine_id, updated_at DESC
            """)
            df_metrics = pd.read_sql(query, conn)
        return df_metrics
    except Exception as e:
        print(f"Lỗi khi lấy công suất máy: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def get_latest_timestamp():
    try:
        with db_engine.connect() as conn:
            result = conn.execute(text("SELECT MAX(time) FROM telemetry_stream;")).scalar()
        return pd.to_datetime(result) if result else None
    except Exception as e:
        logger.error(f"Lỗi khi lấy latest timestamp: {e}")
        return None


def fetch_live_sensor_data(machine_id, target_time):
    try:
        query = text(f"""
            SELECT 
                (sensor_payload->>'volt')::float as volt,
                (sensor_payload->>'rotate')::float as rotate,
                (sensor_payload->>'pressure')::float as pressure,
                (sensor_payload->>'vibration')::float as vibration,
                (sensor_payload->>'original_datetime') as original_datetime,
                (sensor_payload->>'original_machine_id')::int as original_machine_id
            FROM telemetry_stream 
            WHERE machine_id = :mid AND time <= :target
            ORDER BY time DESC
            LIMIT 24;
        """)
        
        with db_engine.connect() as conn:
            df_24h = pd.read_sql(query, conn, params={'mid': machine_id, 'target': target_time})
            
        # Giảm yêu cầu từ 24 xuống 6 records để có thể chạy AI sớm hơn
        if len(df_24h) < 2: 
            logger.warning(f"⚠️ [Machine {machine_id}] Only {len(df_24h)} records found (need at least 6)")
            return None, None, None
        
        # Check if we have the required fields
        if (pd.isna(df_24h['original_datetime'].iloc[-1]) or 
            pd.isna(df_24h['original_machine_id'].iloc[-1])):
            logger.warning(f"⚠️ [Machine {machine_id}] Missing original_datetime or original_machine_id")
            return None, None, None
        
        # Check if sensor data is valid
        if df_24h[['volt', 'rotate', 'pressure', 'vibration']].isnull().any().any():
            logger.warning(f"⚠️ [Machine {machine_id}] Missing sensor data")
            return None, None, None
        
        df_24h = df_24h.iloc[::-1].reset_index(drop=True)
        sensor_values = df_24h[['volt', 'rotate', 'pressure', 'vibration']].values
        original_datetime_str = df_24h['original_datetime'].iloc[-1]
        original_machine_id = int(df_24h['original_machine_id'].iloc[-1])
        original_datetime = pd.to_datetime(original_datetime_str)
        
        logger.info(f" [Machine {machine_id}] Found {len(df_24h)} records: original_machine_id={original_machine_id}, original_datetime={original_datetime}")
        return sensor_values, original_datetime, original_machine_id
    except Exception as e:
        logger.error(f" [Machine {machine_id}] Error fetching sensor data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None


def load_feature_names():
    gru_features = [f"GRU_Node_{i}" for i in range(GRU_CONFIG['hidden_size'])]
    erp_features = ['error1', 'error2', 'error3', 'error4', 'error5', 'days_comp1', 'days_comp2', 'days_comp3', 'days_comp4', 'age']
    return gru_features + erp_features


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
            reasons.append(f"Quá hạn ({actual_val:.0f} ngày)")
        elif "error" in feat_name: 
            reasons.append(f"Lỗi {feat_name.split('_')[0].upper()} (x{actual_val:.0f})")
        elif "GRU_Node" in feat_name: 
            reasons.append(f"Dao động cảm biến bất thường")
        elif "age" in feat_name: 
            reasons.append(f"Tuổi thọ cao ({actual_val:.0f} năm)")
    if not reasons: 
        reasons.append("Đa thông số rủi ro")
    return reasons


# --- API ENDPOINTS ---
@app.get("/api/live-status")
def get_live_status():
    try:
        df_machines = get_machines_from_db()
        df_components = get_components_from_db()
        df_metrics = get_machine_live_metrics()
        
        latest_time = get_latest_timestamp()
        logger.info(f"\n Latest timestamp: {latest_time}")
        
        # Check if we have ANY telemetry data at all
        has_telemetry = False
        try:
            with db_engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM telemetry_stream;")).scalar()
                has_telemetry = (result > 0)
        except:
            has_telemetry = False
        
        if not has_telemetry or df_machines.empty:
            return {"status": "waiting_data"}
        
        machine_data_list = []
        danger_count = warning_count = 0
        
        active_machines = []
        if latest_time:
            with db_engine.connect() as conn:
                query = text("SELECT DISTINCT machine_id FROM telemetry_stream WHERE time = :latest")
                active_machines = pd.read_sql(query, conn, params={'latest': latest_time})['machine_id'].tolist()
        logger.info(f"🔧 Active machines: {active_machines}")
        
        for _, machine in df_machines.iterrows():
            machine_id = machine['machine_id']
            machine_name = machine['machine_type']
            location = machine['location']
            image_url = machine['image_url']
            status = machine['status']
            
            machine_comp_rows = df_components[df_components['machine_id'] == machine_id]
            
            alerts = {}
            worst_comp = None
            explanations = None
            max_risk = 0
            
            load_percentage = None
            operation_level = None
            if not df_metrics.empty:
                metric_row = df_metrics[df_metrics['machine_id'] == machine_id]
                if not metric_row.empty:
                    load_percentage = metric_row.iloc[0]['load_percentage']
                    operation_level = metric_row.iloc[0]['operation_level']
            
            if latest_time and machine_id in active_machines:
                logger.info(f"\n Processing Machine {machine_id}...")
                sensor_data, original_datetime, original_machine_id = fetch_live_sensor_data(machine_id, latest_time)
                
                if sensor_data is not None and original_datetime is not None and original_machine_id is not None:
                    scaled_sensor_data = (sensor_data - means) / stds
                    
                    # Get ERP data for original machine and original datetime
                    current_erp_rows = erp_matrix[
                        (erp_matrix['datetime'] == original_datetime) & 
                        (erp_matrix['machineID'] == original_machine_id)
                    ]
                    logger.info(f"🔍 [Machine {machine_id}] ERP rows found: {len(current_erp_rows)}, machineID={original_machine_id}, datetime={original_datetime}")
                    
                    if len(current_erp_rows) > 0:
                        current_erp_row = current_erp_rows.iloc[-1:]
                        drop_cols = ['datetime', 'machineID', 'model', 'risk_comp1', 'risk_comp2', 'risk_comp3', 'risk_comp4']
                        erp_state = current_erp_row.drop(labels=drop_cols, axis=1).values.astype(float).reshape(1, -1)
                        
                        pred_alerts, hybrid_features = monitor.predict_live_status(scaled_sensor_data, erp_state)
                        
                        for _, comp in machine_comp_rows.iterrows():
                            comp_code = comp['comp_code'].lower()
                            if comp_code in pred_alerts:
                                risk = pred_alerts[comp_code]
                                alerts[comp_code] = risk
                                
                                if risk > max_risk:
                                    max_risk = risk
                                    worst_comp = comp_code
                        
                        # If danger, get SHAP explanation
                        if max_risk > 80 and monitor.models_loaded:
                            explanations = get_shap_explanation(worst_comp, hybrid_features, monitor)
                    else:
                        logger.warning(f"⚠️ [Machine {machine_id}] No ERP data found for machine {original_machine_id} at {original_datetime}")
                else:
                    logger.warning(f"⚠️ [Machine {machine_id}] No valid sensor data")
            else:
                logger.warning(f"⚠️ [Machine {machine_id}] Not active or no latest time")
            
            # If no alerts from AI, set default 0 risk
            if not alerts:
                logger.warning(f"⚠️ [Machine {machine_id}] No AI alerts, setting 0 risk")
                for _, comp in machine_comp_rows.iterrows():
                    comp_code = comp['comp_code'].lower()
                    alerts[comp_code] = 0.0
            
            # --- SMOOTH RISK PROGRESSION WITH WARNING STATE ---
            # Lưu trạng thái rủi ro trước đó để tạo sự chuyển tiếp mượt mà
            if machine_id not in previous_risk_state:
                previous_risk_state[machine_id] = {
                    "max_risk": 0.0,
                    "history": [],
                    "alert_level": "safe"  # safe, warning, danger
                }
            
            state = previous_risk_state[machine_id]
            state["history"].append(max_risk)
            if len(state["history"]) > RISK_TREND_WINDOW:
                state["history"].pop(0)
            
            # Tính xu hướng rủi ro (trung bình của các chu kỳ gần đây)
            if len(state["history"]) >= 2:
                trend = sum(state["history"][-2:]) / 2 - sum(state["history"][:2]) / 2
            else:
                trend = 0.0
            
            # Điều chỉnh max_risk để có trạng thái cảnh báo trước khi báo động
            if max_risk >= RISK_DANGER_THRESHOLD:
                # Máy ở trạng thái nguy hiểm - cho phép hiển thị đỏ
                state["alert_level"] = "danger"
                # Nếu đang tăng nhanh, tăng nhẹ để thể hiện sự cấp bách
                if trend > 5:
                    max_risk = min(100, max_risk * 1.05)
            elif max_risk >= RISK_WARNING_THRESHOLD:
                # Máy ở trạng thái cảnh báo - hiển thị vàng
                state["alert_level"] = "warning"
                # Đảm bảo giá trị nằm trong khoảng cảnh báo (50-80%)
                max_risk = max(RISK_WARNING_THRESHOLD, min(max_risk, RISK_DANGER_THRESHOLD - 1))
            else:
                # Máy khỏe mạnh - hiển thị xanh
                state["alert_level"] = "safe"
                # Nếu rủi ro đang tăng, tăng nhẹ để người dùng thấy xu hướng
                if trend > 2 and max_risk > 20:
                    max_risk = min(RISK_WARNING_THRESHOLD - 1, max_risk * 1.1)
            
            # Cập nhật trạng thái trước đó
            state["max_risk"] = max_risk
            
            logger.info(f"[Machine {machine_id}] Alerts: {alerts}, max_risk: {max_risk:.2f}%, trend: {trend:.2f}, level: {state['alert_level']}")
            
            if state["alert_level"] == "danger":
                danger_count += 1
            elif state["alert_level"] == "warning":
                warning_count += 1
            
            machine_data_list.append({
                "machine_id": machine_id,
                "machine_name": f"{machine_name} #{machine_id}",
                "location": location,
                "image": image_url,
                "status": status,
                "alerts": alerts,
                "worst_comp": worst_comp,
                "explanations": explanations,
                "load_percentage": load_percentage,
                "operation_level": operation_level
            })
        
        total = len(machine_data_list)
        safe_count = total - danger_count - warning_count
        
        return {
            "status": "success",
            "latest_time": latest_time.strftime('%Y-%m-%d %H:%M') if latest_time else pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
            "kpis": {"total": total, "safe": safe_count, "warn": warning_count, "danger": danger_count},
            "machines": sorted(machine_data_list, key=lambda x: x['machine_id'])
        }
        
    except Exception as e:
        print(f"Lỗi trong get_live_status: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.post("/api/reset-feeddata")
def reset_feeddata():
    global feed_process, feed_thread
    try:
        if feed_process:
            feed_process.terminate()
            feed_process = None
        if feed_thread:
            feed_thread = None

        with db_engine.connect() as conn:
            conn.execute(text("TRUNCATE TABLE telemetry_stream CASCADE;"))
            conn.execute(text("TRUNCATE TABLE machine_live_metrics CASCADE;"))
            conn.commit()
        return {"status": "success", "message": "Đã reset dữ liệu feeddata"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/start-feeddata")
def start_feeddata(speed: int = 2):
    global feed_process, feed_thread
    
    if feed_process:
        feed_process.terminate()
        feed_process = None
    if feed_thread:
        feed_thread = None
    
    try:
        with db_engine.connect() as conn:
            conn.execute(text("TRUNCATE TABLE telemetry_stream CASCADE;"))
            conn.execute(text("TRUNCATE TABLE machine_live_metrics CASCADE;"))
            conn.commit()
        
        script_path = os.path.join(PROJECT_ROOT, "demo", "data_pipeline", "live_data_feeder.py")
        feed_process = subprocess.Popen(
            [sys.executable, script_path],
            stdin=subprocess.PIPE,
            text=True
        )
        
        feed_process.stdin.write(f"{speed}\n")
        feed_process.stdin.flush()
        
        return {"status": "success", "message": f"Đã bắt đầu feed data với tốc độ {speed}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/stop-feeddata")
def stop_feeddata():
    global feed_process, feed_thread
    try:
        if feed_process:
            feed_process.terminate()
            feed_process = None
        if feed_thread:
            feed_thread = None
        return {"status": "success", "message": "Đã dừng feed data"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Run without reload to prevent constant restarts!
    uvicorn.run("backend_api:app", host="0.0.0.0", port=8000, reload=False)
