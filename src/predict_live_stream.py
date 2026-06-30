#& c:\Users\huynd1\Downloads\tBTDD\.venv\Scripts\python.exe c:/Users/huynd1/Downloads/tBTDD/src/predict_live_stream.py
import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
from sqlalchemy import create_engine, text

from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.console import Console
from rich import box

warnings.filterwarnings('ignore')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)
CORRECT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "microsoft-azure-predictive-maintenance")

try:
    from models.config import GRU_CONFIG, XGB_CONFIG
except ModuleNotFoundError:
    from config import GRU_CONFIG, XGB_CONFIG # type: ignore
from models.gru_extractor import GRURiskExtractor

MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "saved_models")
DB_URL = "postgresql://postgres:s@localhost:5432/predictive_maintenance"

console = Console()

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
        return alerts

def get_latest_timestamp():
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT MAX(datetime) FROM telemetry_live;")).scalar()
    return pd.to_datetime(result) if result else None

def fetch_live_sensor_data(machine_id, target_time):
    engine = create_engine(DB_URL)
    query = f"""
    SELECT volt, rotate, pressure, vibration 
    FROM telemetry_live 
    WHERE "machineID" = {machine_id} AND datetime <= '{target_time}'
    ORDER BY datetime DESC
    LIMIT 24;
    """
    with engine.connect() as conn:
        df_24h = pd.read_sql(text(query), conn)
        
    if len(df_24h) < 24: return None
    df_24h = df_24h.iloc[::-1].reset_index(drop=True)
    return df_24h.values

def format_risk_cell(prob):
    if prob > 80: return f"[bold white on red]{prob:5.1f}%[/]"
    elif prob > 40: return f"[bold yellow]{prob:5.1f}%[/]"
    elif prob > 15: return f"[bold cyan]{prob:5.1f}%[/]"
    else: return f"[green]{prob:5.1f}%[/]"

def get_overall_status(alerts):
    max_risk = max(alerts.values())
    if max_risk > 80: return "[bold red]🚨 BÁO ĐỘNG[/]"
    elif max_risk > 40: return "[bold yellow]⚠️ CẢNH BÁO[/]"
    elif max_risk > 15: return "[bold cyan]👀 THEO DÕI[/]"
    else: return "[bold green]✅ ỔN ĐỊNH[/]"

def generate_dashboard(latest_time, machine_data_dict):
    table = Table(show_header=True, header_style="bold magenta", expand=True, box=box.SIMPLE)
    table.add_column("MÁY", justify="center", style="cyan", no_wrap=True)
    table.add_column("COMP 1", justify="center", no_wrap=True)
    table.add_column("COMP 2", justify="center", no_wrap=True)
    table.add_column("COMP 3", justify="center", no_wrap=True)
    table.add_column("COMP 4", justify="center", no_wrap=True)
    table.add_column("TRẠNG THÁI", justify="center", no_wrap=True)

    if not machine_data_dict:
        table.add_row("...", "...", "...", "...", "...", "Đang kết nối Database...")
    else:
        for m_id, alerts in sorted(machine_data_dict.items()):
            table.add_row(
                f"#{m_id}",
                format_risk_cell(alerts['comp1']),
                format_risk_cell(alerts['comp2']),
                format_risk_cell(alerts['comp3']),
                format_risk_cell(alerts['comp4']),
                get_overall_status(alerts)
            )

    time_str = latest_time.strftime('%Y-%m-%d %H:%M:%S') if latest_time else "Đang khởi tạo kết nối..."
    panel = Panel(
        table,
        title=f"[bold green]🏭 TRUNG TÂM GIÁM SÁT AI (PdM)[/]",
        subtitle=f"[bold blue]🕒 Thời gian hệ thống: {time_str}[/]",
        border_style="green",
        expand=False
    )
    return panel

def start_factory_monitoring():
    os.system('cls' if os.name == 'nt' else 'clear')
    console.print("[bold yellow]Đang nạp não bộ AI và tải dữ liệu nền (ERP)...[/]")
    
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
    
    last_processed_time = None
    machine_data_dict = {}
    
    with Live(generate_dashboard(None, {}), refresh_per_second=4, console=console) as live:
        while True:
            try:
                latest_time = get_latest_timestamp()
                
                if latest_time is None or latest_time == last_processed_time:
                    time.sleep(0.5)
                    continue
                    
                last_processed_time = latest_time
                machine_data_dict.clear()
                
                engine = create_engine(DB_URL)
                with engine.connect() as conn:
                    machines_active = pd.read_sql(text(f"SELECT DISTINCT \"machineID\" FROM telemetry_live WHERE datetime = '{latest_time}'"), conn)['machineID'].tolist()
                    
                    # [SỬA LỖI]: Bốc thời gian gốc (2015) ra để lừa AI
                    orig_time_res = conn.execute(text(f"SELECT original_datetime FROM telemetry_live WHERE datetime = '{latest_time}' LIMIT 1")).scalar()
                    orig_time = pd.to_datetime(orig_time_res)
                
                for m_id in machines_active:
                    sensor_data = fetch_live_sensor_data(m_id, latest_time)
                    if sensor_data is None: continue
                        
                    scaled_sensor_data = (sensor_data - means) / stds
                    
                    # [SỬA LỖI]: Dùng thời gian 2015 để lục lại chính xác hồ sơ ERP của máy đó
                    current_erp_records = erp_matrix[(erp_matrix['datetime'] == orig_time) & (erp_matrix['machineID'] == m_id)]
                    if len(current_erp_records) == 0: continue
                    current_erp_record = current_erp_records.iloc[-1:]
                        
                    drop_cols = ['datetime', 'machineID', 'model', 'risk_comp1', 'risk_comp2', 'risk_comp3', 'risk_comp4']
                    erp_state = current_erp_record.drop(labels=drop_cols, axis=1).values.astype(float).reshape(1, -1)
                    
                    alerts = monitor.predict_live_status(scaled_sensor_data, erp_state)
                    machine_data_dict[m_id] = alerts
                
                live.update(generate_dashboard(latest_time, machine_data_dict))
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                time.sleep(2)

    console.print("\n[bold red][HỆ THỐNG] Đã tắt bộ phận trực ban.[/]")

if __name__ == "__main__":
    start_factory_monitoring()