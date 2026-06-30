# Chạy lệnh: streamlit run src/frontend_tv.py
import time
import requests
import streamlit as st

# --- CẤU HÌNH GIAO DIỆN STREAMLIT CHO TV ---
st.set_page_config(page_title="AI Predictive Maintenance TV", layout="wide", initial_sidebar_state="collapsed")

# API BACKEND URL
API_URL = "http://localhost:8000/api/live-status"
st.markdown("""
    <style>
    html, body, [class*="css"], .stApp {
        overflow: hidden !important; margin: 0 !important; padding: 0 !important; background-color: #000000 !important;
    }
    header, footer, #MainMenu { visibility: hidden !important; display: none !important; }
    
    .tv-dashboard-wrapper {
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background-color: #121212; z-index: 999999; padding: 2vh 2vw; box-sizing: border-box;
        display: flex; flex-direction: column;
    }

    .top-banner {
        display: flex; justify-content: space-between; align-items: center;
        background-color: #1a1a1a; border-radius: 1vh; border-bottom: 0.4vh solid #00c0f2;
        height: 10vh; padding: 0 2vw; margin-bottom: 2vh; box-shadow: 0.2vh 0.2vh 1vh rgba(0,0,0,0.5); flex-shrink: 0;
    }
    .banner-title { font-size: 3.5vh; font-weight: 900; color: #FFFFFF; white-space: nowrap; }
    .kpi-group { display: flex; gap: 1vw; align-items: center; }
    .kpi-item { font-size: 2.2vh; font-weight: bold; padding: 1vh 1vw; border-radius: 0.5vh; white-space: nowrap; }
    .kpi-time { color: #00c0f2; }
    .kpi-safe { color: #28a745; background: rgba(40, 167, 69, 0.15); border: 0.2vh solid #28a745; }
    .kpi-warn { color: #ffc107; background: rgba(255, 193, 7, 0.15); border: 0.2vh solid #ffc107; }
    .kpi-danger { color: #ff4b4b; background: rgba(255, 75, 75, 0.15); border: 0.2vh solid #ff4b4b; animation: text-blink 1s infinite; }
    
    .machine-grid {
        display: grid; grid-template-columns: repeat(5, 1fr); grid-template-rows: repeat(3, 1fr);
        grid-gap: 1.5vh 1vw; flex-grow: 1; width: 100%;
    }
    
    .machine-card {
        border-radius: 1vh; padding: 1vh 0.5vw; display: flex; flex-direction: column; justify-content: flex-start;
        color: white; background-color: #1e1e1e; box-shadow: 0.3vh 0.3vh 0.8vh rgba(0,0,0,0.5); overflow: hidden;
    }
    .title-text { font-size: 2.8vh; font-weight: 900; margin-bottom: 1.5vh; text-align: center; border-bottom: 0.2vh solid #333; padding-bottom: 0.5vh; white-space: nowrap;}
    .comp-row { display: flex; justify-content: space-between; margin-bottom: 1vh; font-size: 2.2vh; font-weight: bold; font-family: monospace; }
    
    .danger-mode { display: flex; flex-direction: column; height: 100%; justify-content: center; }
    .worst-comp-title { color: #ff4b4b; font-size: 2.6vh; font-weight: 900; text-align: center; margin-bottom: 1vh; animation: text-blink 1s infinite;}
    .shap-box { background: rgba(255,255,255,0.1); border-left: 0.5vh solid #ffc107; padding: 1vh; border-radius: 0.5vh; }
    .shap-reason { color: #ffffff; display: block; margin-bottom: 0.5vh; font-size: 1.8vh; line-height: 1.3; white-space: normal;}
    
    .status-normal { border: 0.3vh solid #28a745; }
    .status-warning { border: 0.3vh solid #ffc107; background-color: #2a2000; }
    .status-danger { border: 0.5vh solid #ff4b4b; background-color: #3a0000; animation: card-blink 1.5s infinite; }
    
    @keyframes card-blink { 0%, 100% { box-shadow: 0 0 1vh #ff4b4b; } 50% { box-shadow: 0 0 3vh #ff4b4b; border-color: #ff7676; } }
    @keyframes text-blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
    </style>
""", unsafe_allow_html=True)

# --- TRÌNH TẠO MÃ HTML ---
def get_status_style(prob):
    if prob > 80: return "#ff4b4b", "status-danger"
    elif prob > 40: return "#ffc107", "status-warning"
    elif prob > 15: return "#00c0f2", "status-normal"
    else: return "#28a745", "status-normal"

def generate_machine_card_html(machine_data):
    m_id = machine_data['machine_id']
    alerts = machine_data['alerts']
    max_risk = max(alerts.values())
    main_color, css_class = get_status_style(max_risk)
    
    html = f"<div class='machine-card {css_class}'>"
    html += f"<div class='title-text'>MÁY #{m_id:02d}</div>" 
    
    if max_risk > 80 and machine_data['explanations']:
        worst_comp = machine_data['worst_comp']
        html += "<div class='danger-mode'>"
        html += f"<div class='worst-comp-title'>LỖI {worst_comp.upper()} ({max_risk:.1f}%)</div>"
        html += "<div class='shap-box'>"
        html += "<span style='color:#ffc107; font-weight:900; font-size:2vh; display:block; margin-bottom:0.5vh;'>NGUYÊN NHÂN:</span>"
        for reason in machine_data['explanations']:
            html += f"<span class='shap-reason'>- {reason}</span>"
        html += "</div></div>"
    else:
        for comp, risk in alerts.items():
            color, _ = get_status_style(risk)
            html += f"<div class='comp-row'> <span>{comp.upper()}:</span><span style='color: {color};'>{risk:5.1f}%</span></div>"
    html += "</div>"
    return html

def generate_top_banner_html(latest_time, kpis):
    html = f"""
    <div class='top-banner'>
        <div style="display: flex; flex-direction: column; justify-content: center;">
            <div class='banner-title'>TRUNG TÂM GIÁM SÁT AI (PdM 4.0)</div>
            <div style="color: #aaaaaa; font-size: 1.8vh; margin-top: 0.5vh; font-weight: bold;"> Khả năng xảy ra sự cố trên máy tự động trong vòng 48h tiếp theo </div>
        </div>
        <div class='kpi-group'>
            <div class='kpi-item kpi-time'>🕒 {latest_time}</div>
            <div class='kpi-item kpi-safe'>✅ ỔN ĐỊNH: {kpis['safe']}/{kpis['total']}</div>
            <div class='kpi-item kpi-warn'>⚠️ CẢNH BÁO: {kpis['warn']}</div>
            <div class='kpi-item kpi-danger'>🚨 BÁO ĐỘNG: {kpis['danger']}</div>
        </div>
    </div>
    """
    return html

# --- VÒNG LẶP CHÍNH CỦA FRONTEND ---
def main():
    ui_placeholder = st.empty()
    
    while True:
        try:
            # Chỉ việc GỌI ĐIỆN đến Backend lấy dữ liệu JSON
            response = requests.get(API_URL, timeout=5)
            data = response.json()
            
            if data['status'] == "waiting_data":
                ui_placeholder.info("Đang chờ dữ liệu từ cảm biến...")
                time.sleep(1)
                continue
            
            if data['status'] == "error":
                ui_placeholder.error(f"Lỗi Backend: {data['message']}")
                time.sleep(2)
                continue
                
            # Đã có dữ liệu, tiến hành VẼ GIAO DIỆN
            full_html = "<div class='tv-dashboard-wrapper'>"
            full_html += generate_top_banner_html(data['latest_time'], data['kpis'])
            
            full_html += "<div class='machine-grid'>"
            for m_data in data['machines']:
                full_html += generate_machine_card_html(m_data)
            full_html += "</div></div>" 
            
            # Đẩy HTML lên Tivi
            ui_placeholder.markdown(full_html, unsafe_allow_html=True)
            
        except requests.exceptions.ConnectionError:
            ui_placeholder.warning("⚠️ Mất kết nối đến Máy chủ AI Backend. Vui lòng kiểm tra lại...")
            
        time.sleep(1) # Cứ 1 giây lại lên web hỏi xin dữ liệu 1 lần

if __name__ == "__main__":
    main()