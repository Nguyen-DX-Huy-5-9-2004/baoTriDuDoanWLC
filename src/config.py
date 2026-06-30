import os

# ==============================================================================
# 1. ĐỊNH TUYẾN THƯ MỤC CỐT LÕI (BASE PATHS)
# ==============================================================================
# Tự động lấy thư mục gốc của dự án (thư mục tBTDD)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'deployed_models')

# ==============================================================================
# 2. CẤU HÌNH CƠ SỞ DỮ LIỆU TIMESCALEDB
# ==============================================================================
DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "database": "weldcom_pdm",
    "user": "postgres",
    "password": "your_password" # Thay bằng pass thực tế
}

# ==============================================================================
# 3. BẢN ĐỒ QUY HOẠCH HỆ ĐA CHUYÊN GIA (EXPERT REGISTRY)
# ==============================================================================
# Đây là "linh hồn" của Router. Khi IoT gửi data kèm machine_type, 
# Router sẽ tra cứu Dictionary này để biết đẩy data cho AI nào xử lý.

EXPERT_CONFIGS = {
    # ---------------------------------------------------------
    # CHUYÊN GIA 1: MÁY HÀN ROBOT
    # ---------------------------------------------------------
    "Máy Hàn Robot Tự Động": {
        "model_folder": os.path.join(MODELS_DIR, 'robot_expert'),
        "data_file": os.path.join(DATA_PROCESSED_DIR, 'welding_robot_final_features.csv'),
        
        # Cấu trúc Data Input
        "dynamic_cols": ['robot_vibration_g', 'servo_motor_temp_c', 'robot_torque_proxy_amp', 'wire_feed_speed_mmin'],
        "static_cols": ['hours_since_maintenance'],
        "target_col": 'target_failure_24h',
        
        # Hyperparameters cho GRU + XGBoost
        "time_steps": 60,         # Dùng 60 phút quá khứ
        "lookahead_window": 2880, # 48 giờ * 60 phút (Cửa sổ 48h)
        "batch_size": 128,
        "risk_threshold": 75.0    # Ngưỡng % kích báo động đỏ
    },

    # ---------------------------------------------------------
    # CHUYÊN GIA 2: MÁY CẮT LASER
    # ---------------------------------------------------------
    "Máy Cắt Laser Fiber": {
        "model_folder": os.path.join(MODELS_DIR, 'laser_expert'),
        "data_file": os.path.join(DATA_PROCESSED_DIR, 'laser_final_features.csv'),
        
        "dynamic_cols": ['lens_temperature_c', 'laser_source_temp_c', 'delta_temperature', 'xy_axis_torque_nm', 'z_axis_tracking_error_proxy'],
        "static_cols": ['days_since_lens_inspect'],
        "target_col": 'target_lens_failure',
        
        "time_steps": 30,         # Máy laser biến thiên nhiệt nhanh, chỉ cần nhìn 30 phút quá khứ
        "lookahead_window": 2880, 
        "batch_size": 64,
        "risk_threshold": 80.0
    },

    # ---------------------------------------------------------
    # CHUYÊN GIA 3: MÁY CHẤN TÔN
    # ---------------------------------------------------------
    "Máy Chấn Tôn CNC": {
        "model_folder": os.path.join(MODELS_DIR, 'press_brake_expert'),
        "data_file": os.path.join(DATA_PROCESSED_DIR, 'press_brake_final_features.csv'),
        
        # Lấy đại diện vài cột từ 63 cột ép xung của file ETL
        "dynamic_cols": ['ps1_max', 'ps1_slope', 'eps1_max', 'ts1_mean', 'fs1_mean'], 
        "static_cols": ['days_since_oil_filter'],
        "target_col": 'target_valve_condition',
        
        "time_steps": 100,        # Học 100 chu kỳ ép (Cycle) quá khứ
        "lookahead_window": 2880, # Dự báo 2880 chu kỳ tới
        "batch_size": 64,
        "risk_threshold": 70.0
    },

    # ---------------------------------------------------------
    # CHUYÊN GIA 4: MÁY ĐỘT DẬP
    # ---------------------------------------------------------
    "Máy Đột Dập CNC": {
        "model_folder": os.path.join(MODELS_DIR, 'punching_expert'),
        "data_file": os.path.join(DATA_PROCESSED_DIR, 'punching_final_features.csv'),
        
        "dynamic_cols": ['CF_Feature_1', 'CF_Feature_2', 'Vib_Feature_1', 'Vib_Feature_2'],
        "static_cols": ['tool_hits_count'],
        "target_col": 'target_wear',
        
        "time_steps": 50,         # 50 nhát dập liên tiếp
        "lookahead_window": 1000, # Báo trước 1000 nhát dập
        "batch_size": 64,
        "risk_threshold": 85.0
    }
}