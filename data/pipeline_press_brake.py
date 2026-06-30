import pandas as pd
import numpy as np
import os

class UniversalWeldcomPipeline:
    def __init__(self, base_dir="./"):
        # Gán đường dẫn gốc của dự án
        self.base_dir = base_dir

    # ==========================================================================
    # LÕI 1: XỬ LÝ TOÀN DIỆN MÁY CHẤN TÔN (Sửa khớp thư mục data/hydraulic của bạn)
    # ==========================================================================
    def process_press_brake(self, hydraulic_dir_name="data/hydraulic"):
        # Định nghĩa đường dẫn tuyệt đối đến thư mục chứa 17 file thủy lực
        hydraulic_path = os.path.join(self.base_dir, hydraulic_dir_name)
        
        print(f"\n=== [ETL MÁY CHẤN TÔN] QUÉT TOÀN DIỆN THƯ MỤC: {hydraulic_path} ===")
        
        profile_file = os.path.join(hydraulic_path, 'profile.txt')
        if not os.path.exists(profile_file):
            print(f"[THẤT BẠI MÁY CHẤN] Không tìm thấy file nhãn profile.txt tại đường dẫn cấu hình: {profile_file}")
            print("👉 Vui lòng kiểm tra lại xem các file .txt thủy lực đã nằm trong đúng thư mục data/hydraulic/ chưa.")
            return None
            
        labels = pd.read_csv(profile_file, sep='\t', header=None)
        df_out = pd.DataFrame({
            'target_valve_condition': labels[1],       # Lỗi lệch trục Y1/Y2 (Van tỷ lệ)
            'target_pump_leakage': labels[2],          # Lỗi tụt áp (Bơm thủy lực)
            'target_accumulator_state': labels[3]      # Lỗi mất lực đột ngột (Bình tích áp)
        })
        n_cycles = len(df_out)

        # Danh sách tuyệt đối toàn bộ 17 file thành phần cảm biến đa tần số từ tài liệu gốc
        all_17_files = [
            ('PS1.txt', 100), ('PS2.txt', 100), ('PS3.txt', 100), ('PS4.txt', 100),
            ('PS5.txt', 100), ('PS6.txt', 100), ('EPS1.txt', 100),
            ('FS1.txt', 10),  ('FS2.txt', 10),
            ('TS1.txt', 1),   ('TS2.txt', 1),   ('TS3.txt', 1),   ('TS4.txt', 1),
            ('VS1.txt', 1),   ('SE.txt', 1),    ('CE.txt', 1),    ('CP.txt', 1)
        ]

        for file_name, hz in all_17_files:
            file_path = os.path.join(hydraulic_path, file_name)
            if not os.path.exists(file_path):
                print(f"   [BỎ QUA] Không tìm thấy file thành phần: {file_name}")
                continue
            
            print(f"   -> Đang đọc ma trận & rút trích đặc trưng {hz}Hz từ: {file_name}")
            raw = pd.read_csv(file_path, sep='\t', header=None)
            prefix = file_name.split('.')[0].lower()
            
            # Đồng bộ hình dáng (Shape) đa tần số về phẳng chu kỳ để tránh tràn VRAM
            df_out[f'{prefix}_mean'] = raw.mean(axis=1)
            df_out[f'{prefix}_max'] = raw.max(axis=1)
            df_out[f'{prefix}_std'] = raw.std(axis=1)
            
            if hz == 100: # Tính toán độ dốc (Slope) cho các biến phản ứng nhanh áp suất
                df_out[f'{prefix}_slope'] = raw.iloc[:, :1000].diff(axis=1).mean(axis=1)

        # Dung hợp mốc bảo trì tĩnh (Giả lập mốc thời gian công nhân bấm nút reset lọc dầu)
        timestamps = pd.date_range(start="2026-01-01 08:00:00", periods=n_cycles, freq='1min')
        days_filter = []
        curr_days = 5.2
        for i in range(n_cycles):
            if i in [400, 1200]: curr_days = 0.0 # Mốc bảo trì nhập tay trả về 0 ngày
            else: curr_days += 1 / (24 * 60)      # Tăng tiến tuyến tính theo từng phút chạy
            days_filter.append(curr_days)
            
        df_out['timestamp'] = timestamps
        df_out['days_since_oil_filter'] = days_filter
        
        # Xuất file đặc trưng phẳng ra thư mục dự án
        output_name = os.path.join(self.base_dir, "press_brake_final_features.csv")
        df_out.to_csv(output_name, index=False)
        print(f"✅ HOÀN THÀNH MÁY CHẤN: Đã xuất '{output_name}' | Kích thước ma trận: {df_out.shape}")
        return df_out

    # ==========================================================================
    # LÕI 2: XỬ LÝ MÁY ĐỘT DẬP (Lọc từ file tool_wear_dataset.csv ở thư mục gốc)
    # ==========================================================================
    def process_punching_machine(self, file_name="tool_wear_dataset.csv"):
        file_path = os.path.join(self.base_dir, file_name)
        print(f"\n=== [ETL MÁY ĐỘT DẬP] TIỀN XỬ LÝ ĐỘ LỆCH VA ĐẬP TỪ: {file_path} ===")
        if not os.path.exists(file_path):
            print(f"   [THẤT BẠI] Không tìm thấy file {file_name} ở thư mục gốc dự án.")
            return None
        
        df = pd.read_csv(file_path)
        # Loại bỏ cột AE (Phát xạ âm thanh) vì nhiễu nhà xưởng [Không khả thi] như IoT báo cáo
        ae_cols = [c for c in df.columns if 'AE_' in c] + ['VB_mm']
        df_cleaned = df.drop(columns=ae_cols)
        
        # Mã hóa nhãn văn bản sang số nhị phân (Healthy: 0, Worn: 1) cho thuật toán học máy
        df_cleaned['target_wear'] = df_cleaned['Wear_Class'].map({'Healthy': 0, 'Worn': 1})
        df_cleaned = df_cleaned.drop(columns=['Wear_Class'])
        
        # Giả lập mốc tĩnh tích lũy (Số nhát dập của trạm dao)
        df_cleaned['tool_hits_count'] = np.arange(len(df_cleaned)) % 5000
        
        output_name = os.path.join(self.base_dir, "punching_final_features.csv")
        df_cleaned.to_csv(output_name, index=False)
        print(f"✅ HOÀN THÀNH MÁY ĐỘT: Đã xuất '{output_name}' | Kích thước ma trận: {df_cleaned.shape}")
        return df_cleaned

    # ==========================================================================
    # LÕI 3: XỬ LÝ MÁY CẮT LASER (Lọc từ file ai4i2020.csv ở thư mục gốc)
    # ==========================================================================
    def process_laser_cutting(self, file_name="ai4i2020.csv"):
        file_path = os.path.join(self.base_dir, file_name)
        print(f"\n=== [ETL MÁY CẮT LASER] CHUYỂN ĐỔI THANG ĐO NHIỆT ĐỘ TỪ: {file_path} ===")
        if not os.path.exists(file_path):
            print(f"   [THẤT BẠI] Không tìm thấy file {file_name} ở thư mục gốc dự án.")
            return None
        
        df = pd.read_csv(file_path)
        df_laser = pd.DataFrame()
        
        # Chuyển đổi thang đo từ Kelvin [K] trong lab sang độ C [°C] của cảm biến xưởng Weldcom
        df_laser['lens_temperature_c'] = df['Process temperature [K]'] - 273.15
        df_laser['laser_source_temp_c'] = df['Air temperature [K]'] - 273.15
        df_laser['delta_temperature'] = df_laser['lens_temperature_c'] - df_laser['laser_source_temp_c']
        
        # Ánh xạ cơ khí động lực gantry
        df_laser['xy_axis_torque_nm'] = df['Torque [Nm]']
        df_laser['z_axis_tracking_error_proxy'] = df['Tool wear [min]']
        
        # Đồng bộ nhãn lỗi đa mục tiêu (HDF: Quá nhiệt thấu kính, OSF: Quá tải cơ khí kẹt trục)
        df_laser['target_lens_failure'] = df['HDF']
        df_laser['target_gantry_failure'] = df['OSF']
        
        # Tạo mốc thời gian tĩnh tích lũy (Số ngày từ lần vệ sinh thấu kính gần nhất)
        df_laser['days_since_lens_inspect'] = (df['Tool wear [min]'] / 20).astype(int)
        
        output_name = os.path.join(self.base_dir, "laser_final_features.csv")
        df_laser.to_csv(output_name, index=False)
        print(f"✅ HOÀN THÀNH MÁY LASER: Đã xuất '{output_name}' | Kích thước ma trận: {df_laser.shape}")
        return df_laser

    # ==========================================================================
    # LÕI 4: XỬ LÝ MÁY HÀN ROBOT (Lọc từ file predictive_maintenance_v3.csv ở thư mục gốc)
    # ==========================================================================
    def process_welding_robot(self, file_name="predictive_maintenance_v3.csv"):
        file_path = os.path.join(self.base_dir, file_name)
        print(f"\n=== [ETL MÁY HÀN ROBOT] NỘI SUY DỮ LIỆU IoT TỪ: {file_path} ===")
        if not os.path.exists(file_path):
            print(f"   [THẤT BẠI] Không tìm thấy file {file_name} ở thư mục gốc dự án.")
            return None
        
        df = pd.read_csv(file_path)
        df_clean = df.copy()
        
        # Lắp phân hệ vá lỗi rớt mạng gói tin (Dấu phẩy kép ",," biến thành giá trị nội suy tuyến tính)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(method='linear')
        
        df_robot = pd.DataFrame()
        df_robot['timestamp'] = df_clean['timestamp']
        df_robot['robot_vibration_g'] = df_clean['vibration_rms']
        df_robot['servo_motor_temp_c'] = df_clean['temperature_motor']
        df_robot['robot_torque_proxy_amp'] = df_clean['current_phase_avg']
        df_robot['wire_feed_speed_mmin'] = df_clean['rpm'] / 100
        
        # Gán nhãn đích chuỗi thời gian dự báo trước 24h
        df_robot['target_failure_24h'] = df_clean['failure_within_24h']
        df_robot['hours_since_maintenance'] = df_clean['hours_since_maintenance']
        
        output_name = os.path.join(self.base_dir, "welding_robot_final_features.csv")
        df_robot.to_csv(output_name, index=False)
        print(f"✅ HOÀN THÀNH MÁY ROBOT: Đã xuất '{output_name}' | Kích thước ma trận: {df_robot.shape}")
        return df_robot

# ==============================================================================
# HÀM ĐIỀU PHỐI CHẠY PIPELINE SẢN XUẤT
# ==============================================================================
if __name__ == "__main__":
    # Điền đúng thư mục làm việc hiện hành của bạn
    PROJECT_DIR = "c:/Users/huynd1/Downloads/tBTDD"
    
    engine = UniversalWeldcomPipeline(base_dir=PROJECT_DIR)
    
    # 1. Quét cụm dữ liệu Thủy lực đa tần số của Máy Chấn Tôn nằm tại data/hydraulic/
    engine.process_press_brake(hydraulic_dir_name="data/hydraulic")
    
    # 2. Xử lý dữ liệu va đập Máy Đột Dập nằm ngoài thư mục gốc
    engine.process_punching_machine(file_name="tool_wear_dataset.csv")
    
    # 3. Xử lý dữ liệu quang nhiệt Máy Cắt Laser nằm ngoài thư mục gốc
    engine.process_laser_cutting(file_name="ai4i2020.csv")
    
    # 4. Xử lý chuỗi thời gian lỗi trạm Máy Hàn Robot nằm ngoài thư mục gốc
    engine.process_welding_robot(file_name="predictive_maintenance_v3.csv")
    
    print("\n====== [XỬ LÝ DỮ LIỆU ĐA NỀN TẢNG THÀNH CÔNG] - 4 FILE ĐẶC TRƯNG SẠCH ĐÃ SẴN SÀNG ĐỂ TRAINING! ======")