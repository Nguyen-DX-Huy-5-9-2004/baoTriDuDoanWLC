'''import pandas as pd
import numpy as np
import os

class DynamicRiskWindowAnalyzer:
    def __init__(self, processed_dir):
        self.processed_dir = processed_dir
        
        # Cấu hình tần suất (phút) và cột nhãn mục tiêu thô của từng máy
        self.machine_configs = {
            "laser_final_features.csv": {
                "name": "Máy Cắt Laser",
                "freq_min": 60, # 1 giờ / dòng
                "targets": ['target_lens_failure', 'target_gantry_failure']
            },
            "punching_final_features.csv": {
                "name": "Máy Đột Dập",
                "freq_min": 15, # 15 phút / dòng
                "targets": ['target_wear']
            },
            "welding_robot_final_features.csv": {
                "name": "Robot Hàn",
                "freq_min": 15, # 15 phút / dòng
                "targets": ['target_failure_24h'] 
            },
            "press_brake_final_features.csv": {
                "name": "Máy Chấn Tôn",
                "freq_min": 1,  # 1 phút / dòng
                "targets": ['target_valve_condition', 'target_pump_leakage', 'target_accumulator_state']
            }
        }

    def _unify_target(self, df, config):
        """Hợp nhất các nhãn lỗi thô thành 1 nhãn nhị phân chung (0: Khỏe, 1: Lỗi)"""
        if config["name"] == "Máy Chấn Tôn":
            fail_valve = df['target_valve_condition'].apply(lambda x: 0 if x == 100 else 1)
            fail_pump = df['target_pump_leakage'].apply(lambda x: 0 if x == 0 else 1)
            fail_acc = df['target_accumulator_state'].apply(lambda x: 0 if x >= 115 else 1)
            return (fail_valve | fail_pump | fail_acc).astype(int)
        
        elif config["name"] == "Máy Cắt Laser":
            return (df['target_lens_failure'] | df['target_gantry_failure']).astype(int)
            
        elif config["name"] == "Máy Đột Dập":
            # Xử lý NaN của Đột dập
            df = df.dropna(subset=['target_wear'])
            return df['target_wear'].astype(int)
            
        elif config["name"] == "Robot Hàn":
            return df['target_failure_24h'].astype(int)

    def analyze_machine(self, filename, config):
        filepath = os.path.join(self.processed_dir, filename)
        if not os.path.exists(filepath):
            print(f"Không tìm thấy file: {filename}")
            return

        df = pd.read_csv(filepath)
        target_series = self._unify_target(df, config)
        
        # Các chỉ số cơ bản
        total_rows = len(target_series)
        freq_min = config['freq_min']
        total_hours = (total_rows * freq_min) / 60.0
        total_failures = target_series.sum()
        fail_ratio = total_failures / total_rows
        
        # Tính toán các chuỗi lỗi (Failure Events)
        # Tìm các điểm chuyển trạng thái (Từ 0 lên 1 là bắt đầu lỗi)
        diff = target_series.diff().fillna(0)
        failure_events = (diff == 1).sum()
        
        print(f"\nMÁY: {config['name'].upper()} ({filename})")
        print(f"  ├─ Tần suất lấy mẫu : {freq_min} phút / dòng")
        print(f"  ├─ Tổng thời gian   : {total_hours:.2f} giờ ({total_rows} dòng)")
        print(f"  ├─ Tỷ lệ lỗi thô    : {fail_ratio*100:.2f}%")
        
        if total_failures == 0:
            print("  └─ KHÔNG CÓ LỖI: Không thể tính toán cửa sổ rủi ro.")
            return
            
        if failure_events == 0 and total_failures > 0:
            # Lỗi ngay từ đầu và kéo dài mãi
            failure_events = 1
            
        mttr_hours = (total_failures * freq_min) / 60.0 / failure_events
        mtbf_hours = ((total_rows - total_failures) * freq_min) / 60.0 / failure_events if failure_events > 0 else 0
        
        print(f"  ├─ Số sự kiện lỗi   : {failure_events} lần")
        print(f"  ├─ MTTR (Thời gian lỗi trung bình/lần) : {mttr_hours:.2f} giờ")
        print(f"  ├─ MTBF (Thời gian chạy khỏe/lần)      : {mtbf_hours:.2f} giờ")
        # Nguyên tắc 1: Không vượt quá 15% MTBF (Tránh trùm lấp nhãn)
        # Nguyên tắc 2: Tối thiểu phải đủ 3-4 steps để con người kịp phản ứng
        
        max_safe_window = mtbf_hours * 0.15 
        min_action_window = (freq_min * 4) / 60.0 # Ít nhất 4 steps
        
        if total_hours < 50: # Dataset quá ngắn (như máy Chấn)
            optimal_window_hr = min(2.0, total_hours * 0.05) # Không quá 5% tổng thời lượng
        else:
            optimal_window_hr = max(min_action_window, min(max_safe_window, 24.0)) # Giới hạn tối đa 24h
        
        # Làm tròn thành số bước (steps)
        optimal_steps = max(1, int((optimal_window_hr * 60) / freq_min))
        actual_window_hr = (optimal_steps * freq_min) / 60.0

        print(f"  └─ ĐỀ XUẤT CỬA SỔ RỦI RO: {actual_window_hr:.1f} Giờ (Tương đương {optimal_steps} steps)")
        
        if actual_window_hr > mtbf_hours * 0.3:
            print("     Cảnh báo: Cửa sổ đề xuất khá sát với MTBF do dữ liệu quá ngắn.")

    def run(self):
        for filename, config in self.machine_configs.items():
            self.analyze_machine(filename, config)
        print("\n" + "="*80)
        print("💡 HƯỚNG DẪN SỬ DỤNG:")
        print("- Cập nhật số 'steps' đề xuất vào hàm process_* tương ứng trong file data_remediation_pipeline.py")
        print("="*80)

if __name__ == "__main__":
    # Điền đường dẫn tới thư mục chứa 4 file final_features.csv (chưa qua Ultimate)
    PROCESSED_DIR = "C:/Users/huynd1/Downloads/tBTDD/data/processed"
    
    analyzer = DynamicRiskWindowAnalyzer(processed_dir=PROCESSED_DIR)
    analyzer.run()'''
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

class ComprehensiveDatasetAnalyzer:
    def __init__(self, processed_dir):
        self.processed_dir = processed_dir
        print("="*95)
        print("🔍 HỆ THỐNG KIỂM TOÁN DỮ LIỆU & ĐỀ XUẤT CHIẾN THUẬT HỌC MÁY (ML STRATEGY)")
        print("="*95)
        
        # Cấu hình máy và các từ khóa để tự động nhận diện nhãn
        self.machine_configs = {
            "laser_final_features.csv": {
                "name": "Máy Cắt Laser",
                "freq_min": 60,
                "target_cols": ['target_lens_failure', 'target_gantry_failure']
            },
            "punching_final_features.csv": {
                "name": "Máy Đột Dập",
                "freq_min": 15,
                "target_cols": ['target_wear']
            },
            "welding_robot_final_features.csv": {
                "name": "Robot Hàn",
                "freq_min": 15,
                "target_cols": ['target_failure_24h']
            },
            "press_brake_final_features.csv": {
                "name": "Máy Chấn Tôn",
                "freq_min": 1,
                "target_cols": ['target_valve_condition', 'target_pump_leakage', 'target_accumulator_state']
            }
        }

    def _unify_target_for_analysis(self, df, config):
        """Hợp nhất nhãn để đánh giá tỷ lệ lỗi hệ thống"""
        name = config["name"]
        if name == "Máy Chấn Tôn":
            fail_valve = df['target_valve_condition'].apply(lambda x: 0 if x == 100 else 1)
            fail_pump = df['target_pump_leakage'].apply(lambda x: 0 if x == 0 else 1)
            fail_acc = df['target_accumulator_state'].apply(lambda x: 0 if x >= 115 else 1)
            return (fail_valve | fail_pump | fail_acc).astype(int)
        elif name == "Máy Cắt Laser":
            return (df['target_lens_failure'] | df['target_gantry_failure']).astype(int)
        elif name == "Máy Đột Dập":
            return df['target_wear'].dropna().astype(int)
        elif name == "Robot Hàn":
            return df['target_failure_24h'].dropna().astype(int)
        return pd.Series([0]*len(df))

    def profile_features(self, df):
        """Quét chi tiết dải giá trị và thống kê phân phối của các trường dữ liệu"""
        print("\n  📊 CHI TIẾT CÁC TRƯỜNG DỮ LIỆU (FEATURES):")
        cols_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            nulls = df[col].isnull().sum()
            unique_vals = df[col].nunique()
            
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val = round(df[col].min(), 2)
                max_val = round(df[col].max(), 2)
                mean_val = round(df[col].mean(), 2)
                std_val = round(df[col].std(), 2)
                
                # Tính độ lệch (skewness) để xem dữ liệu có phân phối chuẩn không
                skew_val = round(df[col].skew(), 2) if unique_vals > 1 else 0
                
                sample_vals = f"Min: {min_val:<6} | Max: {max_val:<6} | Mean: {mean_val:<6} | Std: {std_val:<5} | Skew: {skew_val}"
            else:
                sample_vals = f"Giá trị mẫu: {df[col].dropna().unique()[:3]}"
                
            cols_info.append({
                "Tên Cột": col[:25], # Cắt ngắn tên cột cho dễ nhìn
                "Kiểu": dtype,
                "NaN": nulls,
                "Duy nhất": unique_vals,
                "Thống kê (Min/Max/Mean/Std/Skew)": sample_vals
            })
            
        profile_df = pd.DataFrame(cols_info)
        if len(profile_df) > 12:
            display_df = pd.concat([profile_df.head(6), profile_df.tail(4)])
            print(display_df.to_string(index=False))
            print(f"  ... (Đã ẩn {len(profile_df) - 10} cột cảm biến khác)")
        else:
            print(profile_df.to_string(index=False))

    def analyze_labels_and_strategy(self, df, config, unified_target):
        """Đánh giá chất lượng nhãn và đề xuất chiến thuật Machine Learning"""
        print("\n  🎯 PHÂN TÍCH NHÃN & ĐỀ XUẤT CHIẾN THUẬT AI:")
        
        # In thông tin nhãn gốc
        targets = config['target_cols']
        for t in targets:
            if t in df.columns:
                nulls = df[t].isnull().sum()
                val_counts = df[t].dropna().value_counts().to_dict()
                print(f"    - Nhãn gốc [{t}]: Rỗng {nulls} dòng | Phân phối: {val_counts}")

        # Phân tích dựa trên nhãn đã hợp nhất (Lỗi vs Khỏe)
        total_valid = len(unified_target)
        if total_valid == 0:
            print("    ❌ KHÔNG CÓ DỮ LIỆU NHÃN HỢP LỆ!")
            return

        total_failures = unified_target.sum()
        total_healthy = total_valid - total_failures
        fail_ratio = total_failures / total_valid
        
        print(f"\n    >> Thống kê tổng quan (Hợp nhất): Mẫu Khỏe (0): {total_healthy} | Mẫu Lỗi (1): {total_failures}")
        print(f"    >> Tỷ lệ Lỗi (Positive Rate): {fail_ratio*100:.2f}%")

        print("\n    🧠 ĐỀ XUẤT CHIẾN THUẬT MÔ HÌNH HỌC MÁY:")
        
        # ĐÁNH GIÁ DỰA TRÊN SỐ LƯỢNG MẪU (SIZE)
        if total_valid < 3000:
            print("      ⚠️ DỮ LIỆU QUÁ ÍT (< 3000 mẫu):")
            print("         -> Tuyệt đối KHÔNG DÙNG Học Sâu (Deep Learning như GRU/LSTM) vì sẽ bị Overfit nặng.")
            print("         -> Bắt buộc sử dụng các mô hình Machine Learning dạng bảng (XGBoost, Random Forest, SVM).")
        else:
            print("      ✅ Dữ liệu đủ lớn để áp dụng các mô hình phức tạp (XGBoost mạnh, hoặc Shallow GRU nếu chuỗi dài).")

        # ĐÁNH GIÁ DỰA TRÊN PHÂN PHỐI NHÃN (DISTRIBUTION)
        if fail_ratio < 0.05: # Ít hơn 5% lỗi
            print("\n      🚨 MẤT CÂN BẰNG CỰC ĐOAN (Lỗi quá ít):")
            print("         -> CHIẾN THUẬT: HỌC KHÔNG GIÁM SÁT (Unsupervised Anomaly Detection).")
            print("         -> Giải pháp: Không dự đoán phân loại 0/1. Hãy huấn luyện mô hình Isolation Forest hoặc ")
            print("            Autoencoder CHỈ trên dữ liệu Khỏe (0). Khi máy có tín hiệu lạ, mô hình sẽ xuất ra ")
            print("            'Điểm bất thường' (Anomaly Score) tương đương với 'Xác suất xảy ra sự cố'.")
            
        elif fail_ratio > 0.85: # Hơn 85% lỗi
            print("\n      🚨 DỮ LIỆU BỊ NGHIÊNG VỀ TRẠNG THÁI LỖI (Nghi ngờ file log test-rig):")
            print("         -> CHIẾN THUẬT: TÁI CẤU TRÚC HOẶC HỌC GIÁM SÁT CÓ TRỌNG SỐ (Weighted Supervised Learning).")
            print("         -> Giải pháp: Dữ liệu này ghi nhận lúc máy hỏng liên tục. Cần Undersampling mạnh lớp Lỗi, ")
            print("            hoặc dùng XGBoost Regression để dự đoán RUL (Remaining Useful Life - Thời gian hữu ích còn lại).")
            
        else: # Tỷ lệ lỗi từ 5% đến 85%
            print("\n      🎯 PHÂN PHỐI ĐẸP CHO HỌC CÓ GIÁM SÁT:")
            print("         -> CHIẾN THUẬT: HỌC PHÂN LOẠI CÓ GIÁM SÁT (Supervised Classification).")
            print("         -> Giải pháp: Sử dụng XGBoost Classifier. Mô hình sẽ học cách phân tách ranh giới rõ ràng ")
            print("            giữa Khỏe và Lỗi. Dùng hàm predict_proba() để xuất ra xác suất rủi ro (Risk Probability %)")

    def analyze_machine(self, filename, config):
        filepath = os.path.join(self.processed_dir, filename)
        if not os.path.exists(filepath):
            print(f"\n⚠️ BỎ QUA: Không tìm thấy file {filename}")
            return

        print("\n" + "="*95)
        print(f"🏭 MÁY: {config['name'].upper()} | File: {filename}")
        print("="*95)
        
        df = pd.read_csv(filepath)
        print(f"  📌 KÍCH THƯỚC: {len(df)} dòng x {len(df.columns)} cột")
        
        # 1. Quét Đặc trưng (Dải giá trị, phân phối, nhiễu)
        self.profile_features(df)
        
        # 2. Phân tích Nhãn & Đề xuất Mô hình
        unified_target = self._unify_target_for_analysis(df, config)
        self.analyze_labels_and_strategy(df, config, unified_target)

    def run(self):
        for filename, config in self.machine_configs.items():
            self.analyze_machine(filename, config)
        print("\n" + "="*95)
        print("✅ KIỂM TOÁN HOÀN TẤT. Dựa vào báo cáo trên để chốt kiến trúc: XGBoost hay Isolation Forest!")

if __name__ == "__main__":
    # Tự động lấy đường dẫn của thư mục chứa file script này (thư mục 'processed')
    # Bằng cách này, dù bạn có đổi tên thư mục gốc thành tBTDDpause25_06 hay copy đi nơi khác, code vẫn chạy chuẩn.
    PROCESSED_DIR = os.path.dirname(os.path.abspath(__file__))
    
    analyzer = ComprehensiveDatasetAnalyzer(processed_dir=PROCESSED_DIR)
    analyzer.run()