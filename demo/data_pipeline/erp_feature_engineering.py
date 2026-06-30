#file: src/data_pipeline/erp_feature_engineering.py là nơi chúng ta sẽ thực hiện việc gộp dữ liệu từ 4 file CSV ERP (Errors, Maintenance, Failures, Machines) để tạo ra một ma trận đặc trưng tĩnh (Static Feature Matrix) phục vụ cho việc huấn luyện mô hình dự đoán rủi ro hỏng hóc của máy móc trong tương lai.
# Mỗi dòng trong ma trận này sẽ đại diện cho một thời điểm cụ thể của một máy, với các đặc trưng được trích xuất từ dữ liệu ERP và nhãn rủi ro tương ứng. Đây là bước quan trọng để biến dữ liệu thô thành một định dạng có thể sử dụng được cho mô hình học máy. mỗi mẻ (batch) sẽ đưa 32 đoạn băng lịch sử vào card đồ họa, mỗi đoạn băng dài 24 giờ, và chứa 4 cảm biến (điện áp, độ rung, áp suất, vòng quay)
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore') # Tắt các cảnh báo tương thích của Pandas

class ERPFeatureEngineer:
    def __init__(self, data_dir):
        """Khởi tạo với đường dẫn tới thư mục chứa dữ liệu raw Azure."""
        self.data_dir = data_dir
        
    def load_data(self):
        """Tải 4 file dữ liệu ERP cốt lõi."""
        print("Đang tải dữ liệu ERP...")
        self.errors = pd.read_csv(f'{self.data_dir}/PdM_errors.csv', parse_dates=['datetime'])
        self.maint = pd.read_csv(f'{self.data_dir}/PdM_maint.csv', parse_dates=['datetime'])
        self.failures = pd.read_csv(f'{self.data_dir}/PdM_failures.csv', parse_dates=['datetime'])
        self.machines = pd.read_csv(f'{self.data_dir}/PdM_machines.csv')

    def build_error_features(self, telemetry_dates):
        print("Đang xử lý Đặc trưng Tích lũy Lỗi (24h)...")

        # Đổi 'error' thành 'errorID' cho khớp với file CSV. 
        # prefix='' và prefix_sep='' giúp tạo thẳng tên cột là 'error1', 'error2' thay vì 'errorID_error1'
        error_count = pd.get_dummies(self.errors, columns=['errorID'], prefix='', prefix_sep='')

        # Nhóm theo máy và thời gian để tính tổng
        error_count = error_count.groupby(['machineID', 'datetime']).sum().reset_index()

        # Merge với telemetry_dates (khung xương thời gian)
        error_feat = telemetry_dates.merge(
            error_count,
            on=['machineID', 'datetime'],
            how='left'
        ).fillna(0.0)

        # Tính tổng trượt (rolling sum) trong 24 giờ
        error_feat = error_feat.sort_values(['machineID', 'datetime']).reset_index(drop=True)
        
        # Chỉ áp dụng rolling cho các cột lỗi, giữ nguyên cột machineID và datetime
        error_cols = [c for c in error_feat.columns if str(c).startswith('error')]
        error_feat[error_cols] = (
            error_feat.groupby('machineID')[error_cols]
            .rolling(window=24, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        return error_feat

    def build_maintenance_features(self, telemetry_dates):
        print("Đang xử lý Đặc trưng Vòng đời Linh kiện (Time-Since-Last-Event)...")
        # Sử dụng đúng prefix để tên cột tự sinh ra là comp1, comp2, comp3, comp4
        comp_rep = pd.get_dummies(self.maint, columns=['comp'], prefix='', prefix_sep='')
        comp_rep = comp_rep.groupby(['machineID', 'datetime']).sum().reset_index()
        
        # Merge với khung thời gian chuẩn
        comp_rep = telemetry_dates.merge(
            comp_rep, 
            on=['datetime', 'machineID'], 
            how='outer'
        ).fillna(0).sort_values(by=['machineID', 'datetime'])

        # Kéo giãn mốc thời gian bảo trì và tính số ngày
        for comp in ['comp1', 'comp2', 'comp3', 'comp4']:
            if comp not in comp_rep.columns:
                comp_rep[comp] = 0.0 # Phòng hờ trường hợp dữ liệu khuyết thiếu linh kiện
                
            # [SỬA LỖI PANDAS 2.0+]: Dùng np.where để tạo mảng datetime an toàn
            # Nếu có thay linh kiện (>=1), lưu lại datetime, ngược lại gán NaT (Not a Time)
            replacement_dates = np.where(comp_rep[comp] >= 1, comp_rep['datetime'], pd.NaT)
            comp_rep[comp] = pd.to_datetime(replacement_dates)
            
            # Forward fill mốc thời gian bảo trì cho các dòng sau đó
            comp_rep[comp] = comp_rep.groupby('machineID')[comp].ffill()
            
            # Tính số ngày an toàn bằng phương thức .dt của Pandas
            comp_rep[comp] = (comp_rep['datetime'] - comp_rep[comp]).dt.total_seconds() / 86400.0
            
        return comp_rep.fillna(0) # Điền 0 nếu linh kiện chưa từng được thay thế

    def build_risk_window_labels(self, telemetry_dates, risk_window_hours=48):
        """
        Dán nhãn cho Cửa sổ Rủi ro Động: 
        1 nếu máy sẽ hỏng bộ phận đó trong vòng `risk_window_hours` tới, ngược lại là 0.
        """
        print(f"Đang dán nhãn Cửa sổ Rủi ro ({risk_window_hours}h)...")
        labels = telemetry_dates.copy()
        
        # Khởi tạo cột nhãn rủi ro cho 4 linh kiện
        for comp in ['comp1', 'comp2', 'comp3', 'comp4']:
            labels[f'risk_{comp}'] = 0
            
            # Lọc ra các mốc hỏng hóc của linh kiện hiện tại
            comp_failures = self.failures[self.failures['failure'] == comp]
            
            for _, row in comp_failures.iterrows():
                m_id = row['machineID']
                f_time = row['datetime']
                
                # Cửa sổ rủi ro: [Thời điểm hỏng - Risk Window, Thời điểm hỏng]
                mask = (
                    (labels['machineID'] == m_id) & 
                    (labels['datetime'] >= f_time - pd.Timedelta(hours=risk_window_hours)) & 
                    (labels['datetime'] <= f_time)
                )
                labels.loc[mask, f'risk_{comp}'] = 1
                
        return labels

    def execute_pipeline(self, telemetry_dates):
        """Thực thi toàn bộ pipeline gộp dữ liệu ERP."""
        self.load_data()
        
        feat_errors = self.build_error_features(telemetry_dates)
        feat_maint = self.build_maintenance_features(telemetry_dates)
        labels = self.build_risk_window_labels(telemetry_dates, risk_window_hours=48)
        
        # Hợp nhất với thông số tĩnh (Age) của máy
        final_erp = feat_errors.merge(feat_maint, on=['datetime', 'machineID'], how='inner')
        final_erp = final_erp.merge(labels, on=['datetime', 'machineID'], how='inner')
        final_erp = final_erp.merge(self.machines, on=['machineID'], how='left')
        
        print(f" Đã trích xuất xong Ma trận Đặc trưng Tĩnh: {final_erp.shape}")
        return final_erp


# Khối test (Chỉ chạy khi bạn gọi trực tiếp file này)
if __name__ == "__main__":
    # Đường dẫn trỏ tới thư mục chứa 5 file CSV Azure
    DATA_PATH = r"C:\Users\huynd1\Downloads\tBTDD\data\raw\microsoft-azure-predictive-maintenance"
    
    # Tạo một khung thời gian giả lập từ file telemetry để làm "xương sống" nối các bảng
    print("Tải khung thời gian gốc...")
    telemetry_base = pd.read_csv(f"{DATA_PATH}/PdM_telemetry.csv", usecols=['datetime', 'machineID'], parse_dates=['datetime'])
    
    # Khởi chạy Pipeline
    engineer = ERPFeatureEngineer(DATA_PATH)
    erp_matrix = engineer.execute_pipeline(telemetry_base)
    
    print("\n[THÀNH CÔNG] - 5 dòng đầu tiên của dữ liệu đã gộp:")
    print(erp_matrix.head())