#file: src/data_pipeline/time_series_generator.py dùng để tạo ra các cửa sổ trượt (sliding windows) từ dữ liệu cảm biến, đồng thời chuẩn hóa chúng để đưa vào mô hình GRU.
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class TelemetryDataset(Dataset):
    """
    Bộ Dataset chuẩn PyTorch: Chịu trách nhiệm cắt cửa sổ trượt (sliding windows)
    và cấp phát dữ liệu vào RAM/VRAM một cách tối ưu.
    """
    def __init__(self, telemetry_df, erp_matrix, seq_length=24):
        print("Đang đồng bộ và sắp xếp lại luồng thời gian...")
        # Đảm bảo dữ liệu được sắp xếp chuẩn xác theo máy và thời gian
        self.telemetry = telemetry_df.sort_values(['machineID', 'datetime']).reset_index(drop=True)
        self.erp_matrix = erp_matrix.sort_values(['machineID', 'datetime']).reset_index(drop=True)
        self.seq_length = seq_length
        self.features = ['volt', 'rotate', 'pressure', 'vibration']
        
        # [QUAN TRỌNG]: Lọc các chỉ mục hợp lệ (Valid Indices)
        # Máy phải có đủ 24h lịch sử phía trước thì mới có thể tạo thành 1 sequence.
        # Ta dùng hàm shift() để kiểm tra xem dòng thứ [i - 23] có cùng một machineID với dòng [i] hay không.
        self.telemetry['machine_shift'] = self.telemetry['machineID'].shift(self.seq_length - 1)
        self.valid_indices = self.telemetry[self.telemetry['machineID'] == self.telemetry['machine_shift']].index.values
        
        # Chuyển dataframe thành Numpy array để PyTorch truy xuất với tốc độ ánh sáng
        self.X_data = self.telemetry[self.features].values
        
        # Lấy nhãn rủi ro từ ma trận ERP (4 linh kiện)
        self.y_data = self.erp_matrix[['risk_comp1', 'risk_comp2', 'risk_comp3', 'risk_comp4']].values

    def __len__(self):
        # Trả về tổng số sequence (cửa sổ 24h) có thể tạo ra
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Lấy index thực tế trong mảng dữ liệu gốc
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.seq_length + 1
        
        # Cắt đoạn băng lịch sử cảm biến (24h)
        x_seq = self.X_data[start_idx : end_idx + 1]
        
        # Lấy nhãn rủi ro tương ứng ở thời điểm hiện tại (Cuối cửa sổ)
        y_label = self.y_data[end_idx]
        
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_label, dtype=torch.float32)


class TimeSeriesPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform_telemetry(self, telemetry_df):
        """Chuẩn hóa dữ liệu cảm biến (Standardization)"""
        print("Đang chuẩn hóa (Scaling) tín hiệu cảm biến...")
        df_scaled = telemetry_df.copy()
        features = ['volt', 'rotate', 'pressure', 'vibration']
        
        # Ép các dải sóng về trung bình = 0, độ lệch chuẩn = 1
        df_scaled[features] = self.scaler.fit_transform(df_scaled[features])
        return df_scaled

    def create_dataloaders(self, telemetry_df, erp_matrix, batch_size=256, seq_length=24):
        """Đóng gói dữ liệu thành các DataLoader để đưa vào luồng huấn luyện"""
        # 1. Chuẩn hóa Telemetry
        telemetry_scaled = self.fit_transform_telemetry(telemetry_df)
        
        # 2. Khởi tạo Dataset
        dataset = TelemetryDataset(telemetry_scaled, erp_matrix, seq_length)
        
        # 3. Tạo DataLoader (Tự động băm dữ liệu thành các batch nhỏ để nhồi vào card đồ họa)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        print(f" Đã khởi tạo DataLoader. Tổng số mẫu hợp lệ: {len(dataset)}")
        return dataloader


# ==========================================
# KHỐI TEST (Kiểm tra hình dáng của Tensor 3D)
# ==========================================
if __name__ == "__main__":
    from demo.data_pipeline.erp_feature_engineering import ERPFeatureEngineer
    
    DATA_PATH = r"C:\Users\huynd1\Downloads\tBTDD\data\raw\microsoft-azure-predictive-maintenance"
    print("--- CHẠY PIPELINE KIỂM THỬ ---")
    
    # 1. Tái tạo lại ma trận ERP từ file trước đó
    telemetry_raw = pd.read_csv(f"{DATA_PATH}/PdM_telemetry.csv", parse_dates=['datetime'])
    erp_engineer = ERPFeatureEngineer(DATA_PATH)
    erp_matrix = erp_engineer.execute_pipeline(telemetry_raw[['datetime', 'machineID']])
    
    # 2. Chạy luồng xử lý Time-Series mới
    preprocessor = TimeSeriesPreprocessor()
    dataloader = preprocessor.create_dataloaders(telemetry_raw, erp_matrix, batch_size=32, seq_length=24)
    
    # 3. Rút thử 1 Batch dữ liệu đầu tiên để kiểm tra
    x_batch, y_batch = next(iter(dataloader))
    
    print("\n[HÌNH DÁNG DỮ LIỆU ĐƯA VÀO GRU]")
    print(f"-> Tensor Đặc trưng (X): {x_batch.shape} (Batch Size, Sequence Length, Features)")
    print(f"-> Tensor Nhãn rủi ro (y): {y_batch.shape} (Batch Size, Target Components)")