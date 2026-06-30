import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.mixed_precision import set_global_policy
from sklearn.preprocessing import StandardScaler
import joblib

# Thêm thư mục gốc của dự án vào PYTHONPATH để có thể import từ src.config một cách an toàn
# Bất kể bạn chạy file này từ thư mục nào, nó cũng sẽ tìm được config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.config import EXPERT_CONFIGS

# ==============================================================================
# 1. CẤU HÌNH MÔI TRƯỜNG & TỐI ƯU HÓA PHẦN CỨNG (GPU / VRAM)
# ==============================================================================
def setup_environment():
    """Tự động nhận diện chạy trên Local hay Google Colab để tối ưu GPU"""
    # Kích hoạt Mixed Precision (Dùng float16 thay vì float32) -> Tiết kiệm 50% VRAM GPU
    try:
        set_global_policy('mixed_float16')
        print("⚡ Kích hoạt Mixed Precision: Tối ưu hóa VRAM GPU thành công!")
    except Exception as e:
        print(f"Không thể bật Mixed Precision (Có thể do chạy CPU): {e}")

    import sys

    if 'google.colab' in sys.modules:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Colab detected")
    else:
        print("Local machine")

# ==============================================================================
# 2. XÂY DỰNG CLASS CHUYÊN GIA ROBOT (TÍCH HỢP CONFIG.PY)
# ==============================================================================
class WeldingRobotExpert:
    def __init__(self):
        # Lấy toàn bộ cấu hình từ config.py thay vì gõ cứng
        self.config = EXPERT_CONFIGS["Máy Hàn Robot Tự Động"]
        
        # Đường dẫn
        self.data_path = self.config["data_file"]
        self.save_dir = self.config["model_folder"]
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Hyperparameters (Tự động kéo từ config)
        self.time_steps = self.config["time_steps"]
        self.lookahead = self.config["lookahead_window"]
        self.batch_size = self.config["batch_size"]
        
        # Columns (Tự động kéo từ config)
        self.dynamic_cols = self.config["dynamic_cols"]
        self.static_cols = self.config["static_cols"]
        self.target_col = self.config["target_col"]
        
        # Scalers
        self.dynamic_scaler = StandardScaler()
        self.static_scaler = StandardScaler()

    def prepare_data(self):
        print(f"\n[1] Đang tải dữ liệu từ: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # TẠO NHÃN CỬA SỔ RỦI RO 48H (48H Risk Window Target Engineering)
        print(f"[2] Đang xử lý dán nhãn ngược (Backward Labeling) cho cửa sổ {self.lookahead} phút...")
        # Sử dụng self.target_col lấy từ config.py
        df['risk_48h'] = df[self.target_col].rolling(window=self.lookahead, min_periods=1).max().shift(-self.lookahead)
        df['risk_48h'] = df['risk_48h'].fillna(0)

        # Lọc bỏ cột Timestamp nếu có vì Mạng Nơ-ron không đọc được chuỗi
        if 'timestamp' in df.columns:
            df = df.drop(columns=['timestamp'])

        # Cắt tập Train (80%) và Validation (20%) CHUẨN TIME-SERIES (Không Shuffle)
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()

        # Chuẩn hóa dữ liệu chống Data Leakage
        print("[3] Đang chuẩn hóa (Scaling) và lưu Scaler...")
        train_df[self.dynamic_cols] = self.dynamic_scaler.fit_transform(train_df[self.dynamic_cols])
        val_df[self.dynamic_cols] = self.dynamic_scaler.transform(val_df[self.dynamic_cols])
        
        train_df[self.static_cols] = self.static_scaler.fit_transform(train_df[self.static_cols])
        val_df[self.static_cols] = self.static_scaler.transform(val_df[self.static_cols])
        
        joblib.dump(self.dynamic_scaler, os.path.join(self.save_dir, 'robot_dynamic_scaler.pkl'))
        joblib.dump(self.static_scaler, os.path.join(self.save_dir, 'robot_static_scaler.pkl'))

        # Sử dụng tf.keras.utils.timeseries_dataset_from_array để tối ưu RAM/VRAM
        print("[4] Đang tạo TensorFlow Datasets (Tối ưu I/O cho GPU)...")
        train_ds = self._create_tf_dataset(train_df, self.dynamic_cols, self.static_cols, 'risk_48h')
        val_ds = self._create_tf_dataset(val_df, self.dynamic_cols, self.static_cols, 'risk_48h')
        
        return train_ds, val_ds, len(self.dynamic_cols), len(self.static_cols)

    def _create_tf_dataset(self, df, dyn_cols, stat_cols, target_col):
        """Hàm sinh chuỗi Sliding Window siêu nhẹ cho RAM bằng Generator"""
        dyn_data = df[dyn_cols].values
        stat_data = df[stat_cols].values
        targets = df[target_col].values

        dyn_ds = tf.keras.utils.timeseries_dataset_from_array(
            data=dyn_data, targets=None, sequence_length=self.time_steps,
            sequence_stride=1, batch_size=self.batch_size
        )
        
        stat_ds = tf.data.Dataset.from_tensor_slices(stat_data[self.time_steps-1:]).batch(self.batch_size)
        target_ds = tf.data.Dataset.from_tensor_slices(targets[self.time_steps-1:]).batch(self.batch_size)

        input_ds = tf.data.Dataset.zip((dyn_ds, stat_ds))
        final_ds = tf.data.Dataset.zip((input_ds, target_ds)).cache().prefetch(tf.data.AUTOTUNE)
        return final_ds

    def build_hybrid_model(self, dyn_dim, stat_dim):
        """Khởi tạo Kiến trúc Mạng Nơ-ron Kép (Dual-Branch Network)"""
        print("\n[5] Đang khởi tạo Kiến trúc Hybrid (GRU + Dense)...")
        
        # Nhánh 1: Mạng hồi quy GRU cho chuỗi thời gian Động
        input_dyn = Input(shape=(self.time_steps, dyn_dim), name="Dynamic_Sensor_Input")
        x_dyn = GRU(64, return_sequences=True)(input_dyn)
        x_dyn = Dropout(0.2)(x_dyn)
        x_dyn = GRU(32)(x_dyn) 
        
        # Nhánh 2: Mạng Đa tầng (MLP) cho dữ liệu Tĩnh
        input_stat = Input(shape=(stat_dim,), name="Static_ERP_Input")
        x_stat = Dense(16, activation='relu')(input_stat)
        x_stat = BatchNormalization()(x_stat)
        
        # LỚP DUNG HỢP (Fusion Layer)
        concat = Concatenate()([x_dyn, x_stat])
        z = Dense(32, activation='relu')(concat)
        z = Dropout(0.2)(z)
        
        # Đầu ra dự báo
        output = Dense(1, activation='sigmoid', dtype='float32', name="Risk_Probability")(z)
        
        model = Model(inputs=[input_dyn, input_stat], outputs=output)
        
        optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
        
        return model

    def train(self):
        train_ds, val_ds, dyn_dim, stat_dim = self.prepare_data()
        model = self.build_hybrid_model(dyn_dim, stat_dim)
        
        print("\n[6] Cấu hình Kỹ thuật Huấn luyện (Callbacks)...")
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(self.save_dir, 'robot_hybrid_best.keras'),
                monitor='val_auc', mode='max', save_best_only=True, verbose=1
            ),
            EarlyStopping(monitor='val_auc', mode='max', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        ]
        
        print("\nBẮT ĐẦU HUẤN LUYỆN (TRAINING)...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=50,
            callbacks=callbacks
        )
        print("HUẤN LUYỆN HOÀN TẤT. Mô hình đã được lưu tại:", self.save_dir)
        return history

# ==============================================================================
# KHỞI CHẠY CHƯƠNG TRÌNH
# ==============================================================================
if __name__ == "__main__":
    try:
        # Cấu hình tự động
        setup_environment()
        
        # Khởi tạo chuyên gia (không cần truyền base_dir vì nó tự động đọc từ config)
        expert = WeldingRobotExpert()
        
        # Bắt đầu Train
        expert.train()
    except Exception as e:
        print(f"\nLỖI HỆ THỐNG: {e}")