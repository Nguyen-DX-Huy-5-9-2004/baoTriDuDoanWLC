import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Concatenate, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.mixed_precision import set_global_policy
from sklearn.preprocessing import StandardScaler
import joblib

# Thêm thư mục gốc để import config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.config import EXPERT_CONFIGS

# ==============================================================================
# 1. CẤU HÌNH MÔI TRƯỜNG & TỐI ƯU HÓA PHẦN CỨNG
# ==============================================================================
def setup_environment():
    try:
        set_global_policy('mixed_float16')
        print("Kích hoạt Mixed Precision cho Press Brake Expert thành công!")
    except Exception as e:
        print(f"Không thể bật Mixed Precision: {e}")

    import sys

    if 'google.colab' in sys.modules:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Colab detected")
    else:
        print("Local machine")

# ==============================================================================
# 2. XÂY DỰNG CLASS CHUYÊN GIA MÁY CHẤN TÔN (HYDRAULIC EXPERT)
# ==============================================================================
class PressBrakeExpert:
    def __init__(self):
        self.config = EXPERT_CONFIGS["Máy Chấn Tôn CNC"]
        
        self.data_path = self.config["data_file"]
        self.save_dir = self.config["model_folder"]
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.time_steps = self.config["time_steps"]
        self.lookahead = self.config["lookahead_window"]
        self.batch_size = self.config["batch_size"]
        
        self.dynamic_cols = self.config["dynamic_cols"]
        self.static_cols = self.config["static_cols"]
        self.target_col = self.config["target_col"]
        
        self.dynamic_scaler = StandardScaler()
        self.static_scaler = StandardScaler()

    def prepare_data(self):
        print(f"\n[1] Đang tải dữ liệu Máy Chấn Tôn từ: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # ----------------------------------------------------------------------
        # KỸ THUẬT CỐT LÕI: TARGET BINARIZATION (CHUYỂN ĐỔI NHÃN RỦI RO)
        # ----------------------------------------------------------------------
        print("[1.5] Đang xử lý Nhãn: Chuyển đổi trạng thái Van sang chuẩn Rủi ro Nhị phân...")
        # Nhãn gốc target_valve_condition có các giá trị: 100 (Tốt), 90, 80, 73 (Kém dần)
        # Quy chuẩn: Cứ dưới 100 là máy bắt đầu có rủi ro suy thoái (is_degrading = 1)
        df['is_degrading'] = (df[self.target_col] < 100).astype(int)

        print(f"[2] Đang dán nhãn ngược 48H (lookahead = {self.lookahead} chu kỳ chấn)...")
        # Nhìn trước 2880 chu kỳ, nếu có dấu hiệu suy thoái -> Cảnh báo ngay từ hiện tại
        df['risk_48h'] = df['is_degrading'].rolling(window=self.lookahead, min_periods=1).max().shift(-self.lookahead)
        df['risk_48h'] = df['risk_48h'].fillna(0)

        # Nếu bạn muốn tự động vét toàn bộ 63 cột thay vì cấu hình cứng trong config.py:
        # self.dynamic_cols = [c for c in df.columns if c not in self.static_cols + [self.target_col, 'timestamp', 'is_degrading', 'risk_48h']]

        if 'timestamp' in df.columns:
            df = df.drop(columns=['timestamp'])

        # Cắt tập dữ liệu theo trình tự thời gian (Không xáo trộn)
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()

        print("[3] Đang chuẩn hóa (Scaling) các cột đặc trưng Thủy lực đa chiều...")
        train_df[self.dynamic_cols] = self.dynamic_scaler.fit_transform(train_df[self.dynamic_cols])
        val_df[self.dynamic_cols] = self.dynamic_scaler.transform(val_df[self.dynamic_cols])
        
        train_df[self.static_cols] = self.static_scaler.fit_transform(train_df[self.static_cols])
        val_df[self.static_cols] = self.static_scaler.transform(val_df[self.static_cols])
        
        joblib.dump(self.dynamic_scaler, os.path.join(self.save_dir, 'press_brake_dynamic_scaler.pkl'))
        joblib.dump(self.static_scaler, os.path.join(self.save_dir, 'press_brake_static_scaler.pkl'))

        print("[4] Đang đóng gói dữ liệu thành Tensor 3D cho Mạng lai...")
        train_ds = self._create_tf_dataset(train_df, self.dynamic_cols, self.static_cols, 'risk_48h')
        val_ds = self._create_tf_dataset(val_df, self.dynamic_cols, self.static_cols, 'risk_48h')
        
        return train_ds, val_ds, len(self.dynamic_cols), len(self.static_cols)

    def _create_tf_dataset(self, df, dyn_cols, stat_cols, target_col):
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
        print("\n[5] Đang khởi tạo Kiến trúc Hybrid Nâng cao (Conv1D + GRU + Dense)...")
        
        # Nhánh 1: Mạng Xử lý Động lực học (Bổ sung Conv1D lọc nhiễu không gian)
        input_dyn = Input(shape=(self.time_steps, dyn_dim), name="Hydraulic_Dynamic_Input")
        
        # VŨ KHÍ RIÊNG CHO MÁY CHẤN: Dùng Conv1D để trích xuất đặc trưng hình học từ nhiều cột cảm biến
        x_dyn = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_dyn)
        x_dyn = MaxPooling1D(pool_size=2)(x_dyn)
        
        x_dyn = GRU(64, return_sequences=True)(x_dyn)
        x_dyn = Dropout(0.3)(x_dyn)
        x_dyn = GRU(32)(x_dyn) 
        
        # Nhánh 2: Mạng Xử lý Tuổi thọ ERP (Tĩnh)
        input_stat = Input(shape=(stat_dim,), name="Hydraulic_Static_Input")
        x_stat = Dense(16, activation='relu')(input_stat)
        x_stat = BatchNormalization()(x_stat)
        
        # Lớp Dung hợp
        concat = Concatenate()([x_dyn, x_stat])
        z = Dense(64, activation='relu')(concat) # Tăng số node Dense vì dữ liệu máy chấn phức tạp hơn
        z = Dropout(0.3)(z)
        z = Dense(16, activation='relu')(z)
        
        output = Dense(1, activation='sigmoid', dtype='float32', name="Press_Brake_Risk_Probability")(z)
        
        model = Model(inputs=[input_dyn, input_stat], outputs=output)
        optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
        
        return model

    def train(self):
        train_ds, val_ds, dyn_dim, stat_dim = self.prepare_data()
        model = self.build_hybrid_model(dyn_dim, stat_dim)
        
        print("\n[6] Cấu hình Callbacks Máy Chấn Tôn...")
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(self.save_dir, 'press_brake_hybrid_best.keras'),
                monitor='val_auc', mode='max', save_best_only=True, verbose=1
            ),
            EarlyStopping(monitor='val_auc', mode='max', patience=12, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        ]
        
        print("\nBẮT ĐẦU HUẤN LUYỆN CHUYÊN GIA MÁY CHẤN TÔN...")
        history = model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=callbacks)
        print("HUẤN LUYỆN HOÀN TẤT. Mô hình CHẤN TÔN đã được lưu tại:", self.save_dir)
        return history

# ==============================================================================
# KHỞI CHẠY CHƯƠNG TRÌNH
# ==============================================================================
if __name__ == "__main__":
    try:
        setup_environment()
        press_brake_expert = PressBrakeExpert()
        press_brake_expert.train()
    except Exception as e:
        print(f"\nLỖI HỆ THỐNG MÁY CHẤN: {e}")