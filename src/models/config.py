#C:\Users\huynd1\Downloads\tBTDD\src\models\config.py
import os

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN (PATHS)
# ==========================================
# Tự động lấy đường dẫn gốc của dự án (TBTDD)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Đường dẫn dữ liệu và mô hình
DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "microsoft-azure-predictive-maintenance")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models", "saved_models")

# Tạo thư mục lưu model nếu chưa có
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


# ==========================================
# 2. CẤU HÌNH DỮ LIỆU (DATA)
# ==========================================
SEQ_LENGTH = 24       # Độ dài cửa sổ thời gian (24 giờ)
NUM_SENSORS = 4       # Số lượng cảm biến (volt, rotate, pressure, vibration)
NUM_COMPONENTS = 4    # Số lượng linh kiện dự báo rủi ro (comp1 -> comp4)
RISK_WINDOW = 48      # Cửa sổ rủi ro để dán nhãn (Dự báo trước 48h)


# ==========================================
# 3. CẤU HÌNH GIAI ĐOẠN 1: MẠNG GRU
# ==========================================
GRU_CONFIG = {
    "input_size": NUM_SENSORS,
    "hidden_size": 64,      # Tăng kích thước Vector Ký ức (Embeddings) để học sâu hơn
    "num_layers": 2,        # Độ sâu của mạng GRU
    "dropout": 0.3,         # Tăng tỉ lệ Dropout lên 0.3 để chống Overfitting
    "num_classes": NUM_COMPONENTS
}


# ==========================================
# 4. CẤU HÌNH GIAI ĐOẠN 2: XGBOOST (META-CLASSIFIER)
# ==========================================
XGB_CONFIG = {
    "max_depth": 7,                 # Tăng độ sâu của cây để nắm bắt tương tác phức tạp
    "learning_rate": 0.01,          # Giảm tốc độ học giúp mô hình hội tụ an toàn và ổn định hơn
    "n_estimators": 1000,           # Tăng số cây lên mức tối đa (kết hợp với early_stopping)
    "subsample": 0.8,               # Lấy ngẫu nhiên 80% dữ liệu để xây mỗi cây
    "colsample_bytree": 0.8,        # Lấy ngẫu nhiên 80% đặc trưng để xây mỗi cây
    "reg_alpha": 0.1,               # L1 Regularization: Phạt các đặc trưng không quan trọng
    "reg_lambda": 1.5,              # L2 Regularization: Ngăn chặn Overfitting triệt để
    "random_state": 42
}


# ==========================================
# 5. CẤU HÌNH QUÁ TRÌNH HUẤN LUYỆN (TRAIN PYTORCH TRÊN COLAB)
# ==========================================
TRAIN_CONFIG = {
    "batch_size": 256,               # Tăng Batch Size để vắt kiệt RAM của Colab GPU
    "learning_rate": 1e-3,          # Tốc độ học khởi điểm cho Adam/AdamW
    "weight_decay": 1e-4,           # L2 Penalty giúp mô hình GRU mượt mà hơn
    "epochs": 20,                  # Huấn luyện dài hơi hơn
    
    # --- Cấu hình Bộ Lập Lịch (Learning Rate Scheduler) ---
    "scheduler_patience": 5,        # Nếu sau 5 epoch mà Loss không giảm, sẽ kích hoạt Scheduler
    "scheduler_factor": 0.5,        # Giảm tốc độ học xuống còn một nửa (LR * 0.5)
    "min_lr": 1e-6,                 # Tốc độ học thấp nhất giới hạn
    
    # --- Cấu hình Dừng Sớm (Early Stopping) ---
    "early_stopping_patience": 3   # Nếu sau 15 epoch Loss vẫn không giảm, sẽ ép dừng để tiết kiệm thời gian
}