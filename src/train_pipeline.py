import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Đảm bảo luôn trỏ đúng về thư mục TBTDD bất kể config.py nằm ở đâu
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
CORRECT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "microsoft-azure-predictive-maintenance")

# Tạo đường dẫn lưu mô hình an toàn
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models", "saved_models")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

try:
    from models.config import GRU_CONFIG, XGB_CONFIG, TRAIN_CONFIG, SEQ_LENGTH
except ModuleNotFoundError:
    from config import GRU_CONFIG, XGB_CONFIG, TRAIN_CONFIG, SEQ_LENGTH #type: ignore

from demo.data_pipeline.erp_feature_engineering import ERPFeatureEngineer
from demo.data_pipeline.time_series_generator import TimeSeriesPreprocessor
from models.gru_extractor import GRURiskExtractor
from models.xgboost_classifier import XGBoostRiskPredictor

def main():
    print("="*60)
    print(" BẮT ĐẦU PIPELINE HUẤN LUYỆN HYBRID (GRU + XGBOOST)")
    print("="*60)

    # Tự động nhận diện phần cứng (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HỆ THỐNG] Đang sử dụng thiết bị tính toán: {device.type.upper()}")
    print(f"[HỆ THỐNG] Đang đọc dữ liệu từ: {CORRECT_DATA_DIR}")

    # ---------------------------------------------------------
    # BƯỚC 1: XỬ LÝ DỮ LIỆU TĨNH (ERP)
    # ---------------------------------------------------------
    print("\n[BƯỚC 1] Khởi tạo dữ liệu ERP...")
    telemetry_base = pd.read_csv(f"{CORRECT_DATA_DIR}/PdM_telemetry.csv", usecols=['datetime', 'machineID'], parse_dates=['datetime'])
    erp_engineer = ERPFeatureEngineer(CORRECT_DATA_DIR)
    erp_matrix = erp_engineer.execute_pipeline(telemetry_base)

    # ---------------------------------------------------------
    # BƯỚC 2: CHUẨN BỊ CHUỖI THỜI GIAN (TELEMETRY)
    # ---------------------------------------------------------
    print("\n[BƯỚC 2] Khởi tạo Dữ liệu Chuỗi thời gian cho GRU...")
    telemetry_raw = pd.read_csv(f"{CORRECT_DATA_DIR}/PdM_telemetry.csv", parse_dates=['datetime'])
    ts_preprocessor = TimeSeriesPreprocessor()
    dataloader = ts_preprocessor.create_dataloaders(
        telemetry_raw, erp_matrix, 
        batch_size=TRAIN_CONFIG['batch_size'], 
        seq_length=SEQ_LENGTH
    )

    # ---------------------------------------------------------
    # BƯỚC 3: HUẤN LUYỆN MẠNG GRU (FEATURE EXTRACTOR)
    # ---------------------------------------------------------
    print("\n[BƯỚC 3] Bắt đầu huấn luyện mạng Deep Learning GRU...")
    gru_model = GRURiskExtractor(config=GRU_CONFIG).to(device)
    
    # Cấu hình hàm mất mát (Binary Cross Entropy)
    criterion = nn.BCELoss()
    
    # Cấu hình Bộ Tối Ưu kèm L2 Regularization (Weight Decay)
    optimizer = optim.Adam(
        gru_model.parameters(), 
        lr=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG.get('weight_decay', 1e-4)
    )
    
    # Cấu hình Bộ Lập Lịch (Learning Rate Scheduler)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=TRAIN_CONFIG.get('scheduler_factor', 0.5), 
        patience=TRAIN_CONFIG.get('scheduler_patience', 5),
        min_lr=TRAIN_CONFIG.get('min_lr', 1e-6)
    )
    
    epochs = TRAIN_CONFIG.get('epochs', 50)
    best_loss = float('inf') 
    early_stop_counter = 0
    early_stopping_patience = TRAIN_CONFIG.get('early_stopping_patience', 15)
    
    gru_model.train()
    for epoch in range(epochs):
        # Lưu lại learning rate hiện tại để kiểm tra xem có bị giảm không
        current_lr = optimizer.param_groups[0]['lr']
        
        total_loss = 0
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            risk_probs, _ = gru_model(x_batch)
            
            loss = criterion(risk_probs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 500 == 0:
                print(f"   Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(dataloader)}] | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss/len(dataloader)
        print(f"-> Hoàn thành Epoch {epoch+1} | Average Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
        
        # 1. Kích hoạt Scheduler dựa trên hàm Loss trung bình
        scheduler.step(avg_loss)
        
        # Kiểm tra xem LR có bị giảm sau khi step không để in thông báo
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < current_lr:
            print(f"    [SCHEDULER] Tốc độ học đã giảm xuống: {new_lr:.6f}")
        
        # 2. Logic Lưu Mô Hình Tốt Nhất & Early Stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            early_stop_counter = 0 # Reset counter vì mô hình tốt lên
            best_model_path = os.path.join(MODEL_SAVE_DIR, "gru_best_model.pth")
            torch.save(gru_model.state_dict(), best_model_path)
            print(f"     [LƯU MÔ HÌNH] Đã cập nhật GRU Best Model (Loss giảm xuống: {best_loss:.4f})")
        else:
            early_stop_counter += 1
            print(f"     Không cải thiện. Dấu hiệu đi ngang: {early_stop_counter}/{early_stopping_patience}")
            
        # 3. Kích hoạt Dừng Sớm (Early Stopping)
        if early_stop_counter >= early_stopping_patience:
            print(f"\n [EARLY STOPPING] Kích hoạt dừng sớm tại Epoch {epoch+1} để tránh Overfitting!")
            break

    # Lưu Mô Hình Cuối Cùng (Latest Model)
    latest_model_path = os.path.join(MODEL_SAVE_DIR, "gru_latest_model.pth")
    torch.save(gru_model.state_dict(), latest_model_path)
    print(f"     [LƯU MÔ HÌNH] Đã lưu GRU Latest Model.")

    # Trước khi trích xuất vector, hãy Load lại trọng số tốt nhất (Best Model) để chất lượng đạt đỉnh
    gru_model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, "gru_best_model.pth")))

    # ---------------------------------------------------------
    # BƯỚC 4: RÚT VECTOR KÝ ỨC (EMBEDDINGS) TỪ GRU
    # ---------------------------------------------------------
    print("\n[BƯỚC 4] Trích xuất Vector Ký ức (Embeddings) toàn bộ dữ liệu từ Best Model...")
    all_embeddings = []
    
    gru_model.eval()
    with torch.no_grad():
        for x_batch, _ in dataloader:
            x_batch = x_batch.to(device)
            emb = gru_model.extract_features(x_batch)
            all_embeddings.append(emb.cpu().numpy())
            
    embeddings_matrix = np.vstack(all_embeddings)
    print(f"   Kích thước ma trận Ký ức (GRU): {embeddings_matrix.shape}")

    # ---------------------------------------------------------
    # BƯỚC 5: CHUẨN BỊ MA TRẬN LAI (HYBRID MATRIX) CHO XGBOOST
    # ---------------------------------------------------------
    print("\n[BƯỚC 5] Lắp ráp Ma trận Lai (GRU + ERP)...")
    valid_indices = dataloader.dataset.valid_indices
    valid_erp_matrix = erp_matrix.iloc[valid_indices].reset_index(drop=True)
    
    y_final = valid_erp_matrix[['risk_comp1', 'risk_comp2', 'risk_comp3', 'risk_comp4']]
    drop_cols = ['datetime', 'machineID', 'model'] + list(y_final.columns)
    X_erp_static = valid_erp_matrix.drop(columns=drop_cols)
    
    X_final = np.hstack((embeddings_matrix, X_erp_static.values))
    print(f"   Kích thước Ma trận Lai cuối cùng: {X_final.shape}")

    split_idx = int(len(X_final) * 0.8)
    X_train, X_test = X_final[:split_idx], X_final[split_idx:]
    y_train, y_test = y_final.iloc[:split_idx], y_final.iloc[split_idx:]

    # ---------------------------------------------------------
    # BƯỚC 6: HUẤN LUYỆN & ĐÁNH GIÁ XGBOOST
    # ---------------------------------------------------------
    print("\n[BƯỚC 6] Bắt đầu Giai đoạn 2 - XGBoost Meta-Classifier")
    xgb_predictor = XGBoostRiskPredictor(config=XGB_CONFIG)
    xgb_predictor.train(X_train, y_train, X_val=X_test, y_val=y_test)
    
    xgb_predictor.evaluate(X_test, y_test)
    
    # ---------------------------------------------------------
    # BƯỚC 7: LƯU MÔ HÌNH XGBOOST
    # ---------------------------------------------------------
    print("\n[BƯỚC 7] Lưu các mô hình XGBoost...")
    for comp, model in xgb_predictor.models.items():
        xgb_path = os.path.join(MODEL_SAVE_DIR, f"xgb_model_{comp}.json")
        model.save_model(xgb_path)
    print(f"     [LƯU MÔ HÌNH] Đã lưu thành công 4 mô hình XGBoost tại: {MODEL_SAVE_DIR}")
    
    print("\n ĐÃ HOÀN TẤT TOÀN BỘ PIPELINE DỰ BÁO BẢO TRÌ!")

if __name__ == "__main__":
    main()