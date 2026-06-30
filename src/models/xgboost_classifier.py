#C:\Users\huynd1\Downloads\tBTDD\src\models\xgboost_classifier.py
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class XGBoostRiskPredictor:
    """
    Mô hình XGBoost Giai đoạn 2: 
    Nhận Vector Ký ức (GRU) + Đặc trưng ERP để xuất ra dự đoán hỏng hóc cho 4 linh kiện.
    """
    def __init__(self, config):
        """Khởi tạo dựa trên cấu hình từ file config.py"""
        self.config = config
        self.components = ['comp1', 'comp2', 'comp3', 'comp4']
        self.models = {} # Chứa 4 mô hình độc lập cho 4 nhóm linh kiện

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Huấn luyện 4 mô hình XGBoost cho 4 linh kiện.
        y_train phải là ma trận chứa 4 cột nhãn tương ứng.
        """
        print("Bắt đầu huấn luyện hệ thống XGBoost Đa linh kiện...")
        
        for i, comp in enumerate(self.components):
            print(f"\n[Đang huấn luyện] Mô hình cho cụm: {comp.upper()}")
            
            # Tách nhãn của linh kiện hiện tại
            y_train_comp = y_train.iloc[:, i] if isinstance(y_train, pd.DataFrame) else y_train[:, i]
            
            # Cấu hình tham số cho bài toán Phân loại nhị phân (Binary Classification)
            params = {
                'objective': 'binary:logistic', # Dự đoán xác suất (0 -> 1)
                'max_depth': self.config.get('max_depth', 6),
                'learning_rate': self.config.get('learning_rate', 0.05),
                'n_estimators': self.config.get('n_estimators', 300),
                'subsample': self.config.get('subsample', 0.8),
                'colsample_bytree': self.config.get('colsample_bytree', 0.8),
                'random_state': self.config.get('random_state', 42),
                'eval_metric': 'auc',
                'n_jobs': -1 # Dùng toàn bộ nhân CPU
            }
            
            # Tự động tính toán Scale Pos Weight để phạt nặng lỗi False Negative (Dữ liệu mất cân bằng)
            num_negative = sum(y_train_comp == 0)
            num_positive = sum(y_train_comp == 1)
            if num_positive > 0:
                params['scale_pos_weight'] = num_negative / num_positive
            
            model = xgb.XGBClassifier(**params)
            
            # Thiết lập tập validation nếu có để tránh Overfitting
            eval_set = None
            if X_val is not None and y_val is not None:
                y_val_comp = y_val.iloc[:, i] if isinstance(y_val, pd.DataFrame) else y_val[:, i]
                eval_set = [(X_train, y_train_comp), (X_val, y_val_comp)]
                
            # Bắt đầu Train
            model.fit(
                X_train, y_train_comp,
                eval_set=eval_set,
                verbose=False
            )
            
            self.models[comp] = model
            print(f" Đã huấn luyện xong {comp.upper()}.")

    def predict_proba(self, X_test):
        """Xuất ra ma trận Xác suất % rủi ro cho cả 4 linh kiện"""
        probs = {}
        for comp in self.components:
            # Lấy xác suất của lớp 1 (Rủi ro hỏng hóc)
            probs[comp] = self.models[comp].predict_proba(X_test)[:, 1]
        
        return pd.DataFrame(probs)

    def evaluate(self, X_test, y_test, threshold=0.5):
        """Đánh giá chất lượng mô hình với Cost-Sensitive Threshold"""
        probs_df = self.predict_proba(X_test)
        
        print("\n" + "="*50)
        print("BÁO CÁO CHẤT LƯỢNG MÔ HÌNH (XGBOOST META-CLASSIFIER)")
        print("="*50)
        
        for i, comp in enumerate(self.components):
            print(f"\n--- Linh kiện: {comp.upper()} ---")
            y_true = y_test.iloc[:, i] if isinstance(y_test, pd.DataFrame) else y_test[:, i]
            y_probs = probs_df[comp]
            
            # Áp dụng ngưỡng cảnh báo
            y_preds = (y_probs >= threshold).astype(int)
            
            auc_score = roc_auc_score(y_true, y_probs)
            print(f"Chỉ số AUC: {auc_score:.4f}")
            print(classification_report(y_true, y_preds, target_names=['Bình thường', 'Rủi ro (Cảnh báo)']))

# ==========================================
# KHỐI TEST NHANH
# ==========================================
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from config import XGB_CONFIG
    
    print("--- KIỂM THỬ MODULE XGBOOST ---")
    
    # Giả lập ma trận đầu vào: 1000 mẫu, 40 đặc trưng (32 vector Ký ức + 8 đặc trưng ERP)
    X_dummy = pd.DataFrame(np.random.rand(1000, 40))
    # Giả lập nhãn rủi ro cho 4 linh kiện (Mất cân bằng dữ liệu: Rất hiếm khi hỏng)
    y_dummy = pd.DataFrame(np.random.choice([0, 1], size=(1000, 4), p=[0.95, 0.05]), columns=['comp1', 'comp2', 'comp3', 'comp4'])
    
    predictor = XGBoostRiskPredictor(config=XGB_CONFIG)
    predictor.train(X_dummy, y_dummy)
    
    print("\n[KIỂM TRA DỰ ĐOÁN XÁC SUẤT]")
    probs = predictor.predict_proba(X_dummy.head(3))
    print(probs)