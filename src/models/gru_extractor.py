#C:\Users\huynd1\Downloads\tBTDD\src\models\gru_extractor.py
import torch
import torch.nn as nn

class GRURiskExtractor(nn.Module):
    """
    Mạng Neural GRU làm nhiệm vụ trích xuất Ký ức Lão hóa (Degradation Memory).
    """
    def __init__(self, config):
        """
        Args:
            config (dict): Dictionary chứa cấu hình mạng GRU từ file config.py
        """
        super(GRURiskExtractor, self).__init__()
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        
        # 1. Lớp GRU cốt lõi: Đọc hiểu chuỗi thời gian (24 giờ)
        self.gru = nn.GRU(
            input_size=config['input_size'], 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            batch_first=True, # Định dạng dữ liệu (Batch, Seq, Feature)
            dropout=config['dropout'] if self.num_layers > 1 else 0.0 # Chống Overfitting
        )
        
        # 2. Lớp Fully Connected (Linear) để xuất ra dự đoán xác suất rủi ro
        # Lớp này chủ yếu dùng để "ép" GRU phải học cách dự báo đúng trong giai đoạn huấn luyện (Pre-training)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(16, config['num_classes'])
        )
        
        # Hàm Sigmoid để đưa đầu ra về dạng xác suất (0% -> 100%)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Luồng đi của dữ liệu qua mạng Neural.
        x có kích thước: (batch_size, sequence_length, input_size)
        """
        # Khởi tạo trạng thái ẩn (hidden state) ban đầu là 0
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Cho dữ liệu chảy qua GRU
        # out: chứa toàn bộ lịch sử đầu ra của cửa sổ thời gian.
        # h_n: chứa trạng thái ẩn ở thời điểm cuối cùng. ĐÂY CHÍNH LÀ VECTOR KÝ ỨC!
        out, h_n = self.gru(x, h0)
        
        # Trích xuất Vector Ký ức ở lớp GRU cuối cùng (layer trên cùng)
        # Kích thước: (batch_size, hidden_size)
        memory_vector = h_n[-1, :, :]
        
        # Đưa Vector Ký ức qua lớp Linear để dự báo rủi ro
        risk_logits = self.fc(memory_vector)
        risk_probs = self.sigmoid(risk_logits)
        
        # Trả về cả Xác suất dự báo và Vector Ký ức (để đưa cho XGBoost ở Giai đoạn 2)
        return risk_probs, memory_vector

    def extract_features(self, x):
        """
        Hàm dùng riêng cho Giai đoạn 2: Chỉ lấy Vector Ký ức, không cần dự đoán.
        """
        self.eval() # Chuyển sang chế độ suy luận
        with torch.no_grad():
            _, memory_vector = self.forward(x)
        return memory_vector


# ==========================================
# KHỐI TEST (Kiểm tra hình dáng của Tensor 3D)
# ==========================================
if __name__ == "__main__":
    import sys
    import os
    
    # Thêm thư mục gốc vào path để import config khi test trực tiếp file này
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from config import GRU_CONFIG, TRAIN_CONFIG, SEQ_LENGTH

    print("--- KIỂM THỬ KIẾN TRÚC MẠNG GRU (SỬ DỤNG CONFIG) ---")
    
    # 1. Giả lập một Batch dữ liệu đầu vào lấy thông số tự động từ config
    batch_size = TRAIN_CONFIG['batch_size']
    input_size = GRU_CONFIG['input_size']
    
    dummy_input = torch.randn(batch_size, SEQ_LENGTH, input_size)
    print(f"1. Tensor Đầu vào (Dữ liệu thô): {dummy_input.shape}")
    
    # 2. Khởi tạo mô hình chỉ bằng 1 biến config
    model = GRURiskExtractor(config=GRU_CONFIG)
    
    # 3. Đưa dữ liệu qua mạng
    predicted_risks, embeddings = model(dummy_input)
    
    # 4. In kết quả để xác nhận
    print(f"2. Đầu ra 1 - Xác suất rủi ro (Risk Probs): {predicted_risks.shape} -> Dự báo % hỏng cho {GRU_CONFIG['num_classes']} comp")
    print(f"3. Đầu ra 2 - Vector Ký ức (Embeddings) : {embeddings.shape} -> Khối dữ liệu cô đặc sẽ giao cho XGBoost")
    
    print("\n KIẾN TRÚC HOẠT ĐỘNG HOÀN HẢO! Hệ thống config đã được liên kết thành công.")