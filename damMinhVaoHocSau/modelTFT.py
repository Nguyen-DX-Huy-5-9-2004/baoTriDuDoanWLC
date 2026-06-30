import torch
import torch.nn as nn

#ĐỊnh nghĩa khối cổng GLU
class gatedLinearUnit(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        #nhân đôi kích thước để chẻ đôi tensor ở đầu ra
        self.linear = nn.Linear(d_model, d_model*2)
    
    def forward(self, x):
        # x có shape: [batch_size, time_steps, d_model]
        x_projected = self.linear(x) #kích thước d_model được nhân đôi

        #chẻ đôi tensor theo chiều cuối cùng (dim=-1)
        #nửa đầu làm giá trị (val), nửa sau làm cổng chọn lọc (gate)
        val, gate = torch.chunk(x_projected, chunks=2, dim=-1)

        #Trả về theo cơ chế gating: giá trị nhân với SIgmoid của cổng
        return val * torch.sigmoid(gate)

#Định nghĩa khối nền tảng GRN
class gatedResidualNetwork(nn.Module):
    def __init__(self, d_input, d_hidden, d_output):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output

        #Các tầng xử lý tuần tự theo công thức toán học của TFT
        self.linear_1=nn.Linear(d_input, d_hidden)
        self.elu = nn.ELU()
        self.linear_2 = nn.Linear(d_hidden, d_output)

        #Khối cổng GLU đã định nghĩa ở bước 1
        self.glu = gatedLinearUnit(d_output)

        #tầng chuẩn hóa - layer normalization
        self.layer_norm = nn.LayerNorm(d_output)

        #nếu kích thước đầu vào khác đầu ra, cần 1 layer skip để ép lại kích thước cho phép cộng 
        if d_input != d_output:
            self.skip_layer = nn.Linear(d_input, d_output)
        else:
            self.skip_layer = nn.Identity() #Giữ nguyên đầu vào
    
    def forward(self, x):
        #cộng kết nối tắt (residual) chuẩn bị sẵn
        residual = self.skip_layer(x)

        #Đi qua luồng xử lý phi tuyến
        x_processed = self.linear_1(x)
        x_processed = self.elu(x_processed)
        x_processed = self.linear_2(x_processed)

        #đi qua cổng chọn loc thông tin GLU
        x_gated = self.glu(x_processed)
        
        #cộng residual và chuẩn hóa đầu ra theo đúng thiết kế
        return self.layer_norm(residual + x_gated)
    
if __name__ == "__main__":
    #giả lập batch dữ liệu chuỗi thời gian:
    #Batch size = 4, số bước thời gian (nhìn lại quá khứ) = 10, số đặc trưng = 5
    X_dummy= torch.randn(4, 10, 5)

    #khởi tạo khối GRN: nhận vào 5 features, tầng ẩn 16, đầu ra chuyển đổi thành 32 vector đặc trưng
    grn_block = gatedResidualNetwork(d_input = 5, d_hidden = 16, d_output = 32)
    output = grn_block(X_dummy)

    print(f"Kích thước input ban đầu: {X_dummy.shape}")
    print(f"kích thước output GRN: {output.shape}")