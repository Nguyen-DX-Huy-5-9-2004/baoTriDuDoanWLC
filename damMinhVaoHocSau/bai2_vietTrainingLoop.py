import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
#import class ở bài 1
from bai1_taoCustomDataset import customDataset, X_np, Y_np

#Khởi tạo dataset và dataLoader
dataset = customDataset(X_np, Y_np)
dataloader = DataLoader(dataset, batch_size = 8, shuffle=True)

#Tạo một mô hình tuyến tính (Linear) nhận vào 5 features, đầu ra là 1 giá trị
model = nn.Linear(in_features=5, out_features=1)

#sử dụng hàm mất mát Binary Cross entropy với logits
criterion = nn.BCEWithLogitsLoss()

#thuật toán tối ưu stochastic gradient Descent (SGD) để cập nhật trọng số
optimizer=optim.SGD(model.parameters(), lr=0.01)
epochs = 5000 #chạy qua toàn bộ dữ liệu 5 lần

print("Bắt đàu quá trình huấn luyện)")
print("_"*40)
for epoch in range(epochs):
    epoch_loss=0.0
    #duyệt qua từng batch/gói dữ liệu
    for x_batch, y_batch in dataloader:
        
        #Forward pass. Đưa dữ liệu qua model để dự báo kết quả
        predictions = model(x_batch)
        
        #tính toán độ lỗi-loss giữa kết quả dự báo và nhãn thực tế
        loss = criterion(predictions, y_batch)

        #Xóa các đạo hàm cũ của lượt tính trước
        optimizer.zero_grad()

        #Backward pas. Lan truyền ngược, tính đạo hàmduwj trên độ lỗi
        loss.backward()

        #Cập nhật trọng số của mô hình
        optimizer.step()

        #cộng dồn loss của từng batch để tính toán loss trung bình của epoch
        epoch_loss += loss.item()

    avg_loss = epoch_loss /len(dataloader)
    print(f"Epoch [{epoch+1}/{epoch}] - loss trung bình: {avg_loss:.4f}")

print("_"*40)


