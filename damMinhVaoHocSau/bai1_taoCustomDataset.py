#Tạo một Custom Dataset và DataLoader để xử lý dữ liệu dạng bảng (Tabular Data)
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

#tạo dữ liệu giả
np.random.seed(42) #cố định sự ngẫu nhiên để mỗi lần chạy để ra cùng một số
X_np = np.random.randn(100, 5) #Tạo ngẫu nhiên 1 tensor 100 dòng và 5 cột
Y_np = np.random.randint(0, 2, size=(100, 1)) #vector 100 dòng 1 cột với giá trị 0 hoặc 1

class customDataset(Dataset):
    def __init__(self, X, Y):
        self.x=X
        self.y=Y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        giaTri=self.x[idx]
        nhan=self.y[idx]

        features_tensor=torch.tensor(giaTri, dtype=torch.float32)
        label_tensor=torch.tensor(nhan, dtype=torch.float32)

        return features_tensor, label_tensor

dataset1= customDataset(X_np, Y_np)
myDataloader = DataLoader(dataset=dataset1, batch_size=8, shuffle=True)
if __name__ == '__main__':
    print(f"Tổng số mẫu trong dataset la {len(dataset1)}")
for x_batch, y_batch in myDataloader:
    print(f"Kich thươc X_batch: {x_batch.shape}")
    print(f"kích thước Y_batch: {y_batch.shape}")
    print(f"Kiểu dữ liệu của X_batch: {x_batch.dtype}")
    break