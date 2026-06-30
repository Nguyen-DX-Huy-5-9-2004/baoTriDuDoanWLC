import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, RepeatVector
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Tạo dữ liệu mẫu (dữ liệu 1D)
def generate_data(n_samples=1000, seq_length=20):
    X = np.random.rand(n_samples, seq_length)
    return X

# Chuẩn hóa dữ liệu
data = generate_data()
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Tạo tập huấn luyện và kiểm tra
X_train, X_test = train_test_split(data_scaled, test_size=0.2, random_state=42)

# Xác định kích thước input
input_dim = X_train.shape[1]

# Tạo mô hình autoencoder với GRU
def create_gru_autoencoder(input_dim):
    inputs = Input(shape=(input_dim,))
    
    # Mô hình encoder sử dụng GRU
    encoded = LSTM(256, activation='relu')(inputs)
    
    # Decoder cũng sử dụng GRU
    decoded = RepeatVector(input_dim)(encoded)
    decoded = LSTM(input_dim, activation='sigmoid', return_sequences=True)(decoded)
    
    autoencoder = Model(inputs=inputs, outputs=decoded)
    return autoencoder

# Tạo và biên dịch mô hình
autoencoder = create_gru_autoencoder(input_dim)
autoencoder.compile(optimizer='adam', loss='mse')

# Huấn luyện mô hình
history = autoencoder.fit(X_train, X_train,
                        epochs=50,
                        batch_size=256,
                        shuffle=True,
                        validation_data=(X_test, X_test))

# Kiểm tra mô hình
decoded_imgs = autoencoder.predict(X_test)

# In ra một số mẫu so sánh giữa dữ liệu gốc và dữ liệu tái tạo
n = 10  # Số lượng mẫu để hiển thị
plt.figure(figsize=(20, 4))
for i in range(n):
    # Hiển thị dữ liệu gốc
    ax = plt.subplot(2, n, i + 1)
    plt.plot(X_test[i])
    plt.title("Original")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Hiển thị dữ liệu tái tạo
    ax = plt.subplot(2, n, i + 1 + n)
    plt.plot(decoded_imgs[i])
    plt.title("Reconstructed")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
