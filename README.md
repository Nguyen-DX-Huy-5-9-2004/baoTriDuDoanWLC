# tBTDD - Predictive Maintenance Dashboard

Dự án hệ thống giám sát bảo trì dự đoán cho nhà máy. Hệ thống nhận dữ liệu IoT theo thời gian thực, dùng mô hình AI để ước lượng rủi ro hỏng linh kiện trong 48 giờ tiếp theo, sau đó hiển thị lên dashboard TV.

-Dữ liệu được dùng cho dự án là các dữ liệu chuẩn công nghiệp phổ biến, hoàn toàn có khả năng và dễ dàng triển khai IOT thu thập những dự liệu vận hành cần thiết để đáp ứng yêu cầu AI bảo trì dự đoán. Tuy nhiên tùy theo điều kiện thực tế của mỗi doanh nghiệp có thể dễ dàng lựa chọn/điều chỉnh các trường, đặc điểm dữ liệu đầu vào để mô hình có độ chính xác nhất

-Mô hình được xây dựng để vừa có khả năng triển khai, chạy tại biên hoặc sử dụng server cục bộ/đám mây của doanh nghiệp mà không cần yêu cầu GPU khắt khe
## Chức năng chính

- Backend FastAPI đọc dữ liệu streaming từ PostgreSQL/TimescaleDB.
- Mô hình hybrid GRU + XGBoost dự báo rủi ro cho 4 linh kiện `comp1` -> `comp4`.
- SHAP giải thích nguyên nhân khi rủi ro cao.
- Frontend React/Vite tối ưu cho màn hình TV, tự cập nhật mỗi giây.
- Script giả lập IoT feed dữ liệu lỗi dần theo thời gian thực hoặc tốc độ demo.
- Android TV wrapper dùng WebView để mở dashboard.
- Có thêm pipeline huấn luyện các mô hình chuyên gia theo loại máy trong `src/experts_training`.

## Cấu trúc dự án

```text
tBTDD/
├── data/                         # SQL, script kiểm tra/xử lý dữ liệu, raw dataset local
│   └── raw/microsoft-azure-predictive-maintenance/
├── demo/
│   ├── data_pipeline/            # sinh dữ liệu chuỗi thời gian, ERP features, live feeder
│   └── saved_models/             # model GRU + XGBoost đang được backend demo sử dụng
├── deployed_models/              # model Keras/scaler cho các expert theo loại máy
├── frontend_tv/                  # dashboard React/Vite
├── androidTV/                    # Android TV WebView wrapper
├── notebooks/                    # EDA và thử nghiệm training
├── src/
│   ├── backend_api.py            # FastAPI backend chính cho dashboard
│   ├── train_pipeline.py         # train GRU + XGBoost trên Azure PdM dataset
│   ├── models/                   # kiến trúc GRU, XGBoost wrapper, config training
│   ├── experts_training/         # training model chuyên gia cho từng nhóm máy
│   ├── inference_engine/         # router/buffer inference thử nghiệm
│   └── frontend_tv.py            # dashboard Streamlit cũ
├── requirements.txt              # thư viện ML PyTorch/XGBoost
├── requirements2.txt             # thư viện TensorFlow/expert models
└── README.md
```

## Yêu cầu môi trường

- Python 3.10+ khuyến nghị.
- Node.js/npm để chạy frontend Vite.
- PostgreSQL có cài TimescaleDB extension.
- Dataset Azure Predictive Maintenance tại:

```text
data/raw/microsoft-azure-predictive-maintenance/
├── PdM_errors.csv
├── PdM_failures.csv
├── PdM_machines.csv
├── PdM_maint.csv
└── PdM_telemetry.csv
```

Backend hiện đang hard-code database URL:

```text
postgresql://postgres:s@localhost:5432/predictive_maintenance
```

Nếu máy bạn dùng user/password/database khác, cần sửa đồng bộ trong:

- `src/backend_api.py`
- `demo/data_pipeline/live_data_feeder.py`
- `demo/data_pipeline/init_timescaledb.py`

## Cài đặt

Tạo môi trường Python:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements2.txt
pip install fastapi uvicorn streamlit requests
```

Cài frontend:

```powershell
cd frontend_tv
npm install
cd ..
```

## Chuẩn bị database

Tạo database:

```sql
CREATE DATABASE predictive_maintenance;
```

Chạy file schema và dữ liệu danh mục:

```powershell
psql -U postgres -d predictive_maintenance -f data/createDatabase.sql
```

File SQL tạo các bảng chính:

- `machines`: danh mục máy.
- `components`: linh kiện của từng máy.
- `telemetry_stream`: dữ liệu IoT realtime dạng TimescaleDB hypertable.
- `machine_live_metrics`: tải vận hành hiện tại của máy.
- `ai_predictions`, `maintenance_tickets`: lịch sử dự báo và phiếu bảo trì.

Lưu ý: khi `src/backend_api.py` khởi động, backend sẽ truncate `telemetry_stream` và `machine_live_metrics` để bắt đầu ca demo mới. Vì vậy màn hình sẽ chờ dữ liệu cho đến khi chạy live feeder.

## Chạy demo realtime

Mở 3 terminal từ thư mục gốc dự án.

Terminal 1 - backend:

```powershell
.\.venv\Scripts\Activate.ps1
python src/backend_api.py
```

Backend chạy tại:

```text
http://localhost:8000
```

Terminal 2 - frontend TV:

```powershell
cd frontend_tv
npm run dev -- --host
```

Frontend chạy tại:

```text
http://localhost:5173
```

Terminal 3 - giả lập IoT feed:

```powershell
.\.venv\Scripts\Activate.ps1
python demo/data_pipeline/live_data_feeder.py
```

Chọn tốc độ khi script hỏi:

- `1`: chạy như thời gian thực.
- `2`: demo nhanh, mỗi giờ dữ liệu tương ứng khoảng 2 giây.
- `3`: demo rất nhanh, mỗi giờ dữ liệu tương ứng khoảng 0.2 giây.

Sau khi feeder bắt đầu đẩy dữ liệu, dashboard sẽ chuyển khỏi màn hình chờ và hiển thị rủi ro theo từng máy.

## Android TV

Android wrapper trong `androidTV` mở dashboard qua WebView.

- Laptop/trình duyệt: `http://localhost:5173`
- Android emulator: `http://10.0.2.2:5173`

URL emulator đang được cấu hình trong:

```text
androidTV/src/main/java/com/example/myapplication/MainActivity.kt
```

## API chính

Backend FastAPI cung cấp các endpoint:

```text
GET  /api/live-status
POST /api/start-feeddata?speed=2
POST /api/stop-feeddata
POST /api/reset-feeddata
```

`GET /api/live-status` trả về:

- `status`: `waiting_data`, `success` hoặc `error`.
- `latest_time`: thời điểm dữ liệu mới nhất.
- `kpis`: tổng số máy, số máy ổn định/cảnh báo/báo động.
- `machines`: danh sách máy, rủi ro từng component, ảnh, vị trí, tải vận hành và giải thích SHAP nếu có.

## Huấn luyện model demo GRU + XGBoost

Pipeline chính:

```powershell
.\.venv\Scripts\Activate.ps1
python src/train_pipeline.py
```

Luồng xử lý:

1. Đọc Azure PdM dataset.
2. Tạo feature ERP từ lỗi, bảo trì, tuổi máy.
3. Tạo chuỗi telemetry 24 giờ cho GRU.
4. Train GRU feature extractor.
5. Ghép embedding GRU với ERP features.
6. Train 4 XGBoost classifier cho `comp1` -> `comp4`.
7. Lưu model vào `models/saved_models`.

Backend demo hiện load model từ `demo/saved_models`, nên nếu train lại và muốn dùng model mới cho dashboard, cần đặt các file sau vào `demo/saved_models`:

```text
gru_latest_model.pth
xgb_model_comp1.json
xgb_model_comp2.json
xgb_model_comp3.json
xgb_model_comp4.json
```

## Expert models theo loại máy

Repo còn có hướng huấn luyện model riêng cho từng nhóm máy:

- `src/experts_training/train_robot_expert.py`
- `src/experts_training/train_laser_expert.py`
- `src/experts_training/train_press_brake_expert.py`
- `src/experts_training/train_punching_expert.py`

Các model/scaler đã triển khai nằm trong:

```text
deployed_models/
├── robot_expert/
├── laser_expert/
├── press_brake_expert/
└── punching_expert/
```

Cấu hình registry expert nằm ở `src/config.py`. Phần này là nhánh thử nghiệm/triển khai model chuyên gia, khác với backend demo realtime đang dùng `demo/saved_models`.

## Dashboard Streamlit cũ

Ngoài frontend React, dự án vẫn giữ dashboard Streamlit:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run src/frontend_tv.py
```

Frontend React trong `frontend_tv` là giao diện nên dùng cho TV demo hiện tại.

## Lỗi thường gặp

### Dashboard đứng ở màn "đang chờ dữ liệu"

Kiểm tra:

- Backend `python src/backend_api.py` đã chạy chưa.
- Database `predictive_maintenance` đã có bảng `machines` và `components` chưa.
- Live feeder `python demo/data_pipeline/live_data_feeder.py` đã chạy chưa.
- `data/raw/microsoft-azure-predictive-maintenance/PdM_telemetry.csv` có tồn tại không.

### Backend lỗi kết nối database

Kiểm tra PostgreSQL/TimescaleDB đang chạy và thông tin kết nối có đúng với chuỗi:

```text
postgresql://postgres:s@localhost:5432/predictive_maintenance
```

### Backend không load được model

Kiểm tra thư mục `demo/saved_models` có đủ:

```text
gru_latest_model.pth
xgb_model_comp1.json
xgb_model_comp2.json
xgb_model_comp3.json
xgb_model_comp4.json
```

Nếu thiếu, backend vẫn có fallback mock risk thấp, nhưng dự báo thật và SHAP sẽ không đúng.

### Frontend gọi API thất bại

`frontend_tv/vite.config.js` đã proxy `/api` sang `http://localhost:8000`. Hãy chạy backend trước, rồi reload frontend.

## Ghi chú dữ liệu

`.gitignore` đang bỏ qua `data/**/*.csv` và `data/**/*.txt`, vì vậy nhiều dataset raw có thể chỉ tồn tại trên máy local. Khi clone sang máy khác, cần tự đặt lại dataset vào đúng thư mục `data/raw/...`.
