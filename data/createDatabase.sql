CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ==============================================================================
-- NHÓM 1: DỮ LIỆU TĨNH & CẤU HÌNH CƠ KHÍ (Relational Data)
-- ==============================================================================

DROP TABLE IF EXISTS maintenance_tickets CASCADE;
DROP TABLE IF EXISTS ai_predictions CASCADE;
DROP TABLE IF EXISTS telemetry_stream CASCADE;
DROP TABLE IF EXISTS components CASCADE;
DROP TABLE IF EXISTS machines CASCADE;

-- 1. Bảng Danh mục Máy móc
CREATE TABLE machines (
    machine_id SERIAL PRIMARY KEY,
    machine_code VARCHAR(50) UNIQUE NOT NULL, 
    machine_type VARCHAR(100) NOT NULL,       
    location VARCHAR(255),                    
    image_url VARCHAR(500),                   
    status VARCHAR(50) DEFAULT 'ACTIVE',      
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Bảng Danh mục Cụm Linh Kiện
CREATE TABLE components (
    comp_id SERIAL PRIMARY KEY,
    machine_id INTEGER REFERENCES machines(machine_id) ON DELETE CASCADE,
    comp_code VARCHAR(20) NOT NULL,           
    comp_name VARCHAR(255) NOT NULL,          
    baseline_lifespan_days INTEGER,           
    UNIQUE(machine_id, comp_code)             
);

-- ==============================================================================
-- NHÓM 2: DỮ LIỆU ĐỘNG TỐC ĐỘ CAO (TimescaleDB Hypertable)
-- ==============================================================================

-- 3. Bảng Dữ liệu Cảm Biến Thời Gian Thực
CREATE TABLE telemetry_stream (
    time TIMESTAMPTZ NOT NULL,
    machine_id INTEGER REFERENCES machines(machine_id) ON DELETE CASCADE,
    sensor_payload JSONB NOT NULL             
);
SELECT create_hypertable('telemetry_stream', 'time');
CREATE INDEX ix_telemetry_payload ON telemetry_stream USING GIN (sensor_payload);

-- 4. Bảng Lịch sử Dự báo AI 
CREATE TABLE ai_predictions (
    time TIMESTAMPTZ NOT NULL,
    machine_id INTEGER REFERENCES machines(machine_id) ON DELETE CASCADE,
    comp_id INTEGER REFERENCES components(comp_id) ON DELETE CASCADE,
    risk_probability FLOAT NOT NULL,          
    shap_reasons JSONB,                       
    is_verified_true BOOLEAN                  
);
SELECT create_hypertable('ai_predictions', 'time');

-- ==============================================================================
-- NHÓM 3: DỮ LIỆU QUẢN TRỊ BẢO TRÌ (ERP / Feedback Loop)
-- ==============================================================================

-- 5. Bảng Phiếu sự cố
CREATE TABLE maintenance_tickets (
    ticket_id SERIAL PRIMARY KEY,
    machine_id INTEGER REFERENCES machines(machine_id) ON DELETE CASCADE,
    comp_id INTEGER REFERENCES components(comp_id), 
    created_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    status VARCHAR(50) DEFAULT 'PENDING',     
    resolution_notes TEXT                     
);

-- Thêm bảng lưu chỉ số vận hành thời gian thực
CREATE TABLE machine_live_metrics (
    metric_id SERIAL PRIMARY KEY,
    machine_id INTEGER REFERENCES machines(machine_id) ON DELETE CASCADE,
    load_percentage FLOAT DEFAULT 0.0, -- Công suất từ 0 đến 100%
    operation_level INTEGER DEFAULT 1,  -- 1: Thấp, 2: Trung bình, 3: Cao
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Tạo index để truy vấn nhanh cho dashboard
CREATE INDEX idx_machine_metrics_latest ON machine_live_metrics(machine_id, updated_at DESC);
-- ==============================================================================
-- BƠM DỮ LIỆU MẪU ĐẦY ĐỦ (DUMMY DATA GENERATION)
-- ==============================================================================

-- A. Khởi tạo 12 Máy Móc (Mỗi loại 3 máy, phủ rộng nhà máy)
INSERT INTO machines (machine_id, machine_code, machine_type, location, image_url, status) VALUES 
-- Nhóm Máy Hàn Robot
(1, 'WELD-R01', 'Máy Hàn Robot Tự Động', 'Xưởng A - Line 1', '/hinhMay/mayHanTuDong1.webp', 'ACTIVE'),
(2, 'WELD-R02', 'Máy Hàn Robot Tự Động', 'Xưởng A - Line 2', '/hinhMay/mayHanTuDong2.webp', 'ACTIVE'),
(3, 'WELD-R03', 'Máy Hàn Robot Tự Động', 'Xưởng A - Line 3', '/hinhMay/mayHanTuDong1.webp', 'MAINTENANCE'),

-- Nhóm Máy Cắt Laser
(4, 'LASER-F01', 'Máy Cắt Laser Fiber', 'Xưởng B - Line 1', '/hinhMay/mayCatLaser1.webp', 'ACTIVE'),
(5, 'LASER-F02', 'Máy Cắt Laser Fiber', 'Xưởng B - Line 2', '/hinhMay/mayCatLaser2.webp', 'ACTIVE'),
(6, 'LASER-F03', 'Máy Cắt Laser Fiber', 'Xưởng B - Line 3', '/hinhMay/mayCatLaser1.webp', 'WARNING'),

-- Nhóm Máy Chấn Tôn
(7, 'PRESS-B01', 'Máy Chấn Tôn CNC', 'Xưởng C - Line 1', '/hinhMay/mayChanTon1.webp', 'ACTIVE'),
(8, 'PRESS-B02', 'Máy Chấn Tôn CNC', 'Xưởng C - Line 2', '/hinhMay/mayChanTon1.webp', 'ACTIVE'),
(9, 'PRESS-B03', 'Máy Chấn Tôn CNC', 'Xưởng C - Line 3', '/hinhMay/mayChanTon1.webp', 'ACTIVE'),

-- Nhóm Máy Đột Dập
(10, 'PUNCH-T01', 'Máy Đột Dập CNC', 'Xưởng D - Line 1', '/hinhMay/mayCatLaser3.webp', 'WARNING'),
(11, 'PUNCH-T02', 'Máy Đột Dập CNC', 'Xưởng D - Line 2', '/hinhMay/mayCatLaser3.webp', 'ACTIVE'),
(12, 'PUNCH-T03', 'Máy Đột Dập CNC', 'Xưởng D - Line 3', '/hinhMay/mayCatLaser3.webp', 'ACTIVE');


-- B. Khởi tạo 48 Linh kiện (Mỗi máy 4 linh kiện lõi)
INSERT INTO components (machine_id, comp_code, comp_name, baseline_lifespan_days)
SELECT machine_id, 'COMP1', 'Bộ cấp dây (Wire Feeder)', 90 FROM machines WHERE machine_type = 'Máy Hàn Robot Tự Động' UNION ALL
SELECT machine_id, 'COMP2', 'Béc hàn (Torch Tip)', 15 FROM machines WHERE machine_type = 'Máy Hàn Robot Tự Động' UNION ALL
SELECT machine_id, 'COMP3', 'Nguồn hàn', 730 FROM machines WHERE machine_type = 'Máy Hàn Robot Tự Động' UNION ALL
SELECT machine_id, 'COMP4', 'Trục Servo Robot', 365 FROM machines WHERE machine_type = 'Máy Hàn Robot Tự Động' UNION ALL

SELECT machine_id, 'COMP1', 'Thấu kính quang học', 30 FROM machines WHERE machine_type = 'Máy Cắt Laser Fiber' UNION ALL
SELECT machine_id, 'COMP2', 'Máy làm mát Chiller', 180 FROM machines WHERE machine_type = 'Máy Cắt Laser Fiber' UNION ALL
SELECT machine_id, 'COMP3', 'Nguồn phát Laser', 1460 FROM machines WHERE machine_type = 'Máy Cắt Laser Fiber' UNION ALL
SELECT machine_id, 'COMP4', 'Thanh răng - Bánh răng', 365 FROM machines WHERE machine_type = 'Máy Cắt Laser Fiber' UNION ALL

SELECT machine_id, 'COMP1', 'Hệ thống van Servo Thủy lực', 180 FROM machines WHERE machine_type = 'Máy Chấn Tôn CNC' UNION ALL
SELECT machine_id, 'COMP2', 'Cụm Bơm thủy lực', 730 FROM machines WHERE machine_type = 'Máy Chấn Tôn CNC' UNION ALL
SELECT machine_id, 'COMP3', 'Lọc dầu thủy lực', 90 FROM machines WHERE machine_type = 'Máy Chấn Tôn CNC' UNION ALL
SELECT machine_id, 'COMP4', 'Thước quang học', 365 FROM machines WHERE machine_type = 'Máy Chấn Tôn CNC' UNION ALL

SELECT machine_id, 'COMP1', 'Cụm chày/cối (Punch/Die)', 30 FROM machines WHERE machine_type = 'Máy Đột Dập CNC' UNION ALL
SELECT machine_id, 'COMP2', 'Mâm dao quay (Turret)', 180 FROM machines WHERE machine_type = 'Máy Đột Dập CNC' UNION ALL
SELECT machine_id, 'COMP3', 'Trục X/Y tịnh tiến', 365 FROM machines WHERE machine_type = 'Máy Đột Dập CNC' UNION ALL
SELECT machine_id, 'COMP4', 'Cụm đầu búa đột', 730 FROM machines WHERE machine_type = 'Máy Đột Dập CNC';


-- C. BƠM LỊCH SỬ CẢM BIẾN (Lịch sử 7 ngày qua = 168 giờ = 1008 bản ghi/máy)
-- 1. Máy Hàn Robot (Chạy bình thường, dao động ngẫu nhiên)
INSERT INTO telemetry_stream (time, machine_id, sensor_payload)
SELECT 
    t_time, m.machine_id,
    jsonb_build_object(
        'robot_vibration_g', round((random() * 0.4 + 0.1)::numeric, 3),
        'servo_motor_temp_c', round((random() * 10 + 35)::numeric, 1),
        'wire_feed_speed_mmin', round((random() * 2 + 8)::numeric, 1)
    )
FROM generate_series(NOW() - INTERVAL '7 days', NOW(), INTERVAL '10 minutes') AS t_time
CROSS JOIN machines m WHERE m.machine_type = 'Máy Hàn Robot Tự Động';

-- 2. Máy Cắt Laser (Máy 4,5 bình thường. MÁY 6 BỊ MÔ PHỎNG LỖI 48H)
INSERT INTO telemetry_stream (time, machine_id, sensor_payload)
SELECT 
    t_time, m.machine_id,
    jsonb_build_object(
        'lens_temperature_c', CASE 
            WHEN m.machine_id = 6 AND t_time > (NOW() - INTERVAL '48 hours') -- Bắt đầu tăng nhiệt từ 48h trước
            THEN round((35 + (EXTRACT(EPOCH FROM (t_time - (NOW() - INTERVAL '48 hours')))/3600) * 0.8 + random() * 2)::numeric, 1)
            ELSE round((random() * 5 + 30)::numeric, 1) -- Bình thường
        END,
        'laser_source_temp_c', round((random() * 2 + 25)::numeric, 1),
        'xy_axis_torque_nm', round((random() * 10 + 40)::numeric, 1)
    )
FROM generate_series(NOW() - INTERVAL '7 days', NOW(), INTERVAL '10 minutes') AS t_time
CROSS JOIN machines m WHERE m.machine_type = 'Máy Cắt Laser Fiber';

-- 3. Máy Chấn Tôn (Chạy bình thường)
INSERT INTO telemetry_stream (time, machine_id, sensor_payload)
SELECT 
    t_time, m.machine_id,
    jsonb_build_object(
        'hydraulic_pressure_bar', round((random() * 15 + 160)::numeric, 1),
        'hydraulic_oil_temp_c', round((random() * 5 + 40)::numeric, 1)
    )
FROM generate_series(NOW() - INTERVAL '7 days', NOW(), INTERVAL '10 minutes') AS t_time
CROSS JOIN machines m WHERE m.machine_type = 'Máy Chấn Tôn CNC';

-- 4. Máy Đột Dập (Máy 11, 12 bình thường. MÁY 10 BỊ MÔ PHỎNG MÒN DAO TĂNG RUNG)
INSERT INTO telemetry_stream (time, machine_id, sensor_payload)
SELECT 
    t_time, m.machine_id,
    jsonb_build_object(
        'punch_force_kn', CASE 
            WHEN m.machine_id = 10 AND t_time > (NOW() - INTERVAL '48 hours') 
            THEN round((150 + (EXTRACT(EPOCH FROM (t_time - (NOW() - INTERVAL '48 hours')))/3600) * 1.5 + random() * 5)::numeric, 1)
            ELSE round((random() * 10 + 130)::numeric, 1) 
        END,
        'impact_vibration_g', CASE 
            WHEN m.machine_id = 10 AND t_time > (NOW() - INTERVAL '48 hours') 
            THEN round((3.0 + (EXTRACT(EPOCH FROM (t_time - (NOW() - INTERVAL '48 hours')))/3600) * 0.1 + random() * 0.5)::numeric, 2)
            ELSE round((random() * 1.5 + 2.0)::numeric, 2) 
        END,
        'tool_hits_count', (EXTRACT(EPOCH FROM (t_time - (NOW() - INTERVAL '7 days')))/600)::int * 100 -- Tăng số nhát dập
    )
FROM generate_series(NOW() - INTERVAL '7 days', NOW(), INTERVAL '10 minutes') AS t_time
CROSS JOIN machines m WHERE m.machine_type = 'Máy Đột Dập CNC';


-- D. BƠM LỊCH SỬ DỰ BÁO AI (Lịch sử rủi ro 7 ngày)
-- Các máy bình thường (Risk từ 5% - 25%)
INSERT INTO ai_predictions (time, machine_id, comp_id, risk_probability, shap_reasons, is_verified_true)
SELECT 
    t_time, m.machine_id, c.comp_id, 
    round((random() * 20 + 5)::numeric, 1), 
    '{"reason": "Hao mòn cơ học theo thời gian trong ngưỡng an toàn", "impact": "Low"}', 
    NULL
FROM generate_series(NOW() - INTERVAL '7 days', NOW(), INTERVAL '4 hours') AS t_time
CROSS JOIN machines m 
JOIN components c ON m.machine_id = c.machine_id AND c.comp_code = 'COMP1'
WHERE m.machine_id NOT IN (6, 10);

-- MÁY LASER 6: Cảnh báo tăng vọt trong 48h qua (Risk từ 20% lên 88%)
INSERT INTO ai_predictions (time, machine_id, comp_id, risk_probability, shap_reasons, is_verified_true)
SELECT 
    t_time, 6, (SELECT comp_id FROM components WHERE machine_id = 6 AND comp_code = 'COMP1'), 
    round((20 + (EXTRACT(EPOCH FROM t_time - (NOW() - INTERVAL '48 hours'))/3600) * 1.4)::numeric, 1), 
    '{"reason": "Dự báo cháy thấu kính: Trend nhiệt độ tăng liên tục + Quá hạn vệ sinh", "impact": "Critical"}', 
    NULL
FROM generate_series(NOW() - INTERVAL '48 hours', NOW(), INTERVAL '1 hour') AS t_time;

-- MÁY ĐỘT 10: Cảnh báo vàng trong 48h qua (Risk lên 65%)
INSERT INTO ai_predictions (time, machine_id, comp_id, risk_probability, shap_reasons, is_verified_true)
SELECT 
    t_time, 10, (SELECT comp_id FROM components WHERE machine_id = 10 AND comp_code = 'COMP1'), 
    round((25 + (EXTRACT(EPOCH FROM t_time - (NOW() - INTERVAL '48 hours'))/3600) * 0.8)::numeric, 1), 
    '{"reason": "Dao bắt đầu cùn: Lực đập Loadcell tăng 15% so với baseline + Rung chấn bất thường", "impact": "Medium"}', 
    NULL
FROM generate_series(NOW() - INTERVAL '48 hours', NOW(), INTERVAL '1 hour') AS t_time;


-- E. BƠM PHIẾU BẢO TRÌ SỰ CỐ (Lịch sử phong phú cho ERP)
INSERT INTO maintenance_tickets (machine_id, comp_id, created_at, resolved_at, status, resolution_notes) VALUES 
-- Lịch sử đã hoàn thành (RESOLVED)
(1, (SELECT comp_id FROM components WHERE machine_id=1 AND comp_code='COMP2'), NOW() - INTERVAL '6 days', NOW() - INTERVAL '5 days 20 hours', 'RESOLVED', 'Đã thay béc hàn định kỳ.'),
(4, (SELECT comp_id FROM components WHERE machine_id=4 AND comp_code='COMP1'), NOW() - INTERVAL '5 days', NOW() - INTERVAL '4 days 23 hours', 'RESOLVED', 'Lau sạch thấu kính hội tụ bám mạt sắt.'),
(7, (SELECT comp_id FROM components WHERE machine_id=7 AND comp_code='COMP3'), NOW() - INTERVAL '15 days', NOW() - INTERVAL '14 days', 'RESOLVED', 'Thay bộ lọc dầu thủy lực theo khuyến cáo của AI tháng trước.'),
(11, (SELECT comp_id FROM components WHERE machine_id=11 AND comp_code='COMP1'), NOW() - INTERVAL '3 days', NOW() - INTERVAL '2 days 22 hours', 'RESOLVED', 'Đã tháo chày cối mang đi mài phẳng lại.'),

-- Đang xử lý hoặc chờ (PENDING / IN_PROGRESS)
(3, (SELECT comp_id FROM components WHERE machine_id=3 AND comp_code='COMP4'), NOW() - INTERVAL '1 day', NULL, 'IN_PROGRESS', 'Đang tháo cụm Servo trục X để tra thêm mỡ bò. Robot báo lỗi quá tải.'),
(6, (SELECT comp_id FROM components WHERE machine_id=6 AND comp_code='COMP1'), NOW() - INTERVAL '2 hours', NULL, 'IN_PROGRESS', 'Kỹ sư nhận cảnh báo đỏ từ AI. Đang tiến hành dừng máy kiểm tra thấu kính.'),
(10, (SELECT comp_id FROM components WHERE machine_id=10 AND comp_code='COMP1'), NOW() - INTERVAL '30 minutes', NULL, 'PENDING', 'AI cảnh báo vàng. Lên lịch bảo trì mài dao vào ca 3 tối nay.');

-- Khôi phục lại chuỗi ID để tránh lỗi khi code Python Insert thêm
SELECT setval('machines_machine_id_seq', (SELECT MAX(machine_id) FROM machines));
SELECT setval('components_comp_id_seq', (SELECT MAX(comp_id) FROM components));
/*NHÓM 1: DỮ LIỆU TĨNH (Relational Data)
1. machines (Bảng Máy móc):

Chức năng: Cung cấp danh sách máy để rải ra các "Thẻ Máy" trên màn hình Tivi. Cung cấp đường dẫn ảnh 3D và Vị trí xưởng để hiển thị giống Hình 1 của BA.

2. components (Bảng Linh kiện):

Chức năng: Thay vì code cứng chữ "COMP1, COMP2" trên màn hình, Backend sẽ query bảng này để lấy tên thật (ví dụ: in ra chữ "Bộ cấp dây" thay vì "COMP1"). Quản lý linh hoạt cho từng dòng máy.

NHÓM 2: TIME-SERIES (TimescaleDB Hypertables - Cốt lõi)
3. telemetry_stream (Bảng Dữ liệu IoT):

Chức năng: Lưu dữ liệu đẩy lên từ Gateway IoT.

Đột phá (JSONB): Cột sensor_payload giải quyết triệt để bài toán rác cột. Khi chèn máy Hàn, IoT nhét JSON {"ampe": 100, "volt": 20}. Khi chèn máy Laser, IoT nhét JSON {"lens_temp": 45, "gas_pressure": 15}. Tất cả nằm gọn trong 1 bảng cực kỳ tối ưu, truy vấn trong PostgreSQL rất dễ dàng: SELECT sensor_payload->>'ampe' FROM telemetry_stream;

4. ai_predictions (Bảng Lịch sử Dự báo AI):

Chức năng: Script AI (predict_live_stream.py) thay vì chỉ in ra màn hình thì giờ đây sẽ INSERT kết quả % rủi ro vào bảng này mỗi khi quét xong.

Đột phá (Biểu đồ Đánh giá): Cột is_verified_true chính là linh hồn của "Biểu đồ tròn ở giữa" trong bản thiết kế của BA. Khi kỹ sư bấm xác nhận lỗi, cột này chuyển thành TRUE. Bạn dễ dàng SELECT COUNT(...) WHERE is_verified_true = TRUE để ra con số 18 dự đoán đúng, 6 dự đoán sai.

NHÓM 3: QUẢN TRỊ (ERP Data)
5. maintenance_tickets (Bảng Phiếu bảo trì):

Chức năng: Phục vụ cho "Biểu đồ tròn bên trái" của bản thiết kế. Dùng lệnh GROUP BY status để đếm xem có bao nhiêu lỗi Đã xử lý (Resolved), Chưa xử lý (Pending). Cột resolved_at sẽ được Backend trích xuất để làm feature days_since_comp_change đẩy vào mảng XGBoost để dự đoán.*/

/*1. Hướng Giám sát Thời gian thực (Real-time Dashboarding)
Biểu đồ Line/Time-series: Tận dụng bảng telemetry_stream (đã được tối ưu bằng TimescaleDB Hypertable) để vẽ các biểu đồ theo dõi các chỉ số trực tiếp như dòng điện, nhiệt độ, độ rung, áp suất... với độ trễ gần như bằng không.

Trạng thái tức thời: Sử dụng bảng machines để làm các thẻ (cards) cảnh báo trạng thái hiện tại (Hoạt động, Đang bảo trì, Mất kết nối).

2. Hướng Trí tuệ Nhân tạo & AI Giải thích được (XAI - Explainable AI)
Cảnh báo rủi ro (Risk Alerting): Bảng ai_predictions lưu trữ risk_probability rất phù hợp để làm hệ thống cảnh báo Đỏ/Vàng/Xanh cho kỹ sư.

Tính năng "Bác sĩ chẩn đoán": Việc bạn lưu shap_reasons (ví dụ: "Nhiệt độ thấu kính tăng vọt") là một điểm cộng rất lớn. Hướng đi này giúp người dùng không chỉ biết máy sắp hỏng, mà còn biết tại sao nó hỏng, từ đó tăng độ tin cậy của AI đối với công nhân vận hành.

Mô hình học máy (Feedback Loop): Trường is_verified_true sẽ là hướng đi cốt lõi để bạn liên tục đào tạo lại (retrain) mô hình AI, giúp nó ngày càng thông minh hơn qua những lần kỹ sư xác nhận đúng/sai.

3. Hướng Quản lý Quy trình Bảo trì (CMMS / ERP)
Tự động hóa Ticket: Dữ liệu từ nhóm AI có thể được dùng để tự động sinh ra các phiếu sự cố trong bảng maintenance_tickets khi risk_probability vượt ngưỡng cho phép (ví dụ > 80%).

Phân tích nguyên nhân gốc rễ (Root Cause Analysis): Kỹ sư bảo trì có thể nhìn vào resolution_notes và đối chiếu lại với lịch sử cảm biến (telemetry_stream) trước lúc máy hỏng để tìm ra chu kỳ hao mòn của các cụm linh kiện.

4. Hướng Phân tích Tuổi thọ & Tối ưu Tồn kho (Lifecycle & Inventory)
Đối chiếu baseline_lifespan_days (tuổi thọ lý thuyết) trong bảng components với thời gian hoạt động thực tế để nhắc nhở phòng Mua hàng nhập sẵn vật tư thay thế (ví dụ: béc hàn, thấu kính) trước khi linh kiện thực sự bị hỏng hóc, giúp giảm thiểu thời gian chết (downtime) của nhà máy.*/