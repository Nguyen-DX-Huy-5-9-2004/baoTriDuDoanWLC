import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) #thư mục hiện tại (./data)
BASE_DIR = os.path.dirname(SCRIPT_DIR) #lùi lại 1 cấp
RAW_DATA_DIR=os.path.join(BASE_DIR, 'data/raw') #trỏ thẳng vào (tBTDD/raw)
OUTPUT_REPORT=os.path.join(SCRIPT_DIR,'raw_dataset_audit_report.txt')

def analyze_dataframe(df, file_name, f_out):
    '''Phân tích và ghi kết quả của 1 dataFrame ra file báo cáo'''
    f_out.write(f"{'='*80}\n")
    f_out.write(f"Tên file: {file_name}\n")
    f_out.write(f"Kích thước {df.shape[0]:,} dòng x {df.shape[1]} cột\n")
    f_out.write(f"{'-'*80}\n")

    #Danh sachs cột và kiểu dữ liệu
    f_out.write("Danh sách cột và thông tin cơ bản:\n")
    missing_data = df.isnull().sum()
    for col in df.columns:
        dtype = df[col].dtype
        missing = missing_data[col]
        missing_pct = (missing / len(df)) * 100
        unique_vals = df[col].nunique()
        f_out.write(f"  • {col:<30} | Type: {str(dtype):<10} | Unique: {unique_vals:<6} | Missing: {missing} ({missing_pct:.2f}%)\n")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        f_out.write("Thống kê số liệu:\n")
        stats = df[numeric_cols].describe().T[['mean', 'std', 'min', 'max']]
        f_out.write(stats.to_string())
        f_out.write("\n")
    
    #phân tích nhãn tiềm năng
    f_out.write("CÁC TRƯỜNG CÓ KHẢ NĂNG LÀ NHÃN: \n")
    potential_targets = [col for col in df.columns if df[col].nunique() < 15 and df[col].nunique() > 1]
    if potential_targets:
        for col in potential_targets:
            val_counts = df[col].value_counts().to_dict()
            f_out.write(f"  -{col}: {val_counts}\n")
    else:
        f_out.write(" -> Khôgn tìm thấy cột nào có tính chất phân loại rõ \n")
    f_out.write("\n")

def main():
    print(f"Bắt đầu quét thư mục raw data: {RAW_DATA_DIR}")

    #mở file để ghi
    with open(OUTPUT_REPORT, "w", encoding='utf-8') as f_out:
        f_out.write("Báo cáo khám tổng quát dữ liệu thô raw data\n\n")
        #duyệt qua các thư mục con trong raw
        for root, dirs, files in os.walk(RAW_DATA_DIR):
            for file in files:
                if file.endswith('.csv') or file.endswith('.txt'):
                    file_path=os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, RAW_DATA_DIR)
                    print(f"Đang xử lý: {relative_path}...", end=" ")

                    try:
                        #thử nghiệm với các định dạng phân tách khác nhau
                        if file.endswith('.csv'):
                            df = pd.read_csv(file_path)
                        elif file.endswith('.txt'):
                            #các file txt thường cách nhau bằng tab hoặc khảong trắng
                            df = pd.read_csv(file_path, sep=r'\s+', header = None) #giả định bộ UCI không có header
                            #Đặ tên tạm cho các cột nếu file không có header
                            df.columns = [f"Col_{i+1}" for i in range(df.shape[1])]
                        
                        analyze_dataframe(df, relative_path, f_out)
                        print("OK.")

                    except Exception as e:
                        print(f"lỗi {e}")
                        f_out.write(f"Lỗi khi đọc file {relative_path}: {e}\n\n")
    print(f"Hoàn thnahf, kết quả được lưu tại {OUTPUT_REPORT}")
if __name__== "__main__":
    main()