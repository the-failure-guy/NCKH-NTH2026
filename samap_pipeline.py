import os
import glob
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve
from xgboost import XGBClassifier

# ==========================================
# 1. Định nghĩa các hàm toán học tính điểm
# ==========================================

def calculate_albi(bilirubin, albumin):
    """
    Tính chỉ số ALBI (Albumin-Bilirubin Score)
    :param bilirubin: Nồng độ Bilirubin toàn phần (µmol/L)
    :param albumin: Nồng độ Albumin trong máu (g/L)
    :return: ALBI Score
    """
    return (np.log10(bilirubin) * 0.66) + (albumin * -0.085)

def calculate_amap_raw(age, gender, albi, platelets):
    """
    Tính thang điểm aMAP thô
    :param age: Tuổi (năm)
    :param gender: Giới tính (1 = Nam, 0 = Nữ)
    :param albi: Điểm ALBI
    :param platelets: Số lượng tiểu cầu (10^9/L)
    :return: aMAP Score thô
    """
    return (0.06 * age) + (0.89 * gender) + (0.48 * albi) - (0.01 * platelets)

# ==========================================
# 2. Quy trình Load & Tiền xử lý Dữ liệu
# ==========================================

def load_data(data_dir='./Training'):
    """
    Load dữ liệu từ thư mục, ưu tiên csv, sau đó là excel.
    """
    csv_files = glob.glob(os.path.join(data_dir, '**/*.csv'), recursive=True)
    excel_files = glob.glob(os.path.join(data_dir, '**/*.xlsx'), recursive=True)
    
    if csv_files:
        file_path = csv_files[0]
        print(f"[*] Tìm thấy file CSV: {file_path}")
        df = pd.read_csv(file_path)
        
        # Xử lý riêng cho tập dữ liệu ILPD nếu không có header
        if 'ILPD' in file_path or 'Indian Liver' in file_path:
            print("[!] Nhận diện tập dữ liệu ILPD. Đang gán lại tên cột...")
            df = pd.read_csv(file_path, header=None)
            df.columns = ['age', 'gender', 'bilirubin', 'direct_bilirubin', 'alkaline_phosphotase', 
                          'alamine_aminotransferase', 'aspartate_aminotransferase', 'total_proteins', 
                          'albumin', 'albumin_and_globulin_ratio', 'target']
            # Xử lý giới tính
            df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
            # Đưa target về 0 và 1 (ILPD: 1 là bệnh, 2 là không bệnh)
            df['target'] = df['target'].map({1: 1, 2: 0})
            
        return df
    elif excel_files:
        print(f"[*] Tìm thấy file Excel: {excel_files[0]}")
        return pd.read_excel(excel_files[0])
    else:
        print("[!] Không tìm thấy dữ liệu. Tạo dữ liệu giả lập để minh họa quy trình...")
        return generate_mock_data(data_dir)

def generate_mock_data(data_dir):
    """
    Tạo dữ liệu giả lập (mock data) nếu không có file dữ liệu nào.
    """
    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame({
        'age': np.random.randint(20, 80, n_samples),
        'gender': np.random.choice([0, 1], n_samples),
        'albumin': np.random.normal(40, 5, n_samples),      # g/L
        'bilirubin': np.random.lognormal(mean=2, sigma=0.5, size=n_samples), # umol/L
        'platelets': np.random.normal(200, 50, n_samples),  # 10^9/L
    })
    # Tạo biến mục tiêu (target) giả lập với một số tương quan đơn giản
    risk_factor = (df['age']*0.05 + df['gender']*0.5 - df['albumin']*0.1 + np.log(df['bilirubin'])*1.5 - df['platelets']*0.01)
    prob = 1 / (1 + np.exp(-risk_factor))
    df['target'] = (prob > 0.5).astype(int)
    
    mock_file = os.path.join(data_dir, 'mock_dataset.csv')
    df.to_csv(mock_file, index=False)
    print(f"[*] Đã tạo dữ liệu giả lập tại: {mock_file}")
    return df

def preprocess_and_feature_engineering(df):
    """
    Xử lý missing values (median imputation) và Feature Engineering (tính ALBI, aMAP)
    """
    # Các cột bắt buộc
    required_cols = ['age', 'gender', 'albumin', 'bilirubin', 'platelets']
    
    # Imputation: Điền giá trị khuyết thiếu bằng Median hoặc giá trị mặc định nếu thiếu hẳn cột
    typical_values = {'age': 50, 'gender': 1, 'albumin': 40.0, 'bilirubin': 17.0, 'platelets': 200.0}
    for col in required_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
        else:
            print(f"[!] Dữ liệu thiếu cột '{col}', đang gán giá trị mặc định: {typical_values[col]}")
            df[col] = typical_values[col]
    
    # Tính ALBI và aMAP thô
    df['ALBI_score'] = calculate_albi(df['bilirubin'], df['albumin'])
    df['aMAP_score_raw'] = calculate_amap_raw(df['age'], df['gender'], df['ALBI_score'], df['platelets'])
    
    return df

# ==========================================
# 3. Huấn luyện Mô hình & Đánh giá
# ==========================================

def train_and_evaluate(df, output_dir='./Training'):
    # Lấy các features làm đầu vào
    features = ['age', 'gender', 'albumin', 'bilirubin', 'platelets', 'ALBI_score']
    
    # Chuẩn hóa S_aMAP bằng Min-Max Scaler (chuyển aMAP thô về khoảng [0, 1])
    # Giả định câu hỏi "chuẩn hóa khoảng [5]" là ý định chuẩn hóa Min-Max, hoặc 0-5. 
    # Ở đây chúng ta chuẩn hóa 0-1 để biểu diễn dưới dạng xác suất phân tầng nguy cơ.
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['S_aMAP'] = scaler.fit_transform(df[['aMAP_score_raw']])
    
    # Thêm S_aMAP vào input feature cho mô hình
    features.append('S_aMAP')
    
    X = df[features]
    if 'target' not in df.columns:
        print("[!] Không tìm thấy cột 'target' trong dữ liệu. Sẽ tự động tạo target giả lập để demo...")
        np.random.seed(42)
        df['target'] = np.random.choice([0, 1], size=len(df))
    y = df['target']
    
    # Chia tập Train/Test (80/20, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Tính tỷ lệ lớp để cấu hình scale_pos_weight
    neg_count = sum(y_train == 0)
    pos_count = sum(y_train == 1)
    scale_weight = neg_count / pos_count if pos_count > 0 else 1
    
    print("[*] Đang tối ưu hóa (Hyperparameter Tuning) XGBoost bằng RandomizedSearchCV...")
    from sklearn.model_selection import RandomizedSearchCV
    
    # Định nghĩa không gian tham số cần tối ưu
    param_dist = {
        'max_depth': [3, 4, 5, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 200, 300],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2, 0.5]
    }
    
    # Khởi tạo mô hình cơ sở
    xgb_base = XGBClassifier(
        random_state=42, 
        eval_metric='logloss',
        scale_pos_weight=scale_weight
    )
    
    # Tìm kiếm tham số ngẫu nhiên (RandomizedSearch để tối ưu tốc độ so với GridSearch)
    random_search = RandomizedSearchCV(
        xgb_base, 
        param_distributions=param_dist,
        n_iter=20,           # Số lượng tổ hợp tham số thử nghiệm
        scoring='roc_auc',   # Mục tiêu là tối ưu hóa ROC-AUC
        cv=3,                # K-fold Cross Validation = 3
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    # Huấn luyện tìm siêu tham số
    random_search.fit(X_train, y_train)
    
    # Chọn mô hình tốt nhất
    xgb_model = random_search.best_estimator_
    print(f"\n[+] Siêu tham số tốt nhất tìm được: {random_search.best_params_}")
    
    # Dự đoán trên tập test
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    # Đánh giá hiệu năng S_aMAP (Sử dụng chỉ S_aMAP như một heuristic classifier)
    # (Để đánh giá riêng năng lực phân tầng của S_aMAP theo yêu cầu số 4)
    y_pred_samap = (X_test['S_aMAP'] > X_test['S_aMAP'].median()).astype(int) # Dùng median của test set như threshold thô
    
    print("\n" + "="*40)
    print("ĐÁNH GIÁ MÔ HÌNH XGBOOST TRÊN TẬP TEST")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("="*40)
    
    # 4. Đánh giá riêng chỉ số S_aMAP
    print("\n" + "="*40)
    print("ĐÁNH GIÁ NĂNG LỰC PHÂN LOẠI RIÊNG CỦA S_aMAP (ROC-AUC)")
    samap_auc = roc_auc_score(y_test, X_test['S_aMAP'])
    print(f"ROC-AUC của riêng S_aMAP: {samap_auc:.4f}")
    print("="*40)
    
    # Vẽ biểu đồ ROC
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba)
    fpr_samap, tpr_samap, _ = roc_curve(y_test, X_test['S_aMAP'])
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})', color='blue')
    plt.plot(fpr_samap, tpr_samap, label=f'S_aMAP Chỉ số (AUC = {samap_auc:.3f})', color='red', linestyle='--')
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':')
    plt.title('ROC Curve: XGBoost vs S_aMAP Index')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    plot_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(plot_path)
    print(f"[*] Đã lưu biểu đồ ROC tại: {plot_path}")
    
    # 5. Xuất module (Lưu Model và Scaler)
    model_export_path = os.path.join(output_dir, 'samap_model.joblib')
    joblib.dump({'model': xgb_model, 'scaler': scaler}, model_export_path)
    print(f"[*] Đã lưu mô hình & scaler tại: {model_export_path}")
    
    return xgb_model, scaler

# ==========================================
# 4. Module tái sử dụng: calculate_samap
# ==========================================

def calculate_samap(patient_dict, model_path='./Training/samap_model.joblib'):
    """
    Nhận đầu vào là dictionary của 1 bệnh nhân và tính toán S_aMAP
    """
    age = patient_dict.get('age')
    gender = patient_dict.get('gender')
    albumin = patient_dict.get('albumin')
    bilirubin = patient_dict.get('bilirubin')
    platelets = patient_dict.get('platelets')
    
    albi = calculate_albi(bilirubin, albumin)
    amap_raw = calculate_amap_raw(age, gender, albi, platelets)
    
    # Load mô hình và scaler
    if os.path.exists(model_path):
        artifacts = joblib.load(model_path)
        scaler = artifacts['scaler']
        # Biến đổi (reshape) để tránh warning scaler
        s_amap_val = scaler.transform(np.array([[amap_raw]]))[0][0]
    else:
        # Fallback nếu chưa có scaler đã train
        print("[!] Không tìm thấy scaler đã train. Đang tính điểm thô.")
        s_amap_val = amap_raw
        
    return s_amap_val

# ==========================================
# Script Execution
# ==========================================
if __name__ == '__main__':
    data_dir = './Training'
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Đọc & Tiền xử lý
    df = load_data(data_dir)
    df = preprocess_and_feature_engineering(df)
    
    # 2 & 3. Huấn luyện, Đánh giá, Export
    train_and_evaluate(df, data_dir)
    
    # 4. Test hàm tính toán cho một bệnh nhân mới
    sample_patient = {
        'age': 55,
        'gender': 1,
        'albumin': 35.5,
        'bilirubin': 18.2,
        'platelets': 150
    }
    
    score = calculate_samap(sample_patient)
    print(f"\n[*] Test `calculate_samap` với bệnh nhân mẫu:")
    print(f"    Input: {sample_patient}")
    print(f"    S_aMAP Score chuẩn hóa: {score:.4f}")
