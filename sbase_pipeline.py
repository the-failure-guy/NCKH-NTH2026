import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

def find_and_load_nhanes_data(data_dir):
    """
    Quét và đọc toàn bộ dữ liệu MCQ và DEMO từ thư mục.
    Tự động xử lý định dạng .XPT (SAS) hoặc .CSV.
    """
    print(f"[*] Đang quét thư mục: {data_dir}")
    all_files = glob.glob(os.path.join(data_dir, '*.[xX][pP][tT]')) + glob.glob(os.path.join(data_dir, '*.[cC][sS][vV]'))
    
    mcq_list = []
    demo_list = []
    
    for file in all_files:
        filename = os.path.basename(file).upper()
        print(f"  - Đang đọc: {filename} ...", end=" ")
        
        # Chọn hàm đọc dựa trên định dạng
        if file.lower().endswith('.xpt'):
            df = pd.read_sas(file)
        else:
            df = pd.read_csv(file)
            
        print(f"({df.shape[0]} dòng)")
        
        # Phân loại file thành MCQ hay DEMO
        if 'MCQ' in filename:
            mcq_list.append(df)
        elif 'DEMO' in filename:
            demo_list.append(df)
            
    # Gộp tất cả các file MCQ lại theo chiều dọc (các năm khác nhau)
    if mcq_list:
        df_mcq = pd.concat(mcq_list, ignore_index=True)
        # Giữ lại mẫu quan sát mới nhất nếu SEQN bị trùng
        df_mcq = df_mcq.drop_duplicates(subset=['SEQN'], keep='last') 
    else:
        df_mcq = pd.DataFrame()
        
    if demo_list:
        df_demo = pd.concat(demo_list, ignore_index=True)
        df_demo = df_demo.drop_duplicates(subset=['SEQN'], keep='last')
    else:
        df_demo = pd.DataFrame()
        
    return df_mcq, df_demo

def clean_and_normalize_cdc_codes(df, columns):
    """
    Chuẩn hóa mã giá trị đặc thù của CDC NHANES:
    - 2 -> 0 (No)
    - 1 -> 1 (Yes)
    - 7, 9 -> NaN (Refused, Don't know)
    """
    for col in columns:
        if col in df.columns:
            # pd.read_sas đôi khi trả về float hoặc bytes, ta ép kiểu sang float để map
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Khởi tạo chuỗi map
            mapping = {1.0: 1.0, 2.0: 0.0, 7.0: np.nan, 9.0: np.nan}
            df[col] = df[col].map(mapping)
    return df

def preprocess_and_feature_engineering(df_mcq, df_demo):
    """
    Gắn nhãn mục tiêu S_base và kết hợp dữ liệu nhân khẩu học (nếu có).
    """
    # 1. Trích xuất các biến quan trọng từ MCQ
    target_cols = ['MCQ160L', 'MCQ220']
    
    # Nếu trong tập MCQ hoàn toàn không có các cột này, tạo cột trống
    for col in target_cols:
        if col not in df_mcq.columns:
            df_mcq[col] = np.nan
            
    # Làm sạch các cột mục tiêu
    df_mcq = clean_and_normalize_cdc_codes(df_mcq, target_cols)
    
    # 2. Xây dựng quy tắc gán nhãn S_base
    # S_base = 1 nếu mắc bệnh gan (MCQ160L == 1) hoặc ung thư (MCQ220 == 1)
    df_mcq['S_base'] = np.where((df_mcq['MCQ160L'] == 1.0) | (df_mcq['MCQ220'] == 1.0), 1.0, 0.0)
    
    # Nếu cả 2 đều là NaN, chúng ta không thể xác định được mục tiêu, gán NaN cho S_base
    mask_nan = df_mcq['MCQ160L'].isna() & df_mcq['MCQ220'].isna()
    df_mcq.loc[mask_nan, 'S_base'] = np.nan
    
    # Lọc bỏ các mẫu không có nhãn
    df_processed = df_mcq.dropna(subset=['S_base']).copy()
    
    # 3. Ghép nối với DEMO (Nhân khẩu học) nếu có
    if not df_demo.empty and 'SEQN' in df_demo.columns:
        print("[*] Đã tìm thấy dữ liệu DEMO, tiến hành ghép nối (Merge)...")
        # Giữ lại biến Tuổi (RIDAGEYR) và Giới tính (RIAGENDR)
        demo_cols = ['SEQN']
        if 'RIDAGEYR' in df_demo.columns: demo_cols.append('RIDAGEYR')
        if 'RIAGENDR' in df_demo.columns: demo_cols.append('RIAGENDR')
        
        df_demo_filtered = df_demo[demo_cols].copy()
        
        # Chuẩn hóa giới tính (RIAGENDR: 1-Male->1, 2-Female->0)
        if 'RIAGENDR' in df_demo_filtered.columns:
            df_demo_filtered['RIAGENDR'] = pd.to_numeric(df_demo_filtered['RIAGENDR'], errors='coerce')
            df_demo_filtered['RIAGENDR'] = df_demo_filtered['RIAGENDR'].map({1.0: 1.0, 2.0: 0.0})
            
        df_processed = pd.merge(df_processed, df_demo_filtered, on='SEQN', how='inner')
    else:
        print("[!] Không tìm thấy dữ liệu DEMO. Bỏ qua đặc trưng Tuổi và Giới tính.")
        
    return df_processed

def train_and_evaluate_sbase(df, output_dir):
    """
    Huấn luyện mô hình XGBoost dự đoán nguy cơ S_base
    """
    print(f"\n[*] Tổng số mẫu dữ liệu hợp lệ: {len(df)}")
    
    # Lựa chọn Feature
    features = [c for c in df.columns if c not in ['SEQN', 'S_base', 'MCQ160L', 'MCQ220']]
    
    # Nếu không có features nào (do thiếu DEMO và các cột khác), ta dùng một số biến MCQ có sẵn làm dummy
    if len(features) == 0:
        print("[!] Không đủ feature để train (thiếu DEMO). Trích xuất 5 cột ngẫu nhiên từ MCQ để demo pipeline...")
        mcq_extra = [c for c in df.columns if c.startswith('MCQ') and c not in ['MCQ160L', 'MCQ220']][:5]
        features.extend(mcq_extra)
        
    # Xử lý missing value cho features bằng Median
    df[features] = df[features].fillna(df[features].median())
    
    X = df[features]
    y = df['S_base']
    
    print(f"[*] Các đặc trưng đầu vào: {list(X.columns)}")
    print(f"[*] Phân phối nhãn S_base:\n{y.value_counts()}")
    
    # 1. Chia tập Train/Test (80/20, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Xử lý mất cân bằng lớp
    scale_weight = sum(y_train == 0) / sum(y_train == 1) if sum(y_train == 1) > 0 else 1.0
    print(f"[*] Tham số scale_pos_weight: {scale_weight:.2f}")
    
    # 3 & 4. Huấn luyện XGBoost với RandomizedSearchCV
    print("[*] Đang tinh chỉnh siêu tham số và huấn luyện XGBoost...")
    xgb_base = XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=scale_weight)
    
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 200]
    }
    
    search = RandomizedSearchCV(xgb_base, param_distributions=param_grid, n_iter=10, 
                                scoring='recall', cv=5, random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    print(f"[+] Bộ tham số tốt nhất: {search.best_params_}")
    
    # 5. Đánh giá mô hình
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    sens = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n" + "="*40)
    print("ĐÁNH GIÁ MÔ HÌNH XÁC SUẤT NGUY CƠ NỀN (S_base)")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Sensitivity: {sens:.4f} (Ưu tiên không bỏ sót ca bệnh)")
    print(f"Precision:   {prec:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"ROC-AUC:     {auc:.4f}")
    print("="*40)
    
    # 6. Biểu diễn Confusion Matrix & Feature Importance
    plt.figure(figsize=(12, 5))
    
    # -- Confusion Matrix
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    # -- Feature Importance
    plt.subplot(1, 2, 2)
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[-10:] # Top 10 features
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.title("Top Feature Importances")
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'sbase_evaluation_plots.png')
    plt.savefig(plot_path)
    print(f"[*] Đã lưu biểu đồ đánh giá tại: {plot_path}")
    
    # 7. Xuất mô hình
    model_path = os.path.join(output_dir, 'sbase_model.joblib')
    joblib.dump({'model': best_model, 'features': features}, model_path)
    print(f"[*] Đã lưu mô hình huấn luyện tại: {model_path}")
    
    return best_model, features

# ==========================================
# Module Tái sử dụng: Hàm dự đoán bệnh nhân mới
# ==========================================
def predict_sbase(patient_features, model_path='./Training/Model/MedicalCondition/sbase_model.joblib'):
    """
    Dự đoán xác suất nguy cơ nền S_base cho một bệnh nhân mới.
    :param patient_features: Dictionary chứa giá trị các đặc trưng
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy mô hình tại {model_path}")
        
    artifacts = joblib.load(model_path)
    model = artifacts['model']
    required_features = artifacts['features']
    
    # Chuyển patient_features thành DataFrame có thứ tự cột chuẩn
    input_data = []
    for f in required_features:
        input_data.append(patient_features.get(f, 0)) # Giá trị mặc định là 0 nếu không cung cấp
        
    df_input = pd.DataFrame([input_data], columns=required_features)
    
    # Dự đoán xác suất
    prob = model.predict_proba(df_input)[0][1]
    return prob

if __name__ == "__main__":
    data_dir = './Training/Model/MedicalCondition'
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Tải & Làm sạch
    df_mcq, df_demo = find_and_load_nhanes_data(data_dir)
    
    if df_mcq.empty:
        print("[!] Không tìm thấy dữ liệu. Vui lòng đặt các file NHANES vào thư mục.")
    else:
        # 2. Feature Engineering
        df_processed = preprocess_and_feature_engineering(df_mcq, df_demo)
        
        # 3. Huấn luyện
        model, features = train_and_evaluate_sbase(df_processed, data_dir)
        
        # 4. Test hàm predict
        sample_patient = {f: 1.0 for f in features} # Giá trị giả lập
        score = predict_sbase(sample_patient, os.path.join(data_dir, 'sbase_model.joblib'))
        print(f"\n[*] Test `predict_sbase` với bệnh nhân giả lập. Xác suất S_base: {score*100:.2f}%")
