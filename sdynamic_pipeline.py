import os
import glob
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

# ==========================================
# 1. Định nghĩa các hàm chuẩn hóa sinh hiệu
# ==========================================
def scale_hr(hr):
    """
    Chuẩn hóa HR (Nhịp tim nghỉ): HR càng cao (>80) thì rủi ro tiến về 1.
    Khoảng bình thường: 60 - 80.
    Dưới 60 (vận động viên) coi như an toàn (0).
    Trên 100 coi như cực kỳ nguy hiểm (1).
    """
    scaled = (hr - 60) / (100 - 60)
    return np.clip(scaled, 0.0, 1.0)

def scale_hrv(hrv):
    """
    Chuẩn hóa HRV (Biến thiên nhịp tim): HRV càng thấp (<30ms) thì rủi ro tiến về 1.
    Khoảng an toàn: > 60ms.
    """
    scaled = (60 - hrv) / (60 - 20)
    return np.clip(scaled, 0.0, 1.0)

def scale_spo2(spo2):
    """
    Chuẩn hóa SpO2: SpO2 < 95% là có rủi ro, tiến về 1.
    Khoảng an toàn: >= 98%.
    """
    scaled = (98 - spo2) / (98 - 90)
    return np.clip(scaled, 0.0, 1.0)

def scale_activity(steps):
    """
    Chuẩn hóa vận động: Số bước < 3000 tiến về 1 (ít vận động).
    An toàn: > 8000 bước.
    """
    scaled = (8000 - steps) / (8000 - 2000)
    return np.clip(scaled, 0.0, 1.0)

def calculate_sdynamic(row):
    """
    Tính tổng điểm S_dynamic dựa trên triệu chứng và sinh hiệu.
    """
    s_symptom = (row['spider_nevi'] + row['jaundice']) / 2.0
    s_hr = scale_hr(row['resting_hr'])
    s_hrv = scale_hrv(row['hrv'])
    s_spo2 = scale_spo2(row['spo2'])
    s_act = scale_activity(row['activity_steps'])
    
    # S_dynamic = 0.33*S_symptom + 0.27*S_HR + 0.20*S_activity + 0.10*S_HRV + 0.10*S_SpO2
    s_dyn = (0.33 * s_symptom) + (0.27 * s_hr) + (0.20 * s_act) + (0.10 * s_hrv) + (0.10 * s_spo2)
    return s_dyn

# ==========================================
# 2. Xử lý Dữ liệu
# ==========================================
def load_or_generate_dynamic_data(data_dir):
    """
    Quét dữ liệu sinh hiệu. Nếu không có file đúng định dạng, tạo dữ liệu giả lập.
    """
    required_cols = ['resting_hr', 'hrv', 'spo2', 'activity_steps', 'spider_nevi', 'jaundice']
    
    csv_files = glob.glob(os.path.join(data_dir, '**/*.csv'), recursive=True)
    valid_df = None
    
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if all(col in df.columns for col in required_cols):
                valid_df = df
                print(f"[*] Tìm thấy file dữ liệu hợp lệ: {f}")
                break
        except: pass
        
    if valid_df is not None:
        return valid_df
        
    print("[!] Không tìm thấy dữ liệu có cấu trúc đúng. Đang tạo dữ liệu giả lập...")
    np.random.seed(42)
    n = 2000
    
    # Giả lập phân phối dữ liệu cho người khỏe mạnh và người có bệnh
    is_sick = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    
    resting_hr = np.where(is_sick, np.random.normal(85, 10, n), np.random.normal(65, 8, n))
    hrv = np.where(is_sick, np.random.normal(30, 10, n), np.random.normal(65, 15, n))
    spo2 = np.where(is_sick, np.random.normal(94, 2, n), np.random.normal(98, 1, n))
    activity_steps = np.where(is_sick, np.random.normal(3000, 1500, n), np.random.normal(8000, 2500, n))
    
    # Triệu chứng
    spider_nevi = np.where(is_sick, np.random.choice([0, 0.5, 1], n, p=[0.2, 0.3, 0.5]), 
                                     np.random.choice([0, 0.5, 1], n, p=[0.9, 0.08, 0.02]))
    jaundice = np.where(is_sick, np.random.choice([0, 0.5, 1], n, p=[0.3, 0.4, 0.3]), 
                                  np.random.choice([0, 0.5, 1], n, p=[0.95, 0.05, 0.0]))
                                  
    df = pd.DataFrame({
        'resting_hr': resting_hr,
        'hrv': hrv,
        'spo2': spo2,
        'activity_steps': np.clip(activity_steps, 0, None),
        'spider_nevi': spider_nevi,
        'jaundice': jaundice
    })
    
    # Lưu lại file giả lập
    mock_file = os.path.join(data_dir, 'mock_dynamic_data.csv')
    df.to_csv(mock_file, index=False)
    print(f"[*] Đã lưu dữ liệu giả lập tại: {mock_file}")
    
    return df

# ==========================================
# 3. Huấn luyện Mô hình S_dynamic
# ==========================================
def train_sdynamic(df, output_dir):
    features = ['resting_hr', 'hrv', 'spo2', 'activity_steps', 'spider_nevi', 'jaundice']
    
    # Tính S_dynamic liên tục
    df['S_dynamic_score'] = df.apply(calculate_sdynamic, axis=1)
    
    # Định nghĩa nhãn rủi ro nhị phân (Classifier Target) dựa trên ngưỡng 0.5
    df['target'] = (df['S_dynamic_score'] >= 0.5).astype(int)
    
    X = df[features]
    y = df['target']
    
    print(f"[*] Phân phối nhãn S_dynamic rủi ro cao:\n{y.value_counts()}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scale_weight = sum(y_train == 0) / sum(y_train == 1) if sum(y_train == 1) > 0 else 1.0
    
    xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, 
                        scale_pos_weight=scale_weight, random_state=42, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    
    y_pred = xgb.predict(X_test)
    y_pred_proba = xgb.predict_proba(X_test)[:, 1]
    
    print("\n" + "="*40)
    print("ĐÁNH GIÁ MÔ HÌNH XÁC SUẤT ĐỘNG (S_dynamic)")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("="*40)
    
    # Lưu mô hình
    model_path = os.path.join(output_dir, 'sdynamic_model.joblib')
    joblib.dump({'model': xgb, 'features': features}, model_path)
    print(f"[*] Đã lưu mô hình tại: {model_path}")
    
    return xgb

if __name__ == "__main__":
    data_dir = './Training/Model/DynamicSignal'
    os.makedirs(data_dir, exist_ok=True)
    
    df = load_or_generate_dynamic_data(data_dir)
    train_sdynamic(df, data_dir)
