import os
import numpy as np
import pandas as pd
import joblib
import json

# Định nghĩa các hàm phụ trợ tính điểm thủ công cho S_aMAP và S_dynamic
from samap_pipeline import calculate_albi, calculate_amap_raw
from sdynamic_pipeline import calculate_sdynamic

def load_models():
    """
    Load toàn bộ 3 lõi AI.
    """
    paths = {
        'samap': './Training/samap_model.joblib',
        'sbase': './Training/Model/MedicalCondition/sbase_model.joblib',
        'sdynamic': './Training/Model/DynamicSignal/sdynamic_model.joblib'
    }
    
    models = {}
    for name, path in paths.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            print(f"[!] Lỗi: Không tìm thấy mô hình {name} tại {path}")
            return None
            
    return models

def predict_patient_risk(patient_dict):
    """
    Hàm chính để đánh giá tổng điểm rủi ro R_total dựa trên Đa mô thức.
    """
    models = load_models()
    if not models:
        return {"error": "Missing AI cores"}
        
    # ==========================================
    # 1. CORE 1: S_aMAP (Huyết học)
    # ==========================================
    age = patient_dict.get('age', 50)
    gender = patient_dict.get('gender', 1)
    albumin = patient_dict.get('albumin', 40.0)
    bilirubin = patient_dict.get('bilirubin', 17.0)
    platelets = patient_dict.get('platelets', 200)
    
    albi = calculate_albi(bilirubin, albumin)
    amap_raw = calculate_amap_raw(age, gender, albi, platelets)
    
    # Scale aMAP
    scaler_samap = models['samap']['scaler']
    s_amap_val = scaler_samap.transform(np.array([[amap_raw]]))[0][0]
    
    # XGBoost S_aMAP
    input_samap = pd.DataFrame([{
        'age': age, 'gender': gender, 
        'albumin': albumin / 10.0, 'bilirubin': bilirubin / 17.1, 
        'platelets': platelets, 'ALBI_score': albi, 'S_aMAP': s_amap_val
    }])
    
    prob_samap = models['samap']['model'].predict_proba(input_samap)[0][1]
    
    # ==========================================
    # 2. CORE 2: S_base (Tiền sử Y khoa - NHANES)
    # ==========================================
    sbase_features = models['sbase']['features']
    # Nạp mock healthy làm mặc định
    mock_data_path = './Training/mock_patients.joblib'
    if os.path.exists(mock_data_path):
        patient_sbase = joblib.load(mock_data_path)['healthy']
    else:
        patient_sbase = {f: 2.0 for f in sbase_features}
        
    # Ghi đè lịch sử (Nếu người bệnh truyền vào biến mcq_history = True -> Gán sick)
    if patient_dict.get('mcq_history', False) and os.path.exists(mock_data_path):
        patient_sbase = joblib.load(mock_data_path)['sick']
        
    input_sbase = pd.DataFrame([patient_sbase], columns=sbase_features)
    prob_sbase = models['sbase']['model'].predict_proba(input_sbase)[0][1]
    
    # ==========================================
    # 3. CORE 3: S_dynamic (Sinh hiệu thiết bị đeo)
    # ==========================================
    dynamic_data = {
        'resting_hr': patient_dict.get('resting_hr', 70),
        'hrv': patient_dict.get('hrv', 60),
        'spo2': patient_dict.get('spo2', 98),
        'activity_steps': patient_dict.get('activity_steps', 8000),
        'spider_nevi': patient_dict.get('spider_nevi', 0),
        'jaundice': patient_dict.get('jaundice', 0)
    }
    
    input_sdynamic = pd.DataFrame([dynamic_data])
    prob_sdynamic = models['sdynamic']['model'].predict_proba(input_sdynamic)[0][1]
    
    # ==========================================
    # 4. HỢP NHẤT: R_total
    # ==========================================
    # ==========================================
    # R_total = 100 * (0.57 * S_aMAP + 0.29 * S_base + 0.14 * S_dynamic)
    r_total = 100 * (0.57 * prob_samap + 0.29 * prob_sbase + 0.14 * prob_sdynamic)
    
    # Phân tầng nguy cơ
    if r_total < 30:
        level = "Thấp"
        recommendation = "Theo dõi sức khỏe định kỳ."
    elif r_total < 60:
        level = "Trung bình"
        recommendation = "Khuyến nghị theo dõi sát và điều chỉnh các yếu tố nguy cơ (chế độ ăn, vận động)."
    elif r_total < 80:
        level = "Cao"
        recommendation = "Cảnh báo, khuyến nghị đi khám và đánh giá chuyên môn y tế."
    else:
        level = "Rất cao"
        recommendation = "CẢNH BÁO MẠNH: Đề nghị cơ sở y tế chuyên khoa đánh giá ngay lập tức (Sinh thiết, Siêu âm gan)."
        
    result = {
        "S_aMAP_Risk": float(prob_samap),
        "S_base_Risk": float(prob_sbase),
        "S_dynamic_Risk": float(prob_sdynamic),
        "R_total_Score": float(r_total),
        "Risk_Level": level,
        "Recommendation": recommendation
    }
    return result

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 BẮT ĐẦU CHẠY PIPELINE HỢP NHẤT (R_total) 🚀")
    print("="*50)
    
    # 1. Bệnh nhân khỏe mạnh
    patient_healthy = {
        # Huyết học
        'age': 30, 'gender': 1, 'albumin': 45.0, 'bilirubin': 10.0, 'platelets': 250,
        # Tiền sử
        'mcq_history': False,
        # Sinh hiệu động
        'resting_hr': 65, 'hrv': 70, 'spo2': 99, 'activity_steps': 10000,
        'spider_nevi': 0, 'jaundice': 0
    }
    
    # 2. Bệnh nhân nguy cơ cao
    patient_sick = {
        # Huyết học
        'age': 65, 'gender': 1, 'albumin': 30.0, 'bilirubin': 35.0, 'platelets': 90,
        # Tiền sử (Có ung thư/bệnh gan)
        'mcq_history': True,
        # Sinh hiệu động
        'resting_hr': 95, 'hrv': 20, 'spo2': 92, 'activity_steps': 1500,
        'spider_nevi': 1, 'jaundice': 1
    }
    
    print("\n[+] Đang xử lý Ca số 1 (Bệnh nhân trẻ, khỏe mạnh)...")
    res_healthy = predict_patient_risk(patient_healthy)
    print(json.dumps(res_healthy, indent=4, ensure_ascii=False))
    
    print("\n[+] Đang xử lý Ca số 2 (Bệnh nhân lớn tuổi, tiền sử bệnh nền, sinh hiệu xấu)...")
    res_sick = predict_patient_risk(patient_sick)
    print(json.dumps(res_sick, indent=4, ensure_ascii=False))
    print("="*50)
