import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from samap_pipeline import calculate_albi, calculate_amap_raw

# Cấu hình giao diện
st.set_page_config(page_title="AI Care Assistant - aMAP Predictor", page_icon="⚕️", layout="centered")

st.title("⚕️ AI Care Assistant")
st.subheader("Hệ thống dự báo nguy cơ dựa trên chỉ số Sinh hóa & aMAP")
st.markdown("Hệ thống sử dụng mô hình học máy **XGBoost** (ROC-AUC 0.70) kết hợp trực tiếp với thang điểm **aMAP** để đưa ra cảnh báo sớm về các bệnh lý nguy hiểm về gan.")
st.markdown("---")

# Tải model vào bộ nhớ tạm (Cache) để app phản hồi nhanh hơn
@st.cache_resource
def load_model():
    model_path = './Training/samap_model.joblib'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

artifacts = load_model()

if artifacts is None:
    st.error("Chưa tìm thấy mô hình đã huấn luyện (samap_model.joblib). Vui lòng huấn luyện mô hình trước!")
    st.stop()

xgb_model = artifacts['model']
scaler = artifacts['scaler']

# Xây dựng thanh công cụ bên trái (Sidebar)
st.sidebar.header("📋 Nhập thông tin Bệnh nhân")
age = st.sidebar.slider("Tuổi", min_value=1, max_value=120, value=55)
gender = st.sidebar.radio("Giới tính", ["Nam", "Nữ"])
gender_encoded = 1 if gender == "Nam" else 0

albumin = st.sidebar.number_input("Albumin (g/L)", min_value=1.0, max_value=100.0, value=35.5)
bilirubin = st.sidebar.number_input("Bilirubin toàn phần (µmol/L)", min_value=1.0, max_value=200.0, value=18.2)
platelets = st.sidebar.number_input("Số lượng Tiểu cầu (10^9/L)", min_value=10.0, max_value=1000.0, value=150.0)

# Khung chứa kết quả chính
if st.sidebar.button("Tiến hành Chẩn đoán bằng AI 🚀", use_container_width=True):
    with st.spinner("AI đang phân tích dữ liệu..."):
        # 1. Tính toán các chỉ số y khoa
        albi = calculate_albi(bilirubin, albumin)
        amap_raw = calculate_amap_raw(age, gender_encoded, albi, platelets)
        
        # 2. Chuẩn hóa S_aMAP bằng scaler đã train
        s_amap = scaler.transform(np.array([[amap_raw]]))[0][0]
        
        # 3. Chuẩn bị dữ liệu đầu vào cho XGBoost
        # Note: XGBoost model cần thứ tự cột giống như lúc train: 
        # ['age', 'gender', 'albumin', 'bilirubin', 'platelets', 'ALBI_score', 'S_aMAP']
        input_data = pd.DataFrame([{
            'age': age,
            'gender': gender_encoded,
            'albumin': albumin,
            'bilirubin': bilirubin,
            'platelets': platelets,
            'ALBI_score': albi,
            'S_aMAP': s_amap
        }])
        
        # 4. Đưa ra dự đoán
        prob = xgb_model.predict_proba(input_data)[0][1]
        
        # 5. Hiển thị bảng điều khiển kết quả
        st.markdown("### 📊 Các chỉ số y khoa đã phân tích")
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Chỉ số ALBI", value=f"{albi:.2f}")
        col2.metric(label="aMAP Thô", value=f"{amap_raw:.2f}")
        col3.metric(label="S_aMAP (Đã chuẩn hóa)", value=f"{s_amap:.3f}")
        
        st.markdown("---")
        st.markdown("### 🧠 Kết luận của AI Care Assistant")
        
        # Vẽ thanh phần trăm nguy cơ
        st.progress(float(prob), text=f"Xác suất rủi ro: {prob*100:.1f}%")
        
        if prob > 0.5:
            st.error(f"⚠️ **CẢNH BÁO:** Phát hiện nguy cơ mắc bệnh lý gan tiến triển. Đề nghị chỉ định làm thêm xét nghiệm chuyên sâu hoặc siêu âm.")
        else:
            st.success(f"✅ **AN TOÀN:** Nguy cơ hiện tại rất thấp. Tiếp tục theo dõi sức khỏe định kỳ.")
else:
    st.info("👈 Hãy nhập dữ liệu xét nghiệm sinh hóa vào thanh bên trái và nhấn **'Tiến hành Chẩn đoán'** để xem kết quả.")
