import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import re
import requests
from PIL import Image
from samap_pipeline import calculate_albi, calculate_amap_raw

# Cấu hình giao diện
st.set_page_config(page_title="Hệ thống AI Care Assistant", page_icon="", layout="wide")

st.title("AI Care Assistant - Chẩn đoán Y khoa Đa mô thức (Multi-modal)")
st.markdown("---")

# ==========================================
# KHỞI TẠO CACHE & SESSION STATE
# ==========================================
@st.cache_resource
def load_models():
    samap_path = './Training/samap_model.joblib'
    sbase_path = './Training/Model/MedicalCondition/sbase_model.joblib'
    sdynamic_path = './Training/Model/DynamicSignal/sdynamic_model.joblib'
    
    samap_artifacts = joblib.load(samap_path) if os.path.exists(samap_path) else None
    sbase_artifacts = joblib.load(sbase_path) if os.path.exists(sbase_path) else None
    sdynamic_artifacts = joblib.load(sdynamic_path) if os.path.exists(sdynamic_path) else None
    
    return samap_artifacts, sbase_artifacts, sdynamic_artifacts

samap_artifacts, sbase_artifacts, sdynamic_artifacts = load_models()

if not all([samap_artifacts, sbase_artifacts, sdynamic_artifacts]):
    st.error("Không tìm thấy đầy đủ 3 mô hình đã huấn luyện. Vui lòng kiểm tra lại Pipeline.")
    st.stop()

if 'albumin_val' not in st.session_state: st.session_state.albumin_val = 35.5
if 'bilirubin_val' not in st.session_state: st.session_state.bilirubin_val = 18.2
if 'platelets_val' not in st.session_state: st.session_state.platelets_val = 150.0
if 'last_uploaded_file' not in st.session_state: st.session_state.last_uploaded_file = None
if 'chat_messages' not in st.session_state: st.session_state.chat_messages = []

# ==========================================
# HÀM XỬ LÝ OCR
# ==========================================
def extract_blood_test(image_bytes):
    from google import genai
    import json
    import io
    from PIL import Image
    import streamlit as st
    
    if "GEMINI_API_KEY" not in st.secrets:
        return {}
        
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    img = Image.open(io.BytesIO(image_bytes))
    
    prompt = '''
    Trích xuất 3 chỉ số xét nghiệm máu sau từ hình ảnh:
    1. Albumin (g/L)
    2. Bilirubin toàn phần (µmol/L) 
    3. Tiểu cầu / Platelet / PLT (G/L)
    
    Chỉ trả về MỘT chuỗi JSON hợp lệ theo định dạng sau, không kèm bất kỳ văn bản nào khác:
    {"albumin": 45.2, "bilirubin": 12.5, "platelets": 250.0}
    Nếu không tìm thấy chỉ số nào, hãy bỏ qua hoặc để null.
    '''
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[img, prompt]
        )
        text = response.text.strip()
        if text.startswith("```json"): text = text[7:]
        if text.endswith("```"): text = text[:-3]
        return json.loads(text.strip())
    except Exception as e:
        print("Lỗi OCR Gemini:", e)
        return {}


from google import genai

# ==========================================
# HÀM CALL API ĐÁM MÂY (Google Gemini / Gemma)
# ==========================================
def ask_gemma(prompt, history):
    # Lấy API Key từ hệ thống Secrets của Streamlit Cloud
    if "GEMINI_API_KEY" not in st.secrets:
        return "⚠️ Lỗi: Chưa cấu hình GEMINI_API_KEY trong Streamlit Cloud Secrets."
    
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    
    # Định dạng lại lịch sử chat cho Gemini
    formatted_history = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        formatted_history.append({"role": role, "parts": [{"text": msg["content"]}]})
        
    try:
        # Sử dụng mô hình gemini-2.5-flash siêu tốc (thế hệ mới nhất)
        chat = client.chats.create(
            model="gemini-2.5-flash",
            config={"system_instruction": "Bạn là Trợ lý Bác sĩ AI của hệ thống AI Care Assistant. Hãy tư vấn sức khỏe ngắn gọn, dễ hiểu và chuyên nghiệp bằng Tiếng Việt."},
            history=formatted_history
        )
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        return f" Lỗi kết nối API: {str(e)}"

# ==========================================
# GIAO DIỆN CHÍNH
# ==========================================
col1, col2, col3 = st.columns(3)

with col1:
    st.header("Huyết học & Sinh hóa")
    
    age = st.number_input("Tuổi", min_value=1, max_value=120, value=55)
    gender = st.selectbox("Giới tính", ["Nam", "Nữ"])
    gender_encoded = 1 if gender == "Nam" else 0
    
    input_method = st.radio("Phương thức nhập liệu Sinh hóa:", ["Nhập thủ công", "Quét ảnh tự động (OCR)"])
    
    uploaded_file = None
    if input_method == "Quét ảnh tự động (OCR)":
        uploaded_file = st.file_uploader("Tải lên Phiếu Xét nghiệm", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None and uploaded_file.name != st.session_state.last_uploaded_file:
            with st.spinner("Đang đọc giấy xét nghiệm bằng AI Vision..."):
                image_bytes = uploaded_file.read()
                extracted_data = extract_blood_test(image_bytes)
                
                if len(extracted_data) == 0:
                    st.error("File giấy xét nghiệm không hợp lệ (Không tìm thấy chỉ số).")
                else:
                    st.success(f"Đã quét thành công {len(extracted_data)} chỉ số!")
                    if 'albumin' in extracted_data: st.session_state.albumin_val = extracted_data['albumin']
                    if 'bilirubin' in extracted_data: st.session_state.bilirubin_val = extracted_data['bilirubin']
                    if 'platelets' in extracted_data: st.session_state.platelets_val = extracted_data['platelets']
                
                st.session_state.last_uploaded_file = uploaded_file.name
                
    albumin = float(st.session_state.albumin_val)
    bilirubin = float(st.session_state.bilirubin_val)
    platelets = float(st.session_state.platelets_val)
    
    if input_method == "Nhập thủ công" or uploaded_file is not None:
        st.markdown("**Kết quả Xét nghiệm (Có thể chỉnh sửa nếu AI quét sai):**" if input_method != "Nhập thủ công" else "**Chỉ số Xét nghiệm:**")
        albumin = st.number_input("Albumin (g/L)", min_value=1.0, max_value=100.0, value=float(st.session_state.albumin_val))
        bilirubin = st.number_input("Bilirubin (µmol/L)", min_value=1.0, max_value=200.0, value=float(st.session_state.bilirubin_val))
        platelets = st.number_input("Tiểu cầu (10^9/L)", min_value=10.0, max_value=1000.0, value=float(st.session_state.platelets_val))
        
        st.session_state.albumin_val = albumin
        st.session_state.bilirubin_val = bilirubin
        st.session_state.platelets_val = platelets

with col2:
    st.header("Thông tin Bệnh sử (EMR)")
    
    emr_options = [
        "Hoàn toàn khỏe mạnh (Không có tiền sử bệnh lý cá nhân/gia đình)",
        "Thuộc nhóm nguy cơ (Từng mắc bệnh gan, ung thư, hoặc có tiền sử gia đình)"
    ]
    emr_mode = st.radio("Bệnh nhân thuộc nhóm nào dưới đây?", emr_options)
    

with col3:
    st.header("Sinh hiệu Động")
    st.info("Dữ liệu từ Smartwatch & Lâm sàng.")
    
    resting_hr = st.slider("Nhịp tim nghỉ (bpm)", 40, 150, 75)
    spo2 = st.slider("Nồng độ Oxy máu SpO2 (%)", 80, 100, 98)
    hrv = st.slider("Biến thiên nhịp tim HRV (ms)", 10, 100, 45)
    activity_steps = st.number_input("Số bước chân hôm nay", 0, 30000, 8000)
    
    st.write("**Triệu chứng lâm sàng:**")
    spider_nevi = st.checkbox("Có dấu sao mạch")
    jaundice = st.checkbox("Có dấu hiệu vàng da")

st.markdown("---")

# ==========================================
# KHU VỰC PHÂN TÍCH R_TOTAL VÀ CHATBOT Y KHOA
# ==========================================
res_col, chat_col = st.columns([2, 1.2])

with res_col:
    if st.button("KÍCH HOẠT HỢP NHẤT ĐA MÔ THỨC", use_container_width=True):
        with st.spinner("Đang xử lý song song 3 mạng lưới nơ-ron..."):
            albi = calculate_albi(bilirubin, albumin)
            amap_raw = calculate_amap_raw(age, gender_encoded, albi, platelets)
            s_amap_val = samap_artifacts['scaler'].transform(np.array([[amap_raw]]))[0][0]
            
            input_samap = pd.DataFrame([{
                'age': age, 'gender': gender_encoded, 
                'albumin': albumin / 10.0, 'bilirubin': bilirubin / 17.1, 
                'platelets': platelets, 'ALBI_score': albi, 'S_aMAP': s_amap_val
            }])
            prob_samap = samap_artifacts['model'].predict_proba(input_samap)[0][1]
            
            sbase_features = sbase_artifacts['features']
            mock_data_path = './Training/mock_patients.joblib'
            if os.path.exists(mock_data_path):
                mocks = joblib.load(mock_data_path)
                patient_sbase = mocks['sick'] if emr_mode == emr_options[1] else mocks['healthy']
            else:
                patient_sbase = {f: 2.0 for f in sbase_features}
                
            input_sbase = pd.DataFrame([patient_sbase], columns=sbase_features)
            prob_sbase = sbase_artifacts['model'].predict_proba(input_sbase)[0][1]
            
            def scale_hr(hr): return np.clip((hr - 60) / 40.0, 0.0, 1.0)
            def scale_hrv(hrv): return np.clip((60 - hrv) / 40.0, 0.0, 1.0)
            def scale_spo2(spo2): return np.clip((98 - spo2) / 8.0, 0.0, 1.0)
            def scale_activity(steps): return np.clip((8000 - steps) / 6000.0, 0.0, 1.0)
            
            s_symptom = ( (1.0 if spider_nevi else 0.0) + (1.0 if jaundice else 0.0) ) / 2.0
            prob_sdynamic = float((0.33 * s_symptom) + (0.27 * scale_hr(resting_hr)) + \
                                  (0.20 * scale_activity(activity_steps)) + (0.10 * scale_hrv(hrv)) + \
                                  (0.10 * scale_spo2(spo2)))
            
            r_total = 100 * (0.44 * prob_samap + 0.34 * prob_sbase + 0.22 * prob_sdynamic)
            
            st.header("KẾT QUẢ CHẨN ĐOÁN LÂM SÀNG TỔNG THỂ")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Rủi ro Huyết học (S_aMAP)", f"{prob_samap*100:.1f}%")
            m2.metric("Rủi ro Tiền sử (S_base)", f"{prob_sbase*100:.1f}%")
            m3.metric("Rủi ro Sinh hiệu (S_dynamic)", f"{prob_sdynamic*100:.1f}%")
            
            st.markdown("---")
            st.subheader(f"TỔNG ĐIỂM NGUY CƠ (R_total): {r_total:.1f} / 100")
            
            if r_total < 30:
                st.success("ĐÁNH GIÁ: MỨC ĐỘ THẤP. Bệnh nhân có rủi ro thấp, chỉ cần theo dõi sức khỏe định kỳ.")
                st.progress(float(r_total / 100))
            elif r_total < 60:
                st.info("ĐÁNH GIÁ: MỨC ĐỘ TRUNG BÌNH. Khuyến nghị theo dõi sát và điều chỉnh chế độ ăn uống, vận động.")
                st.progress(float(r_total / 100))
            elif r_total < 80:
                st.warning("ĐÁNH GIÁ: MỨC ĐỘ CAO. Cảnh báo nguy cơ, khuyến nghị đi khám và đánh giá chuyên môn y tế sớm.")
                st.progress(float(r_total / 100))
            else:
                st.error("ĐÁNH GIÁ: MỨC ĐỘ RẤT CAO. CẢNH BÁO MẠNH! Đề nghị cơ sở y tế chuyên khoa đánh giá ngay lập tức (Sinh thiết, Siêu âm gan).")
                st.progress(float(r_total / 100))

with chat_col:
    st.markdown("""
    <style>
    /* Chat Bubbles CSS */
    .chat-row {
        display: flex;
        margin-bottom: 12px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .row-user { justify-content: flex-end; }
    .row-assistant { justify-content: flex-start; }
    .chat-bubble {
        padding: 12px 16px;
        border-radius: 18px;
        max-width: 90%;
        font-size: 15px;
        line-height: 1.5;
        box-shadow: 0px 2px 4px rgba(0,0,0,0.1);
    }
    .bubble-user {
        background-color: #007AFF;
        color: white;
        border-bottom-right-radius: 4px;
    }
    .bubble-assistant {
        background-color: #E9ECEF;
        color: #212529;
        border-bottom-left-radius: 4px;
        border: 1px solid #DEE2E6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.expander("TRỢ LÝ BÁC SĨ AI (GEMMA) - Bấm để thu gọn/mở rộng", expanded=True):
        st.caption("Trực tuyến - Sẵn sàng tư vấn Y khoa")
        
        chat_box = st.container(height=400)
        with chat_box:
            for msg in st.session_state.chat_messages:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-row row-user"><div class="chat-bubble bubble-user">{msg["content"]}</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-row row-assistant"><div class="chat-bubble bubble-assistant"><b>Bác sĩ AI:</b><br>{msg["content"]}</div></div>', unsafe_allow_html=True)
                    
        with st.form("chat_form", clear_on_submit=True, border=False):
            c1, c2 = st.columns([4, 1])
            prompt = c1.text_input("Hỏi bác sĩ...", label_visibility="collapsed", placeholder="Ví dụ: Rủi ro của tôi là gì?")
            submitted = c2.form_submit_button("Gửi ")
            
        if submitted and prompt:
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with chat_box:
                st.markdown(f'<div class="chat-row row-user"><div class="chat-bubble bubble-user">{prompt}</div></div>', unsafe_allow_html=True)
                with st.spinner("Đang chẩn đoán..."):
                    response = ask_gemma(prompt, st.session_state.chat_messages[:-1])
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    st.rerun()
