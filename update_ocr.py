import re

with open('integrated_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove get_ocr_reader
content = re.sub(r'@st\.cache_resource\s*\ndef get_ocr_reader\(\):\s*\n\s*import easyocr\s*\n\s*return easyocr\.Reader\(\[\'vi\', \'en\'\]\)\s*\n', '', content)

# Replace extract_blood_test
new_extract_func = """def extract_blood_test(image_bytes):
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
"""

content = re.sub(r'def extract_blood_test\(image_bytes\):.*?return extracted', new_extract_func, content, flags=re.DOTALL)

with open('integrated_app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated OCR.")
