with open('integrated_app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open('integrated_app.py', 'w', encoding='utf-8') as f:
    for line in lines:
        if "Để AI phân tích chính xác hơn, xin hãy chia sẻ một chút về tiền sử sức khỏe của bệnh nhân nhé!" not in line:
            f.write(line)

print("Removed.")
