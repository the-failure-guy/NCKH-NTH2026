with open('integrated_app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open('integrated_app.py', 'w', encoding='utf-8') as f:
    for line in lines:
        if "70,000 hồ sơ bệnh án từ CDC Hoa Kỳ" not in line:
            f.write(line)

print("Removed.")
