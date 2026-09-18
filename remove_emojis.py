import re

with open('integrated_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Xóa dòng cụ thể
line_to_remove = r'st\.markdown\("Hệ thống kết hợp 3 lõi AI: \*\*Huyết học \(\$S_\{aMAP\}\$\)\*\*, \*\*Bệnh án điện tử \(\$S_\{base\}\$\)\*\* và \*\*Sinh hiệu thông minh \(\$S_\{dynamic\}\$\)\*\* để đưa ra tổng điểm rủi ro ung thư/bệnh gan \(\$R_\{total\}\$\)\."\)\n'
content = re.sub(line_to_remove, '', content)

# Regex để xóa các emoji (chỉ để lại ASCII và một số ký tự đặc biệt tiếng Việt hợp lệ)
# Một cách an toàn hơn: chỉ replace các emoji cụ thể mà ta biết
emojis = ['🏥', '🩸', '📝', '📷', '🔍', '❌', '✅', '📋', '🌿', '⚠️', '📂', '⌚', '🚀', '🎯', '🧠', '🔵', '🚨']
for e in emojis:
    content = content.replace(e + ' ', '')
    content = content.replace(e, '')

with open('integrated_app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done.")
