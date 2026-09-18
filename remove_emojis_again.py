import re

with open('integrated_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

emojis = ['🏥', '🩸', '📝', '📷', '🔍', '❌', '✅', '📋', '🌿', '⚠️', '📂', '⌚', '🚀', '🎯', '🧠', '🔵', '🚨', '💬', '🤖', '🩺']
for e in emojis:
    content = content.replace(e + ' ', '')
    content = content.replace(e, '')

with open('integrated_app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done.")
