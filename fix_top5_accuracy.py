#!/usr/bin/env python3
"""
修復 cnn_model.py 中的 top_5_accuracy 問題
"""

# 讀取檔案
with open('cnn_model.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 在頂部加入需要的匯入
if 'from tensorflow.keras.metrics import TopKCategoricalAccuracy' not in content:
    # 找到 import 區域的結尾
    lines = content.split('\n')
    import_end = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            import_end = i
    
    # 在 import 區域後插入新的 import
    lines.insert(import_end + 1, 'from tensorflow.keras.metrics import TopKCategoricalAccuracy')
    content = '\n'.join(lines)

# 替換 top_5_accuracy
content = content.replace(
    "'top_5_accuracy'",
    "TopKCategoricalAccuracy(k=5, name='top_5_accuracy')"
)

# 寫回檔案
with open('cnn_model.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ 已修復 top_5_accuracy 問題")