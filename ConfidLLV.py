import pandas as pd
import os
from pathlib import Path

# 修正：使用正斜線或原始字串
df = pd.read_csv('downloads/tt.csv')  # 方法1：使用正斜線
# 或者使用：df = pd.read_csv(r'downloads\tt.csv')  # 方法2：原始字串

# 查看檔案內容
print("原始資料：")
print(df.head())
print(f"\n總共有 {len(df)} 筆資料")
print(f"\nConfidence值範圍：{df['Confidence'].min()} - {df['Confidence'].max()}")

# 根據Confidence由低到高排序
df_sorted = df.sort_values('Confidence', ascending=True).reset_index(drop=True)

# 顯示排序後的資料
print("根據Confidence排序後的資料：")
print(df_sorted)

# 為每個檔案生成新的檔案名稱（根據排序後的順序）
df_sorted['New_Filename'] = df_sorted.index.map(lambda x: f"file_{x+1:04d}")

# 保留原始路徑，只更改檔案名稱
def generate_new_path(row):
    original_path = Path(row['Path'])
    parent_dir = original_path.parent
    extension = original_path.suffix
    new_filename = row['New_Filename'] + extension
    return str(parent_dir / new_filename)

df_sorted['New_Path'] = df_sorted.apply(generate_new_path, axis=1)

# 顯示重新命名的對應關係
print("檔案重新命名對應表：")
for i, row in df_sorted.iterrows():
    print(f"Confidence: {row['Confidence']:.4f}")
    print(f"原始路徑: {row['Path']}")
    print(f"新路徑: {row['New_Path']}")
    print("-" * 50)

# 儲存重新命名的結果到新的CSV檔案
output_df = df_sorted[['Path', 'New_Path', 'Confidence', 'New_Filename']].copy()
output_df.to_csv('renamed_files.csv', index=False, encoding='utf-8-sig')

print("重新命名完成！")
print(f"結果已儲存到 'renamed_files.csv'")
print(f"\n統計資訊：")
print(f"最低Confidence: {df_sorted['Confidence'].iloc[0]:.4f}")
print(f"最高Confidence: {df_sorted['Confidence'].iloc[-1]:.4f}")
print(f"總檔案數: {len(df_sorted)}")

def rename_files(df, dry_run=True):
    """
    實際重新命名檔案
    dry_run=True: 只顯示會執行的操作，不實際重新命名
    dry_run=False: 實際執行重新命名
    """
    success_count = 0
    error_count = 0
    
    for i, row in df.iterrows():
        old_path = row['Path']
        new_path = row['New_Path']
        
        try:
            if dry_run:
                print(f"預演：{old_path} -> {new_path}")
            else:
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
                    print(f"已重新命名：{old_path} -> {new_path}")
                    success_count += 1
                else:
                    print(f"檔案不存在：{old_path}")
                    error_count += 1
        except Exception as e:
            print(f"重新命名失敗：{old_path} -> {new_path}, 錯誤：{e}")
            error_count += 1
    
    if not dry_run:
        print(f"\n重新命名完成！成功：{success_count}，失敗：{error_count}")

# 預演重新命名（不實際執行）
print("=== 重新命名預演 ===")
rename_files(df_sorted, dry_run=True)

print("\n如果要實際執行重新命名，請將 dry_run 設為 False")
print("建議先備份檔案！")

# 在檔案最後加入這行，讓您可以選擇是否實際執行
choice = input("\n是否要實際執行重新命名？(y/N): ").lower()
if choice == 'y':
    print("開始實際重新命名...")
    rename_files(df_sorted, dry_run=False)
else:
    print("已取消實際重新命名")