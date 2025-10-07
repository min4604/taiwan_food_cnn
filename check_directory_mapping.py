import pandas as pd
import os

# 讀取CSV檔案
csv_path = r'c:\Users\Chen\Desktop\project\taiwan_food_cnn\archive\tw_food_101\tw_food_101_train.csv'
train_dir = r'c:\Users\Chen\Desktop\project\taiwan_food_cnn\archive\tw_food_101\train'

df = pd.read_csv(csv_path, header=None)
csv_categories = set()

# 從CSV中提取所有類別名稱
for idx, row in df.iterrows():
    img_path = row[2]  # train/category/filename.jpg
    if img_path.startswith('train/'):
        relative_path = img_path[6:]  # 移除 'train/' 前綴
        category_name = relative_path.split('/')[0]
        csv_categories.add(category_name)

# 取得實際目錄名稱
actual_dirs = set(os.listdir(train_dir))

# 找出不匹配的目錄
csv_only = csv_categories - actual_dirs
actual_only = actual_dirs - csv_categories

print('CSV中有但實際目錄沒有的類別:')
for cat in sorted(csv_only):
    print(f'  {cat}')

print('\n實際目錄有但CSV中沒有的類別:')
for cat in sorted(actual_only):
    print(f'  {cat}')

print('\n可能需要映射的配對:')
for csv_cat in sorted(csv_only):
    for actual_cat in sorted(actual_only):
        # 檢查是否只是下劃線和連字號的差異
        if csv_cat.replace('_', '-') == actual_cat:
            print(f'  "{csv_cat}": "{actual_cat}"')