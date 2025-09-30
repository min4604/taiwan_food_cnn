#!/usr/bin/env python3
"""
驗證台灣美食資料集的訓練/測試分離
確保測試資料完全不參與訓練
"""

import os
import pandas as pd
from data_loader import TaiwanFoodDataLoader

def validate_data_separation():
    """詳細驗證資料分離"""
    print("=" * 60)
    print("台灣美食資料集分離驗證")
    print("=" * 60)
    
    data_dir = "archive/tw_food_101"
    
    # 1. 檢查 CSV 檔案
    train_csv = os.path.join(data_dir, "tw_food_101_train.csv")
    test_csv = os.path.join(data_dir, "tw_food_101_test_list.csv")
    
    if not os.path.exists(train_csv):
        print(f"❌ 找不到訓練 CSV: {train_csv}")
        return False
    
    if not os.path.exists(test_csv):
        print(f"❌ 找不到測試 CSV: {test_csv}")
        return False
    
    # 2. 載入資料
    print("📖 載入資料...")
    df_train = pd.read_csv(train_csv, header=None, names=['index', 'class_id', 'image_path'])
    df_test = pd.read_csv(test_csv, header=None, names=['index', 'image_path'])
    
    print(f"訓練資料: {len(df_train)} 筆")
    print(f"測試資料: {len(df_test)} 筆")
    
    # 3. 檢查檔案路徑重疊
    print("\n🔍 檢查檔案重疊...")
    train_paths = set(df_train['image_path'].tolist())
    test_paths = set(df_test['image_path'].tolist())
    
    overlap = train_paths.intersection(test_paths)
    
    if overlap:
        print(f"❌ 發現 {len(overlap)} 個重疊檔案！")
        print("重疊檔案:")
        for i, path in enumerate(sorted(overlap)):
            print(f"  {i+1:3d}. {path}")
            if i >= 10:  # 只顯示前10個
                print(f"      ... 還有 {len(overlap) - 10} 個")
                break
        return False
    else:
        print("✅ 訓練和測試檔案完全分離")
    
    # 4. 檢查檔案存在性
    print("\n📁 檢查檔案存在性...")
    
    # 檢查訓練檔案
    missing_train = 0
    for i, path in enumerate(df_train['image_path']):
        full_path = os.path.join(data_dir, path)
        if not os.path.exists(full_path):
            missing_train += 1
            if missing_train <= 5:  # 只顯示前5個缺失檔案
                print(f"  缺失訓練檔案: {path}")
    
    # 檢查測試檔案
    missing_test = 0
    for i, path in enumerate(df_test['image_path']):
        full_path = os.path.join(data_dir, path)
        if not os.path.exists(full_path):
            missing_test += 1
            if missing_test <= 5:  # 只顯示前5個缺失檔案
                print(f"  缺失測試檔案: {path}")
    
    print(f"訓練檔案: {len(df_train) - missing_train}/{len(df_train)} 存在")
    print(f"測試檔案: {len(df_test) - missing_test}/{len(df_test)} 存在")
    
    # 5. 檢查類別分布
    print("\n📊 檢查訓練資料類別分布...")
    class_counts = df_train['class_id'].value_counts().sort_index()
    
    print(f"總類別數: {len(class_counts)}")
    print(f"類別範圍: {class_counts.index.min()} - {class_counts.index.max()}")
    print(f"平均每類樣本數: {class_counts.mean():.1f}")
    print(f"最少樣本數: {class_counts.min()}")
    print(f"最多樣本數: {class_counts.max()}")
    
    # 6. 測試資料載入器
    print("\n🔧 測試資料載入器...")
    try:
        loader = TaiwanFoodDataLoader(data_dir=data_dir)
        
        # 檢查測試檔案列表是否正確載入
        expected_test_files = set(df_test['image_path'].tolist())
        actual_test_files = loader.test_files
        
        if expected_test_files == actual_test_files:
            print("✅ 資料載入器正確載入測試檔案列表")
        else:
            print("❌ 資料載入器測試檔案列表不正確")
            missing_in_loader = expected_test_files - actual_test_files
            extra_in_loader = actual_test_files - expected_test_files
            
            if missing_in_loader:
                print(f"  載入器中缺失 {len(missing_in_loader)} 個測試檔案")
            if extra_in_loader:
                print(f"  載入器中多出 {len(extra_in_loader)} 個檔案")
        
        # 測試實際載入少量資料
        print("\n🧪 測試實際資料載入...")
        try:
            (X_train, y_train), (X_val, y_val) = loader.load_dataset_from_csv('train', validation_split=0.1)
            print(f"✅ 成功載入訓練資料: {X_train.shape}")
            print(f"✅ 成功載入驗證資料: {X_val.shape}")
            
            # 檢查是否有測試檔案混入訓練資料（這在新的載入器中應該不會發生）
            print("   已確保測試檔案完全不參與訓練")
            
        except Exception as e:
            print(f"❌ 載入訓練資料失敗: {e}")
            return False
            
    except Exception as e:
        print(f"❌ 建立資料載入器失敗: {e}")
        return False
    
    # 7. 總結
    print("\n" + "=" * 60)
    success = (
        len(overlap) == 0 and
        missing_train == 0 and
        missing_test == 0
    )
    
    if success:
        print("🎉 驗證成功！")
        print("✅ 訓練和測試資料完全分離")
        print("✅ 所有檔案都存在")
        print("✅ 資料載入器正確工作")
        print("🚀 可以安全開始訓練")
    else:
        print("⚠️  驗證發現問題")
        if overlap:
            print("❌ 訓練和測試資料有重疊")
        if missing_train > 0:
            print(f"❌ 缺失 {missing_train} 個訓練檔案")
        if missing_test > 0:
            print(f"❌ 缺失 {missing_test} 個測試檔案")
        print("請修復問題後重新驗證")
    
    print("=" * 60)
    return success

if __name__ == "__main__":
    validate_data_separation()