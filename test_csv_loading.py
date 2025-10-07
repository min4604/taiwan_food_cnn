#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
獨立的CSV比對測試工具
測試檔案是否能正確載入和比對
"""

import os
import sys

def test_csv_without_pandas():
    """不使用pandas的CSV測試"""
    print("=== 測試CSV檔案載入和比對 ===")
    
    # 檢查檔案
    truth_file = "train_list.csv"
    pred_file = "test_predictions_optimized_amd_npu.csv"
    
    # 檔案存在性檢查
    if not os.path.exists(truth_file):
        print(f"❌ 錯誤: {truth_file} 不存在")
        return False
    if not os.path.exists(pred_file):
        print(f"❌ 錯誤: {pred_file} 不存在")
        return False
    
    print(f"✓ 檔案存在檢查通過")
    
    # 讀取ground truth
    print(f"\n1. 讀取 {truth_file}...")
    truth_mapping = {}
    truth_count = 0
    
    try:
        with open(truth_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',')
                if len(parts) < 3:
                    print(f"⚠️  第{line_num}行格式錯誤: {line}")
                    continue
                
                # 提取檔案名（不含副檔名）
                path = parts[2]
                filename = os.path.splitext(os.path.basename(path))[0].lower()
                category = parts[1]
                
                truth_mapping[filename] = category
                truth_count += 1
                
                if line_num <= 5:  # 顯示前5行
                    print(f"   第{line_num}行: {parts[0]},{parts[1]},{parts[2]} -> key='{filename}'")
        
        print(f"✓ 載入 {truth_count} 個ground truth項目")
        
    except Exception as e:
        print(f"❌ 載入ground truth失敗: {e}")
        return False
    
    # 讀取predictions
    print(f"\n2. 讀取 {pred_file}...")
    pred_count = 0
    match_count = 0
    error_count = 0
    
    try:
        with open(pred_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num == 1:  # 跳過標題行
                    print(f"   標題行: {line.strip()}")
                    continue
                
                line = line.strip()
                if not line:
                    continue
                
                # 分割CSV（注意路徑可能包含逗號）
                parts = line.split(',', 3)
                if len(parts) < 4:
                    print(f"⚠️  第{line_num}行格式錯誤: {line}")
                    continue
                
                # 提取資料
                pred_id = parts[0]
                pred_category = parts[1]
                confidence = parts[2]
                path = parts[3]
                
                # 提取檔案名
                filename = os.path.splitext(os.path.basename(path))[0].lower()
                
                # 檢查是否有對應的ground truth
                if filename in truth_mapping:
                    true_category = truth_mapping[filename]
                    match_count += 1
                    
                    if pred_category != true_category:
                        error_count += 1
                        if error_count <= 3:  # 只顯示前3個錯誤
                            print(f"   ❌ 分類錯誤: '{filename}' 預測={pred_category}, 實際={true_category}")
                    else:
                        if match_count <= 3:  # 只顯示前3個正確的
                            print(f"   ✓ 分類正確: '{filename}' 類別={pred_category}")
                else:
                    if pred_count - match_count < 3:  # 只顯示前3個找不到的
                        print(f"   ⚠️  找不到對應: '{filename}' (來源: {os.path.basename(path)})")
                
                pred_count += 1
                
                # 限制處理數量以避免輸出過多
                if pred_count >= 100:
                    break
        
        print(f"\n=== 測試結果 ===")
        print(f"Ground Truth項目: {truth_count}")
        print(f"Predictions項目: {pred_count}")
        print(f"成功比對: {match_count}")
        print(f"比對率: {match_count/pred_count*100:.1f}%")
        print(f"分類錯誤: {error_count}")
        if match_count > 0:
            print(f"準確率: {(match_count-error_count)/match_count*100:.1f}%")
        
        if match_count == 0:
            print("❌ 沒有成功比對任何項目！")
            print("可能原因:")
            print("- 檔案名格式不匹配")
            print("- 路徑結構差異")
            print("- 檔案編碼問題")
            return False
        else:
            print("✅ CSV檔案載入和比對測試成功！")
            return True
            
    except Exception as e:
        print(f"❌ 載入predictions失敗: {e}")
        return False

if __name__ == "__main__":
    success = test_csv_without_pandas()
    if success:
        print("\n🎉 測試通過！您的tool.py應該能正常工作")
        print("可以執行: python tool.py --ground-truth train_list.csv --predictions test_predictions_optimized_amd_npu.csv")
    else:
        print("\n💥 測試失敗！需要檢查CSV檔案格式")
    
    sys.exit(0 if success else 1)