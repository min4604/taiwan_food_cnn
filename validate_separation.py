#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
驗證訓練集與測試集分離性腳本

功能：
1. 計算測試集中所有圖片的感知雜湊 (perceptual hash)。
2. 遍歷訓練集，計算每張圖片的雜湊，並與測試集進行比對。
3. 找出與測試集圖片完全相同或高度相似的訓練圖片。
4. 產生報告，並提供可選的自動刪除功能。

使用此腳本可以避免訓練資料中包含測試資料，確保模型評估的公正性。
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict

try:
    from PIL import Image
    import imagehash
except ImportError:
    print("缺少必要的套件，請執行: pip install Pillow imagehash")
    exit(1)

def compute_hashes(directory: Path, hash_size: int = 8) -> dict:
    """
    計算指定目錄下所有圖片的感知雜湊值。
    
    返回一個字典，key 是雜湊值，value 是對應的檔案路徑列表。
    """
    hashes = defaultdict(list)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    
    print(f"正在計算 '{directory}' 中的圖片雜湊值...")
    
    image_files = [p for p in directory.rglob('*') if p.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"警告：在 '{directory}' 中找不到任何圖片檔案。")
        return {}

    for i, img_path in enumerate(image_files):
        try:
            with Image.open(img_path) as img:
                # 使用 average hash，速度快且效果好
                h = imagehash.average_hash(img, hash_size=hash_size)
                hashes[h].append(str(img_path))
        except Exception as e:
            print(f"無法處理檔案 {img_path}: {e}")
        
        # 顯示進度
        if (i + 1) % 200 == 0 or (i + 1) == len(image_files):
            print(f"  已處理 {i + 1}/{len(image_files)} 張圖片", end='\r')
            
    print(f"\n完成！共計算了 {len(image_files)} 張圖片，得到 {len(hashes)} 個獨立雜湊。")
    return hashes

def find_duplicates(test_hashes: dict, train_dir: Path, similarity_threshold: int = 5, hash_size: int = 8):
    """
    在訓練集中尋找與測試集重複或相似的圖片。
    """
    duplicates = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    
    print(f"\n正在掃描訓練目錄 '{train_dir}' 並與測試集比對...")
    
    train_image_files = [p for p in train_dir.rglob('*') if p.suffix.lower() in image_extensions]

    if not train_image_files:
        print(f"警告：在 '{train_dir}' 中找不到任何圖片檔案。")
        return []

    test_hash_keys = list(test_hashes.keys())

    for i, train_img_path in enumerate(train_image_files):
        try:
            with Image.open(train_img_path) as img:
                train_hash = imagehash.average_hash(img, hash_size=hash_size)
                
                # 1. 檢查精確重複
                if train_hash in test_hashes:
                    match_info = {
                        "train_image": str(train_img_path),
                        "test_images": test_hashes[train_hash],
                        "distance": 0,
                        "type": "精確重複"
                    }
                    duplicates.append(match_info)
                    print(f"\n發現精確重複: {train_img_path}")
                    continue # 找到精確重複就不用再比對相似度

                # 2. 檢查相似重複
                for test_hash in test_hash_keys:
                    distance = train_hash - test_hash
                    if distance <= similarity_threshold:
                        match_info = {
                            "train_image": str(train_img_path),
                            "test_images": test_hashes[test_hash],
                            "distance": distance,
                            "type": "相似重複"
                        }
                        duplicates.append(match_info)
                        print(f"\n發現相似重複 (差異度 {distance}): {train_img_path}")
                        break # 找到一個相似的就夠了
        except Exception as e:
            print(f"無法處理檔案 {train_img_path}: {e}")
            
        # 顯示進度
        if (i + 1) % 100 == 0 or (i + 1) == len(train_image_files):
            print(f"  已掃描 {i + 1}/{len(train_image_files)} 張訓練圖片", end='\r')
            
    print(f"\n掃描完成！")
    return duplicates

def main():
    parser = argparse.ArgumentParser(description="驗證訓練集與測試集的分離性，避免資料洩漏。")
    parser.add_argument("--test-dir", type=str, default="archive/tw_food_101/test", help="測試集圖片資料夾路徑。")
    parser.add_argument("--train-dir", type=str, default="downloads/bing_images", help="要檢查的訓練集圖片資料夾路徑。")
    parser.add_argument("--threshold", type=int, default=5, help="相似度閾值 (漢明距離)，數值越小表示要求越相似。0 代表完全相同。")
    parser.add_argument("--delete", action="store_true", help="如果設定此旗標，將會自動刪除在訓練集中找到的重複圖片。")
    
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    train_dir = Path(args.train_dir)

    if not test_dir.exists() or not test_dir.is_dir():
        print(f"錯誤：測試目錄 '{test_dir}' 不存在或不是一個資料夾。")
        return
    if not train_dir.exists() or not train_dir.is_dir():
        print(f"錯誤：訓練目錄 '{train_dir}' 不存在或不是一個資料夾。")
        return

    # 步驟 1: 計算測試集的雜湊
    test_hashes = compute_hashes(test_dir)
    if not test_hashes:
        return

    # 步驟 2: 在訓練集中尋找重複
    duplicates = find_duplicates(test_hashes, train_dir, similarity_threshold=args.threshold)

    # 步驟 3: 報告與處理
    if not duplicates:
        print("\n🎉 恭喜！訓練集中未發現與測試集重複或高度相似的圖片。")
    else:
        print(f"\n⚠️ 發現 {len(duplicates)} 個重複/相似的圖片：")
        files_to_delete = []
        for item in duplicates:
            print(f"  - 訓練圖片: {item['train_image']}")
            print(f"    類型: {item['type']} (差異度: {item['distance']})")
            print(f"    對應測試圖片: {', '.join(item['test_images'])}")
            files_to_delete.append(item['train_image'])
        
        if args.delete:
            print("\n--delete 旗標已設定，開始刪除重複的訓練圖片...")
            deleted_count = 0
            for f_path in files_to_delete:
                try:
                    os.remove(f_path)
                    print(f"  已刪除: {f_path}")
                    deleted_count += 1
                except OSError as e:
                    print(f"  刪除失敗: {f_path} ({e})")
            print(f"\n共刪除了 {deleted_count} 個檔案。")
        else:
            print("\n提示：若要自動刪除這些重複檔案，請在執行指令時加上 --delete 旗標。")

if __name__ == "__main__":
    main()