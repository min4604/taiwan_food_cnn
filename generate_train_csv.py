#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自動生成訓練集 CSV 檔案（使用官方類別對應 ID）

掃描指定訓練集資料夾（如 downloads/bing_images），
依照子資料夾（類別名稱）對應 `archive/tw_food_101/tw_food_101_classes.csv` 取得正確的類別 ID，
輸出格式為「索引,類別ID,路徑」三欄，與原始訓練集一致；
路徑預設以 path_root（預設 'train'）為根，例如: train/<類別名稱>/<檔名>。
"""

import os
import re
import csv
from pathlib import Path
import argparse

DEFAULT_CLASSES_CSV = str(Path('archive') / 'tw_food_101' / 'tw_food_101_classes.csv')

def normalize_name(name: str) -> str:
    """標準化類別名稱以便比對：小寫、空白->_、- -> _"""
    n = name.strip().lower()
    n = n.replace(' ', '_')
    n = n.replace('-', '_')
    return n

def load_class_mapping(classes_csv: str) -> dict:
    """從官方 classes CSV 載入 name->id 對應，使用標準化名稱作為 key"""
    mapping = {}
    csv_path = Path(classes_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到類別對應檔案: {classes_csv}")
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            try:
                cid = int(row[0])
                cname = row[1].strip()
                mapping[normalize_name(cname)] = cid
            except Exception:
                continue
    return mapping

def generate_train_csv(train_dir: str, output_csv: str, classes_csv: str, path_root: str = 'train'):
    train_path = Path(train_dir)
    if not train_path.exists():
        print(f"❌ 訓練集目錄不存在: {train_dir}")
        return
    
    # 載入官方類別對應
    try:
        name_to_id = load_class_mapping(classes_csv)
    except Exception as e:
        print(f"❌ 載入類別對應失敗: {e}")
        return
    
    # 支援格式: downloads/bing_images/000_bawan
    class_folders = [p for p in train_path.iterdir() if p.is_dir()]
    class_folders.sort() # 依資料夾名稱排序
    
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    rows = []
    for class_folder in class_folders:
        folder_name = class_folder.name
        # 先嘗試從資料夾名稱取得標準化的類別名稱
        # 例: "000_bawan" -> "bawan"；或 "bawan" -> "bawan"
        if '_' in folder_name and folder_name.split('_', 1)[0].isdigit():
            candidate_name = folder_name.split('_', 1)[1]
        else:
            candidate_name = folder_name

        norm_name = normalize_name(candidate_name)
        class_id = name_to_id.get(norm_name)

        # 若仍找不到，嘗試使用資料夾前綴數字（若有）
        if class_id is None:
            m = re.match(r"^(\d+)_", folder_name)
            if m:
                try:
                    class_id = int(m.group(1))
                except ValueError:
                    class_id = None

        if class_id is None:
            print(f"⚠️ 無法對應類別資料夾 '{folder_name}' 至 ID，已略過。")
            continue
        
        class_name_dir = norm_name  # 輸出路徑中的類別資料夾採用標準化名稱
        for img_path in class_folder.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in image_exts:
                # 第三欄使用以 path_root 為根的相對路徑: path_root/類別資料夾/檔名
                rel_path = str(Path(path_root) / class_name_dir / img_path.name).replace('\\', '/')
                rows.append([int(class_id), rel_path])
    
    if not rows:
        print("⚠️ 沒有找到任何圖片，請確認資料夾結構與路徑！")
        return
    
    # 產生索引並寫入三欄: idx, class_id, path
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for idx, (cid, rel_path) in enumerate(rows):
            writer.writerow([idx, cid, rel_path])
    print(f"✅ 已產生 {output_csv}，共 {len(rows)} 筆圖片資料（三欄: 索引,類別ID,路徑）。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自動生成訓練集 CSV 檔案（使用官方類別對應 ID）")
    #parser.add_argument('--train-dir', type=str, default='downloads/bing_images', help='訓練集根目錄')
    parser.add_argument('--train-dir', type=str, default='archive/tw_food_101/train', help='訓練集根目錄')
    parser.add_argument('--output-csv', type=str, default='train_list1.csv', help='輸出 CSV 檔案名稱')
    parser.add_argument('--classes-csv', type=str, default=DEFAULT_CLASSES_CSV, help='官方類別列表 CSV 路徑')
    parser.add_argument('--path-root', type=str, default='train', help="輸出路徑的根目錄名稱（預設 'train'）")
    args = parser.parse_args()
    generate_train_csv(args.train_dir, args.output_csv, args.classes_csv, args.path_root)
