#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分類錯誤圖片審查工具

功能：
1. 讀取包含預測結果的CSV檔案
2. 比較預測類別(Id)與實際類別(Category)
3. 當分類錯誤時，顯示圖片供使用者審查
4. 提供刪除、保留、跳過等操作選項
"""

import os
import argparse
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
from pathlib import Path
from threading import Thread

try:
    from PIL import Image, ImageTk
except ImportError:
    messagebox.showerror("缺少套件", "缺少必要的套件，請執行: pip install Pillow pandas")
    exit(1)

class ClassificationReviewApp:
    def __init__(self, root, ground_truth_csv, predictions_csv):
        self.root = root
        self.root.title("台灣美食 CNN 分類審查工具")
        self.root.geometry("1000x800")
        self.ground_truth_csv = ground_truth_csv
        self.predictions_csv = predictions_csv
        self.root.title("分類錯誤圖片審查工具")
        self.root.geometry("1200x800")

        # 載入正確答案和預測結果
        self.ground_truth_df, self.predictions_df, self.misclassified = self.load_and_compare_data()
        self.current_index = 0
        self.deleted_files = []
        self.class_names = self.load_class_names()

        self.setup_ui()
        self.load_first_image()

    def load_and_compare_data(self):
        """載入正確答案和預測結果，並找出分類錯誤的項目"""
        try:
            # 檢查檔案是否存在
            if not os.path.exists(self.ground_truth_csv):
                raise FileNotFoundError(f"正確答案檔案不存在: {self.ground_truth_csv}")
            if not os.path.exists(self.predictions_csv):
                raise FileNotFoundError(f"預測結果檔案不存在: {self.predictions_csv}")
            
            # 載入正確答案 (train_list.csv: index,category,path)
            print(f"正在載入正確答案: {self.ground_truth_csv}")
            ground_truth = pd.read_csv(self.ground_truth_csv, header=None, 
                                     names=['index', 'true_category', 'relative_path'],
                                     skipinitialspace=True, na_filter=False)
            
            # 載入預測結果 (test_predictions_optimized_amd_npu.csv: Id,Category,Confidence,Path)
            print(f"正在載入預測結果: {self.predictions_csv}")
            predictions = pd.read_csv(self.predictions_csv, na_filter=False)
            
            print(f"載入正確答案: {len(ground_truth)} 筆")
            print(f"載入預測結果: {len(predictions)} 筆")
            
            # 清理空值和異常值
            ground_truth = ground_truth.dropna(subset=['relative_path'])
            predictions = predictions.dropna(subset=['Path'])
            
            print(f"清理後正確答案: {len(ground_truth)} 筆")
            print(f"清理後預測結果: {len(predictions)} 筆")
            
            # 驗證欄位
            required_truth_cols = ['index', 'true_category', 'relative_path']
            required_pred_cols = ['Id', 'Category', 'Path']
            
            if not all(col in ground_truth.columns for col in required_truth_cols):
                raise ValueError(f"正確答案檔案缺少必要欄位，需要: {required_truth_cols}, 實際: {ground_truth.columns.tolist()}")
            
            if not all(col in predictions.columns for col in required_pred_cols):
                raise ValueError(f"預測結果檔案缺少必要欄位，需要: {required_pred_cols}, 實際: {predictions.columns.tolist()}")
            
            # 建立路徑到正確類別的映射
            # 從相對路徑提取檔案名進行比對
            def get_filename_key(path):
                """提取用於比對的檔案名鍵值，去除副檔名"""
                if not path or not isinstance(path, str):
                    return ""
                try:
                    return Path(str(path)).stem.lower()
                except Exception:
                    return ""
            
            # 建立正確答案的檔案名到類別的映射
            truth_mapping = {}
            for _, row in ground_truth.iterrows():
                filename_key = get_filename_key(row['relative_path'])
                if filename_key:  # 只有有效的檔名才加入映射
                    truth_mapping[filename_key] = row['true_category']
            
            print(f"建立了 {len(truth_mapping)} 個檔名映射")
            print(f"範例映射: {dict(list(truth_mapping.items())[:5])}")
            
            # 為預測結果添加正確類別
            def get_true_category(path):
                try:
                    if not path:
                        return -1
                    key = get_filename_key(path)
                    return truth_mapping.get(key, -1) if key else -1
                except Exception:
                    return -1
            
            predictions['true_category'] = predictions['Path'].apply(get_true_category)
            
            # 檢查有多少找不到對應
            not_found = predictions[predictions['true_category'] == -1]
            print(f"找不到對應的預測結果: {len(not_found)} 筆")
            if len(not_found) > 0:
                print(f"前5個找不到的檔案: {not_found['Path'].head().tolist()}")
            
            # 過濾掉找不到正確答案的項目
            valid_predictions = predictions[predictions['true_category'] != -1].copy()
            
            # 所有分類錯誤的項目（不再自動刪除、也不以信心度過濾）
            misclassified = valid_predictions[
                valid_predictions['Category'] != valid_predictions['true_category']
            ].copy()
            # 標記高信心誤判
            misclassified['is_high_conf'] = misclassified['Confidence'] >= 0.95
            # 將高信心誤判排序到最前面，方便優先處理
            misclassified = misclassified.sort_values(by=['is_high_conf', 'Confidence'], ascending=[False, False])

            print(f"\n=== 比對結果 ====")
            print(f"總預測數量: {len(predictions)}")
            print(f"有效比對: {len(valid_predictions)} 筆")
            print(f"分類錯誤: {len(misclassified)} 筆")

            if len(valid_predictions) > 0:
                accuracy = (len(valid_predictions) - len(misclassified)) / len(valid_predictions) * 100
                print(f"準確率: {accuracy:.2f}%")

            if len(misclassified) == 0:
                print("\n恭喜！沒有發現分類錯誤的圖片！")
            else:
                print(f"\n找到 {len(misclassified)} 個分類錯誤的項目可供審查")

            return ground_truth, valid_predictions, misclassified
            
        except FileNotFoundError as e:
            error_msg = f"檔案不存在: {e}"
            print(error_msg)
            messagebox.showerror("檔案錯誤", error_msg)
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        except pd.errors.EmptyDataError as e:
            error_msg = f"CSV檔案為空或格式錯誤: {e}"
            print(error_msg)
            messagebox.showerror("資料錯誤", error_msg)
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        except Exception as e:
            import traceback
            error_msg = f"無法載入或比對資料: {e}\n\n詳細錯誤:\n{traceback.format_exc()}"
            print(error_msg)
            messagebox.showerror("載入錯誤", error_msg)
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def load_class_names(self):
        """載入類別名稱對應表"""
        # 台灣美食類別中文翻譯
        chinese_names = {
            'bawan': '肉圓',
            'beef_noodles': '牛肉麵',
            'beef_soup': '牛肉湯',
            'bitter_melon_with_salted_eggs': '鹹蛋苦瓜',
            'braised_napa_cabbage': '燉白菜',
            'braised_pork_over_rice': '滷肉飯',
            'brown_sugar_cake': '黑糖糕',
            'bubble_tea': '珍珠奶茶',
            'caozaiguo': '草仔粿',
            'chicken_mushroom_soup': '香菇雞湯',
            'chinese_pickled_cucumber': '醃黃瓜',
            'coffin_toast': '棺材板',
            'cold_noodles': '涼麵',
            'crab_migao': '螃蟹米糕',
            'deep-fried_chicken_cutlets': '雞排',
            'deep_fried_pork_rib_and_radish_soup': '炸排骨蘿蔔湯',
            'dried_shredded_squid': '魷魚絲',
            'egg_pancake_roll': '蛋餅',
            'eight_treasure_shaved_ice': '八寶冰',
            'fish_head_casserole': '魚頭煲',
            'fried-spanish_mackerel_thick_soup': '旗魚羹',
            'fried_eel_noodles': '鱔魚意麵',
            'fried_instant_noodles': '炒泡麵',
            'fried_rice_noodles': '炒米粉',
            'ginger_duck_stew': '薑母鴨',
            'grilled_corn': '烤玉米',
            'grilled_taiwanese_sausage': '烤香腸',
            'hakka_stir-fried': '客家小炒',
            'hot_sour_soup': '酸辣湯',
            'hung_rui_chen_sandwich': '洪瑞珍三明治',
            'intestine_and_oyster_vermicelli': '大腸蚵仔麵線',
            'iron_egg': '鐵蛋',
            'jelly_of_gravey_and_chicken_feet_skin': '雞腳凍',
            'jerky': '肉乾',
            'kung-pao_chicken': '宮保雞丁',
            'luwei': '滷味',
            'mango_shaved_ice': '芒果冰',
            'meat_dumpling_in_chili_oil': '紅油抄手',
            'milkfish_belly_congee': '虱目魚肚粥',
            'mochi': '麻糬',
            'mung_bean_smoothie_milk': '綠豆沙牛奶',
            'mutton_fried_noodles': '羊肉炒麵',
            'mutton_hot_pot': '羊肉爐',
            'nabeyaki_egg_noodles': '鍋燒意麵',
            'night_market_steak': '夜市牛排',
            'nougat': '牛軋糖',
            'oyster_fritter': '蚵嗲',
            'oyster_omelet': '蚵仔煎',
            'papaya_milk': '木瓜牛奶',
            'peanut_brittle': '花生糖',
            'pepper_pork_bun': '胡椒餅',
            'pig_s_blood_soup': '豬血湯',
            'pineapple_cake': '鳳梨酥',
            'pork_intestines_fire_pot': '豬腸火鍋',
            'potsticker': '鍋貼',
            'preserved_egg_tofu': '皮蛋豆腐',
            'rice_dumpling': '粽子',
            'rice_noodles_with_squid': '花枝米粉',
            'rice_with_soy-stewed_pork': '滷肉燥飯',
            'roasted_sweet_potato': '烤地瓜',
            'sailfish_stick': '旗魚串',
            'salty_fried_chicken_nuggets': '鹽酥雞',
            'sanxia_golden_croissants': '三峽金牛角',
            'saute_spring_onion_with_beef': '蔥爆牛肉',
            'scallion_pancake': '蔥油餅',
            'scrambled_eggs_with_shrimp': '蝦仁炒蛋',
            'scrambled_eggs_with_tomatoes': '番茄炒蛋',
            'seafood_congee': '海鮮粥',
            'sesame_oil_chicken_soup': '麻油雞湯',
            'shrimp_rice': '蝦仁飯',
            'sishen_soup': '四神湯',
            'sliced_pork_bun': '割包',
            'spicy_duck_blood': '麻辣鴨血',
            'steam-fried_bun': '生煎包',
            'steamed_cod_fish_with_crispy_bean': '脆皮豆腐蒸鱈魚',
            'steamed_taro_cake': '芋頭糕',
            'stewed_pig_s_knuckles': '滷豬腳',
            'stinky_tofu': '臭豆腐',
            'stir-fried_calamari_broth': '花枝羹',
            'stir-fried_duck_meat_broth': '鴨肉羹',
            'stir-fried_loofah_with_clam': '絲瓜蛤蠣',
            'stir-fried_pork_intestine_with_ginger': '薑絲大腸',
            'stir_fried_clams_with_basil': '九層塔炒蛤蜊',
            'sugar_coated_sweet_potato': '拔絲地瓜',
            'sun_cake': '太陽餅',
            'sweet_and_sour_pork_ribs': '糖醋排骨',
            'sweet_potato_ball': '地瓜球',
            'taiwanese_burrito': '潤餅',
            'taiwanese_pork_ball_soup': '貢丸湯',
            'taiwanese_sausage_in_rice_bun': '大腸包小腸',
            'tanghulu': '糖葫蘆',
            'tangyuan': '湯圓',
            'taro_ball': '芋圓',
            'three-cup_chicken': '三杯雞',
            'tube-shaped_migao': '筒仔米糕',
            'turkey_rice': '火雞肉飯',
            'turnip_cake': '蘿蔔糕',
            'twist_roll': '雙胞胎',
            'wheel_pie': '車輪餅',
            'xiaolongbao': '小籠包',
            'yolk_pastry': '蛋黃酥',
        }
        
        try:
            # 嘗試載入類別名稱檔案
            class_file = Path("archive/tw_food_101/tw_food_101_classes.csv")
            if class_file.exists():
                with open(class_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                class_names = {}
                for i, line in enumerate(lines):
                    line = line.strip()
                    if ',' in line:
                        parts = line.split(',')
                        english_name = parts[1]
                        chinese_name = chinese_names.get(english_name, english_name)
                        class_names[i] = f"{chinese_name} ({english_name})"
                    else:
                        class_names[i] = line
                return class_names
        except Exception as e:
            print(f"無法載入類別名稱: {e}")
        
        # 如果載入失敗，使用預設編號
        return {i: f"類別_{i}" for i in range(101)}

    def setup_ui(self):
        """設置使用者介面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 標題資訊
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(info_frame, text="分類錯誤圖片審查", font=("Helvetica", 16, "bold")).pack()
        
        self.progress_label = ttk.Label(info_frame, text="", font=("Helvetica", 12))
        self.progress_label.pack(pady=5)

        # 圖片顯示框架
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True)

        # 圖片標籤
        self.image_label = ttk.Label(image_frame, background="gray")
        self.image_label.pack(pady=10)

        # 分類資訊框架
        info_detail_frame = ttk.Frame(main_frame)
        info_detail_frame.pack(fill=tk.X, pady=10)

        # 左側：預測結果
        left_frame = ttk.LabelFrame(info_detail_frame, text="預測結果", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.predicted_label = ttk.Label(left_frame, text="", font=("Helvetica", 12))
        self.predicted_label.pack()
        
        # 右側：實際類別
        right_frame = ttk.LabelFrame(info_detail_frame, text="實際類別", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.actual_label = ttk.Label(right_frame, text="", font=("Helvetica", 12))
        self.actual_label.pack()

        # 路徑與信心度資訊
        detail_frame = ttk.LabelFrame(main_frame, text="檔案資訊", padding="10")
        detail_frame.pack(fill=tk.X, pady=5)

        self.path_label = ttk.Label(detail_frame, text="", wraplength=800, font=("Consolas", 9), 
                                   foreground="blue", relief="sunken", padding="5")
        self.path_label.pack(fill=tk.X, pady=(0, 5))

        confidence_frame = ttk.Frame(detail_frame)
        confidence_frame.pack(fill=tk.X)
        
        ttk.Label(confidence_frame, text="預測信心度:", font=("Helvetica", 10)).pack(side=tk.LEFT)
        self.confidence_label = ttk.Label(confidence_frame, text="", font=("Helvetica", 10, "bold"))
        self.confidence_label.pack(side=tk.LEFT, padx=(5, 0))

        # 控制按鈕
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)

        self.delete_button = ttk.Button(button_frame, text="🗑️ 刪除圖片", command=self.delete_image, width=15)
        self.delete_button.pack(side=tk.LEFT, padx=10)

        self.keep_button = ttk.Button(button_frame, text="✅ 保留圖片", command=self.keep_image, width=15)
        self.keep_button.pack(side=tk.LEFT, padx=10)

        self.skip_button = ttk.Button(button_frame, text="⏭️ 跳過", command=self.next_image, width=15)
        self.skip_button.pack(side=tk.LEFT, padx=10)

        # 導航按鈕
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(pady=10)

        self.prev_button = ttk.Button(nav_frame, text="⬅️ 上一張", command=self.prev_image, width=12)
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.next_button = ttk.Button(nav_frame, text="➡️ 下一張", command=self.next_image, width=12)
        self.next_button.pack(side=tk.LEFT, padx=5)

        # 狀態列
        self.status_label = ttk.Label(main_frame, text="準備就緒", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

        # 鍵盤快捷鍵
        self.root.bind('<Delete>', lambda e: self.delete_image())
        self.root.bind('<Return>', lambda e: self.keep_image())
        self.root.bind('<space>', lambda e: self.next_image())
        self.root.bind('<Left>', lambda e: self.prev_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.focus_set()  # 確保視窗能接收鍵盤事件

    def load_first_image(self):
        """載入第一張圖片"""
        if len(self.misclassified) == 0:
            messagebox.showinfo("完成", "沒有發現分類錯誤的圖片！")
            self.root.quit()
            return
        
        self.display_current_image()

    def display_current_image(self):
        """顯示當前圖片"""
        if self.current_index >= len(self.misclassified):
            messagebox.showinfo("審查完成", f"所有分類錯誤的圖片已審查完畢！\n共刪除 {len(self.deleted_files)} 張圖片。")
            self.root.quit()
            return

        row = self.misclassified.iloc[self.current_index]

        #（已移除）高信心誤判自動彈窗刪除邏輯
        
        # 更新進度資訊
        current_progress = f"{self.current_index + 1}/{len(self.misclassified)}"
        deleted_info = f"已刪除: {len(self.deleted_files)}"
        
        self.progress_label.config(
            text=f"進度: {current_progress} ({deleted_info})"
        )
        
        # 更新窗口標題
        file_name = os.path.basename(row['Path'])
        self.root.title(f"分類審查工具 - {file_name} ({current_progress})")

        # 更新分類資訊
        predicted_id = row['Category']  # 模型預測的類別
        actual_id = row['true_category']  # 正確的類別
        predicted_name = self.class_names.get(predicted_id, f"未知類別_{predicted_id}")
        actual_name = self.class_names.get(actual_id, f"未知類別_{actual_id}")

        self.predicted_label.config(
            text=f"模型預測\nID: {predicted_id}\n{predicted_name}",
            foreground="red",
            font=("Helvetica", 11, "bold")
        )
        self.actual_label.config(
            text=f"正確答案\nID: {actual_id}\n{actual_name}",
            foreground="green",
            font=("Helvetica", 11, "bold")
        )

        # 更新路徑與信心度
        file_path = row['Path']
        confidence = row['Confidence']
        
        # 顯示簡化的檔案路徑
        short_path = file_path.replace('C:\\Users\\Chen\\Desktop\\project\\taiwan_food_cnn\\', '')
        self.path_label.config(text=f"📁 {file_path}")
        
        # 信心度顯示和顏色
        confidence_color = "green" if confidence > 0.8 else "orange" if confidence > 0.5 else "red"
        self.confidence_label.config(
            text=f"{confidence:.4f} ({confidence*100:.2f}%)", 
            foreground=confidence_color
        )

        # 載入並顯示圖片
        self.load_image(row['Path'])

        # 更新按鈕狀態
        self.prev_button.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if self.current_index < len(self.misclassified) - 1 else tk.DISABLED)

    def load_image(self, image_path):
        """載入並顯示圖片"""
        try:
            # 檢查檔案是否存在
            if not os.path.exists(image_path):
                self.image_label.config(text=f"檔案不存在:\n{image_path}", image='')
                self.status_label.config(text="檔案不存在")
                return

            # 載入圖片
            img = Image.open(image_path)
            
            # 調整圖片大小以適合顯示
            max_size = (600, 400)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # 轉換為PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # 顯示圖片
            self.image_label.config(image=photo, text='')
            self.image_label.image = photo  # 保持引用避免被垃圾回收
            
            self.status_label.config(text=f"已載入圖片: {os.path.basename(image_path)}")
            
        except Exception as e:
            self.image_label.config(text=f"無法載入圖片:\n{e}", image='')
            self.status_label.config(text=f"載入失敗: {e}")

    def delete_image(self):
        """刪除當前圖片"""
        if self.current_index >= len(self.misclassified):
            return

        row = self.misclassified.iloc[self.current_index]
        image_path = row['Path']

        # 確認刪除
        result = messagebox.askyesno(
            "確認刪除", 
            f"確定要刪除這張圖片嗎？\n\n{image_path}\n\n此操作無法復原！"
        )
        
        if result:
            try:
                os.remove(image_path)
                self.deleted_files.append(image_path)
                self.status_label.config(text=f"已刪除: {os.path.basename(image_path)}")
                self.next_image()
            except Exception as e:
                messagebox.showerror("刪除失敗", f"無法刪除檔案:\n{e}")

    def keep_image(self):
        """保留當前圖片"""
        if self.current_index >= len(self.misclassified):
            return
            
        row = self.misclassified.iloc[self.current_index]
        self.status_label.config(text=f"已保留: {os.path.basename(row['Path'])}")
        self.next_image()

    def next_image(self):
        """下一張圖片"""
        if self.current_index < len(self.misclassified) - 1:
            self.current_index += 1
            self.display_current_image()
        else:
            messagebox.showinfo("審查完成", f"所有分類錯誤的圖片已審查完畢！\n共刪除 {len(self.deleted_files)} 張圖片。")
            self.root.quit()

    def prev_image(self):
        """上一張圖片"""
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current_image()

def main():
    parser = argparse.ArgumentParser(description="分類錯誤圖片審查工具")
    parser.add_argument("--ground-truth", type=str, default="train_list.csv", 
                       help="正確答案CSV檔案路徑")
    parser.add_argument("--predictions", type=str, default="test_predictions_optimized_amd_npu.csv", 
                       help="模型預測結果CSV檔案路徑")
    
    args = parser.parse_args()

    # 建立主視窗
    root = tk.Tk()
    app = ClassificationReviewApp(root, args.ground_truth, args.predictions)
    root.mainloop()

if __name__ == "__main__":
    main()