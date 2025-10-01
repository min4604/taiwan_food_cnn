#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
視覺化重複圖片審查工具

功能：
1. 啟動後，掃描測試集和訓練集，找出相似的圖片組。
2. 在介面上並排顯示相似的圖片，供使用者比對。
3. 提供 "刪除" 和 "保留" 按鈕，讓使用者決定如何處理訓練集中的重複圖片。
4. 點擊按鈕後，自動處理並顯示下一組相似圖片。
"""

import os
import argparse
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from collections import defaultdict
from pathlib import Path
from threading import Thread

try:
    from PIL import Image, ImageTk
    import imagehash
except ImportError:
    messagebox.showerror("缺少套件", "缺少必要的套件，請執行: pip install Pillow imagehash")
    exit(1)

class DeduplicatorApp:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.root.title("視覺化重複圖片審查工具")
        self.root.geometry("1000x700")

        # 資料
        self.test_hashes = {}
        self.duplicate_generator = None
        self.current_duplicate = None

        # --- UI 元件 ---
        # 主框架
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 圖片顯示框架
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True)
        image_frame.columnconfigure(0, weight=1)
        image_frame.columnconfigure(1, weight=1)
        image_frame.rowconfigure(1, weight=1)

        # 測試集圖片
        ttk.Label(image_frame, text="測試集圖片", font=("Helvetica", 14, "bold")).grid(row=0, column=0, pady=5)
        self.test_img_label = ttk.Label(image_frame, background="gray")
        self.test_img_label.grid(row=1, column=0, sticky="nsew", padx=5)
        self.test_path_label = ttk.Label(image_frame, text="路徑: N/A", wraplength=450)
        self.test_path_label.grid(row=2, column=0, pady=5)

        # 訓練集圖片
        ttk.Label(image_frame, text="訓練集圖片", font=("Helvetica", 14, "bold")).grid(row=0, column=1, pady=5)
        self.train_img_label = ttk.Label(image_frame, background="gray")
        self.train_img_label.grid(row=1, column=1, sticky="nsew", padx=5)
        self.train_path_label = ttk.Label(image_frame, text="路徑: N/A", wraplength=450)
        self.train_path_label.grid(row=2, column=1, pady=5)

        # 資訊與控制框架
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)

        self.info_label = ttk.Label(control_frame, text="點擊 '開始掃描' 以尋找重複圖片", font=("Helvetica", 12))
        self.info_label.pack(pady=5)

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(pady=10)

        self.start_button = ttk.Button(button_frame, text="開始掃描", command=self.start_scan)
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.delete_button = ttk.Button(button_frame, text="刪除訓練圖片", state=tk.DISABLED, command=self.delete_image)
        self.delete_button.pack(side=tk.LEFT, padx=10)

        self.keep_button = ttk.Button(button_frame, text="保留", state=tk.DISABLED, command=self.keep_image)
        self.keep_button.pack(side=tk.LEFT, padx=10)
        
        # 相似度控制框架
        threshold_frame = ttk.Frame(control_frame)
        threshold_frame.pack(pady=10)
        
        ttk.Label(threshold_frame, text="相似度閾值:").pack(side=tk.LEFT, padx=5)
        self.threshold_var = tk.IntVar(value=args.threshold)
        self.threshold_scale = ttk.Scale(threshold_frame, from_=0, to=15, 
                                       variable=self.threshold_var, orient=tk.HORIZONTAL, length=200)
        self.threshold_scale.pack(side=tk.LEFT, padx=5)
        self.threshold_label = ttk.Label(threshold_frame, text=f"{args.threshold}")
        self.threshold_label.pack(side=tk.LEFT, padx=5)
        
        # 綁定滑桿變化事件
        self.threshold_var.trace('w', self.on_threshold_change)
        
        ttk.Label(threshold_frame, text="(0=完全相同, 15=很不相似)").pack(side=tk.LEFT, padx=10)
        
        # 進度條
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        self.status_label = ttk.Label(main_frame, text="準備就緒")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def on_threshold_change(self, *args):
        """當相似度滑桿變化時更新顯示"""
        current_threshold = self.threshold_var.get()
        self.threshold_label.config(text=str(current_threshold))
        # 如果正在掃描，需要重新開始
        if hasattr(self, 'duplicate_generator') and self.duplicate_generator is not None:
            self.info_label.config(text=f"相似度已調整為 {current_threshold}，請重新掃描以套用新設定")
    
    def start_scan(self):
        """啟動掃描執行緒"""
        self.start_button.config(state=tk.DISABLED)
        self.info_label.config(text="正在掃描中，請稍候...")
        # 使用執行緒避免 UI 凍結
        scan_thread = Thread(target=self._scan_worker, daemon=True)
        scan_thread.start()

    def _scan_worker(self):
        """在背景執行緒中執行耗時的掃描任務"""
        try:
            # 步驟 1: 計算測試集的雜湊
            self.status_label.config(text=f"正在計算測試集 '{self.args.test_dir}' 的雜湊值...")
            self.test_hashes = self._compute_hashes(Path(self.args.test_dir))
            if not self.test_hashes:
                messagebox.showwarning("掃描警告", f"在測試目錄 '{self.args.test_dir}' 中找不到任何圖片。")
                self._reset_ui()
                return

            # 步驟 2: 建立重複項的產生器
            self.status_label.config(text=f"正在掃描訓練集 '{self.args.train_dir}'...")
            self.duplicate_generator = self._yield_duplicates(Path(self.args.train_dir))
            
            # 步驟 3: 處理第一個重複項
            self.root.after(0, self.next_duplicate)

        except Exception as e:
            messagebox.showerror("掃描錯誤", f"掃描過程中發生錯誤: {e}")
            self._reset_ui()

    def _compute_hashes(self, directory: Path) -> dict:
        hashes = defaultdict(list)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        image_files = [p for p in directory.rglob('*') if p.suffix.lower() in image_extensions]
        
        total = len(image_files)
        if total == 0:
            return hashes
            
        for i, img_path in enumerate(image_files):
            try:
                with Image.open(img_path) as img:
                    # 縮小圖片以加快雜湊計算
                    img.thumbnail((256, 256), Image.Resampling.LANCZOS)
                    h = imagehash.average_hash(img, hash_size=self.args.hash_size)
                    hashes[h].append(str(img_path))
            except Exception:
                continue
            
            # 減少UI更新頻率，只在每50張或最後更新
            if (i + 1) % 50 == 0 or (i + 1) == total:
                progress = ((i + 1) / total) * 50  # 前50%給雜湊計算
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
                self.root.after(0, lambda: self.status_label.config(text=f"計算測試集雜湊: {i+1}/{total}"))
        return hashes

    def _yield_duplicates(self, train_dir: Path):
        """一個產生器，逐一產出重複的圖片組"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        train_image_files = [p for p in train_dir.rglob('*') if p.suffix.lower() in image_extensions]
        test_hash_keys = list(self.test_hashes.keys())
        
        total = len(train_image_files)
        current_threshold = self.threshold_var.get()  # 取一次就好
        
        for i, train_img_path in enumerate(train_image_files):
            # 減少UI更新頻率但保持響應性
            if i % 50 == 0 or i == total - 1:
                progress = 50 + ((i + 1) / total) * 50  # 後50%給掃描
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
                self.root.after(0, lambda: self.status_label.config(text=f"掃描訓練集: {i+1}/{total}"))
            
            try:
                with Image.open(train_img_path) as img:
                    # 縮小圖片以加快處理
                    img.thumbnail((256, 256), Image.Resampling.LANCZOS)
                    train_hash = imagehash.average_hash(img, hash_size=self.args.hash_size)
                    
                    # 檢查精確重複
                    if train_hash in self.test_hashes:
                        yield {
                            "train_image": str(train_img_path),
                            "test_images": self.test_hashes[train_hash],
                            "distance": 0,
                            "type": "精確重複"
                        }
                        continue

                    # 檢查相似重複
                    for test_hash in test_hash_keys:
                        distance = train_hash - test_hash
                        if distance <= current_threshold:
                            yield {
                                "train_image": str(train_img_path),
                                "test_images": self.test_hashes[test_hash],
                                "distance": distance,
                                "type": "相似重複"
                            }
                            break
            except Exception:
                continue

    def next_duplicate(self):
        """顯示下一組重複的圖片"""
        try:
            self.current_duplicate = next(self.duplicate_generator)
            self.display_duplicate()
            self.delete_button.config(state=tk.NORMAL)
            self.keep_button.config(state=tk.NORMAL)
        except StopIteration:
            messagebox.showinfo("掃描完成", "已掃描所有圖片，沒有更多重複項了。")
            self._reset_ui()

    def display_duplicate(self):
        """在 UI 上顯示圖片和資訊"""
        dup = self.current_duplicate
        test_img_path = dup["test_images"][0]
        train_img_path = dup["train_image"]

        # 更新資訊
        self.info_label.config(text=f"類型: {dup['type']} (差異度: {dup['distance']})")
        self.test_path_label.config(text=f"路徑: {test_img_path}")
        self.train_path_label.config(text=f"路徑: {train_img_path}")

        # 顯示圖片
        self._show_image(self.test_img_label, test_img_path)
        self._show_image(self.train_img_label, train_img_path)

    def _show_image(self, label, path):
        """載入並顯示圖片，自動縮放到適合標籤的大小"""
        try:
            # 使用固定大小以提高效能
            max_size = (400, 400)
            
            with Image.open(path) as img:
                # 轉換為RGB模式以避免顯示問題
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 保持比例縮放
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(img)
                label.config(image=photo, text="")
                label.image = photo # 保持對圖片的引用
        except Exception as e:
            label.config(text=f"無法載入圖片:\n{str(e)[:50]}...", image='', compound=tk.CENTER)

    def delete_image(self):
        """刪除訓練圖片並處理下一張"""
        if not self.current_duplicate: return
        
        train_img_path = self.current_duplicate["train_image"]
        try:
            os.remove(train_img_path)
            self.status_label.config(text=f"已刪除: {train_img_path}")
        except OSError as e:
            messagebox.showerror("刪除失敗", f"無法刪除檔案 {train_img_path}: {e}")
        
        self.next_duplicate()

    def keep_image(self):
        """保留圖片並處理下一張"""
        if not self.current_duplicate: return
        train_img_path = self.current_duplicate["train_image"]
        self.status_label.config(text=f"已保留: {train_img_path}")
        self.next_duplicate()

    def _reset_ui(self):
        """將 UI 重設回初始狀態"""
        self.start_button.config(state=tk.NORMAL)
        self.delete_button.config(state=tk.DISABLED)
        self.keep_button.config(state=tk.DISABLED)
        self.info_label.config(text="點擊 '開始掃描' 以尋找重複圖片")
        self.status_label.config(text="準備就緒")
        self.progress_var.set(0)  # 清空進度條
        # 重設產生器
        self.duplicate_generator = None

def main():
    parser = argparse.ArgumentParser(description="視覺化重複圖片審查工具。")
    parser.add_argument("--test-dir", type=str, default="archive/tw_food_101/test", help="測試集圖片資料夾路徑。")
    parser.add_argument("--train-dir", type=str, default="downloads/bing_images", help="要檢查的訓練集圖片資料夾路徑。")
    parser.add_argument("--threshold", type=int, default=5, help="相似度閾值 (漢明距離)，數值越小表示要求越相似。")
    parser.add_argument("--hash-size", type=int, default=8, help="感知雜湊的精細度。")
    
    args = parser.parse_args()

    # 檢查目錄是否存在
    if not Path(args.test_dir).exists():
        messagebox.showerror("目錄錯誤", f"測試目錄 '{args.test_dir}' 不存在。")
        return
    if not Path(args.train_dir).exists():
        messagebox.showerror("目錄錯誤", f"訓練目錄 '{args.train_dir}' 不存在。")
        return

    root = tk.Tk()
    app = DeduplicatorApp(root, args)
    root.mainloop()

if __name__ == "__main__":
    main()