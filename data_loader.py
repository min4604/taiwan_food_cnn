#!/usr/bin/env python3
"""
台灣美食 CNN 訓練資料載入器
"""

import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class TaiwanFoodDataLoader:
    """台灣美食資料載入器"""
    
    def __init__(self, data_dir="archive/tw_food_101", img_size=(224, 224), batch_size=32):
        """
        初始化資料載入器
        
        Args:
            data_dir: 資料目錄路徑
            img_size: 圖片大小 (height, width)
            batch_size: 批次大小
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        
        # 載入類別映射
        self.classes_df = pd.read_csv(
            os.path.join(data_dir, "tw_food_101_classes.csv"),
            header=None, 
            names=['class_id', 'class_name']
        )
        self.num_classes = len(self.classes_df)
        self.class_names = self.classes_df['class_name'].tolist()
        
        # 建立類別映射
        self.id_to_name = dict(zip(self.classes_df['class_id'], self.classes_df['class_name']))
        self.name_to_id = dict(zip(self.classes_df['class_name'], self.classes_df['class_id']))
        
        # 載入測試資料列表（確保不參與訓練）
        self.test_files = set()
        test_list_path = os.path.join(data_dir, "tw_food_101_test_list.csv")
        if os.path.exists(test_list_path):
            test_df = pd.read_csv(test_list_path, header=None, names=['index', 'image_path'])
            self.test_files = set(test_df['image_path'].tolist())
            print(f"載入測試檔案列表: {len(self.test_files)} 個檔案")
        
        print(f"載入 {self.num_classes} 個類別")
        print(f"測試檔案將完全排除在訓練之外")
        
    def load_image(self, image_path):
        """載入並預處理單張圖片"""
        try:
            # 使用 PIL 載入圖片
            image = Image.open(image_path).convert('RGB')
            
            # 調整大小
            image = image.resize(self.img_size)
            
            # 轉換為 numpy array
            image = np.array(image)
            
            # 正規化到 [0, 1]
            image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            print(f"載入圖片失敗 {image_path}: {e}")
            # 返回黑色圖片作為佔位符
            return np.zeros((*self.img_size, 3), dtype=np.float32)
    
    def load_dataset_from_csv(self, split='train', validation_split=0.2):
        """
        從 CSV 檔案載入資料集，確保測試資料不參與訓練
        
        Args:
            split: 'train' 或 'test'
            validation_split: 驗證集比例（僅適用於訓練集）
        """
        if split == 'train':
            csv_file = 'tw_food_101_train.csv'
            df = pd.read_csv(os.path.join(self.data_dir, csv_file), 
                           header=None, 
                           names=['index', 'class_id', 'image_path'])
            
            # 重要：過濾掉測試檔案，確保訓練集不包含測試資料
            original_count = len(df)
            df = df[~df['image_path'].isin(self.test_files)]
            filtered_count = len(df)
            
            if original_count != filtered_count:
                print(f"警告：從訓練資料中移除了 {original_count - filtered_count} 個測試檔案")
            
        elif split == 'test':
            csv_file = 'tw_food_101_test_list.csv'
            df = pd.read_csv(os.path.join(self.data_dir, csv_file), 
                           header=None, 
                           names=['index', 'image_path'])
            # 測試集沒有標籤
            df['class_id'] = -1  # 設為 -1，表示未知標籤
        else:
            raise ValueError(f"不支援的分割類型: {split}")
        
        images = []
        labels = []
        valid_indices = []
        
        print(f"從 {csv_file} 載入 {split} 資料...")
        print(f"總共 {len(df)} 筆資料")
        
        for idx, row in df.iterrows():
            if idx % 500 == 0:
                print(f"已處理 {idx}/{len(df)} 筆資料")
            
            # 構建完整的圖片路徑
            img_path = os.path.join(self.data_dir, row['image_path'])
            class_id = row['class_id']
            
            # 雙重檢查：確保測試檔案不會出現在訓練中
            if split == 'train' and row['image_path'] in self.test_files:
                print(f"跳過測試檔案: {row['image_path']}")
                continue
            
            # 檢查檔案是否存在
            if os.path.exists(img_path):
                img = self.load_image(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(class_id)
                    valid_indices.append(idx)
            else:
                print(f"檔案不存在: {img_path}")
        
        images = np.array(images)
        labels = np.array(labels)
        
        print(f"成功載入 {len(images)} 張 {split} 圖片")
        
        # 如果是訓練集且需要驗證集分割
        if split == 'train' and validation_split > 0:
            # 確保每個類別都有足夠的樣本進行分層抽樣
            from collections import Counter
            label_counts = Counter(labels)
            min_samples = min(label_counts.values())
            
            if min_samples < 2:
                print(f"警告：某些類別樣本數太少 (最少 {min_samples} 個)，無法進行分層抽樣")
                print("改用隨機抽樣...")
                X_train, X_val, y_train, y_val = train_test_split(
                    images, labels, 
                    test_size=validation_split, 
                    random_state=42
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    images, labels, 
                    test_size=validation_split, 
                    stratify=labels,
                    random_state=42
                )
            
            # 轉換為 one-hot 編碼
            y_train_onehot = to_categorical(y_train, num_classes=self.num_classes)
            y_val_onehot = to_categorical(y_val, num_classes=self.num_classes)
            
            print(f"訓練集分割: {len(X_train)} 訓練, {len(X_val)} 驗證")
            return (X_train, y_train_onehot), (X_val, y_val_onehot)
        
        elif split == 'train':
            # 不分割驗證集的訓練資料
            labels_onehot = to_categorical(labels, num_classes=self.num_classes)
            return images, labels_onehot
        
        else:
            # 測試集（沒有標籤）
            return images, labels  # labels 都是 -1
    
    def load_dataset_from_csv_old(self, csv_file):
        """
        舊版本：從 CSV 檔案載入資料集
        CSV 格式: image_path, class_name 或 image_path, class_id
        """
        df = pd.read_csv(os.path.join(self.data_dir, csv_file))
        
        images = []
        labels = []
        
        print(f"從 {csv_file} 載入資料...")
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"已處理 {idx}/{len(df)} 筆資料")
                
            # 獲取圖片路徑
            img_path = os.path.join(self.data_dir, row.iloc[0])  # 假設第一列是圖片路徑
            
            if os.path.exists(img_path):
                img = self.load_image(img_path)
                if img is not None:
                    images.append(img)
                    
                    # 獲取標籤
                    if isinstance(row.iloc[1], str):
                        # 如果是類別名稱
                        class_name = row.iloc[1]
                        class_id = self.name_to_id.get(class_name, -1)
                    else:
                        # 如果是類別ID
                        class_id = int(row.iloc[1])
                    
                    labels.append(class_id)
        
        images = np.array(images)
        labels = np.array(labels)
        
        # 轉換為 one-hot 編碼
        labels_onehot = to_categorical(labels, num_classes=self.num_classes)
        
        return images, labels_onehot
    
    def create_data_generators(self, train_data, val_data, augment=True):
        """建立資料生成器用於訓練"""
        try:
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
        except ImportError:
            print("警告：無法匯入 ImageDataGenerator，將使用簡單的資料生成器")
            return self.create_simple_generators(train_data, val_data)
        
        # 訓練資料增強
        if augment:
            train_datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                shear_range=0.15,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator()
        
        # 驗證資料不做增強
        val_datagen = ImageDataGenerator()
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def create_simple_generators(self, train_data, val_data):
        """建立簡單的資料生成器（不依賴 Keras）"""
        class SimpleDataGenerator:
            def __init__(self, X, y, batch_size, shuffle=True):
                self.X = X
                self.y = y
                self.batch_size = batch_size
                self.shuffle = shuffle
                self.indices = np.arange(len(X))
                if shuffle:
                    np.random.shuffle(self.indices)
                self.current_idx = 0
            
            def __iter__(self):
                return self
            
            def __next__(self):
                if self.current_idx >= len(self.X):
                    self.current_idx = 0
                    if self.shuffle:
                        np.random.shuffle(self.indices)
                
                batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
                self.current_idx += self.batch_size
                
                return self.X[batch_indices], self.y[batch_indices]
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        train_generator = SimpleDataGenerator(X_train, y_train, self.batch_size, shuffle=True)
        val_generator = SimpleDataGenerator(X_val, y_val, self.batch_size, shuffle=False)
        
        return train_generator, val_generator
    
    def show_sample_images(self, images, labels, num_samples=9):
        """顯示樣本圖片"""
        plt.figure(figsize=(12, 12))
        
        for i in range(min(num_samples, len(images))):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i])
            
            # 獲取類別名稱
            class_id = np.argmax(labels[i])
            class_name = self.id_to_name.get(class_id, f"Unknown_{class_id}")
            
            plt.title(f"ID: {class_id}\n{class_name}", fontsize=10)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_class_distribution(self, labels):
        """分析類別分布"""
        # 將 one-hot 轉換為類別ID
        if labels.ndim == 2:
            class_ids = np.argmax(labels, axis=1)
        else:
            class_ids = labels
            
        unique, counts = np.unique(class_ids, return_counts=True)
        
        distribution = {}
        for class_id, count in zip(unique, counts):
            class_name = self.id_to_name.get(class_id, f"Unknown_{class_id}")
            distribution[f"{class_id}_{class_name}"] = count
        
        return distribution

# 使用範例
if __name__ == "__main__":
    # 建立資料載入器
    loader = TaiwanFoodDataLoader()
    
    # 載入訓練資料（使用 CSV）
    try:
        (X_train, y_train), (X_val, y_val) = loader.load_dataset_from_csv('train')
        
        print(f"訓練集形狀: {X_train.shape}")
        print(f"驗證集形狀: {X_val.shape}")
        print(f"標籤形狀: {y_train.shape}")
        
        # 顯示類別分布
        train_dist = loader.get_class_distribution(y_train)
        print(f"訓練集類別分布 (前10項): {dict(list(train_dist.items())[:10])}")
        
        # 載入測試資料
        X_test, y_test = loader.load_dataset_from_csv('test')
        print(f"測試集形狀: {X_test.shape}")
        
        # 顯示樣本圖片
        # loader.show_sample_images(X_train[:9], y_train[:9])
        
    except Exception as e:
        print(f"載入資料時發生錯誤: {e}")
        print("請確認資料目錄結構正確")