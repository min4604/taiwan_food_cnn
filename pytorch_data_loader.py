import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

class TaiwanFoodDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None, is_test=False):
        self.data = pd.read_csv(csv_path, header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        
        # 建立目錄名稱映射以處理命名不一致問題
        self.dir_mapping = {
            'deep_fried_chicken_cutlets': 'deep-fried_chicken_cutlets',
            'fried_spanish_mackerel_thick_soup': 'fried-spanish_mackerel_thick_soup',
            'hakka_stir_fried': 'hakka_stir-fried',
            'kung_pao_chicken': 'kung-pao_chicken',
            'rice_with_soy_stewed_pork': 'rice_with_soy-stewed_pork',
            'steam_fried_bun': 'steam-fried_bun',
            'stir_fried_calamari_broth': 'stir-fried_calamari_broth',
            'stir_fried_duck_meat_broth': 'stir-fried_duck_meat_broth',
            'stir_fried_loofah_with_clam': 'stir-fried_loofah_with_clam',
            'stir_fried_pork_intestine_with_ginger': 'stir-fried_pork_intestine_with_ginger',
            'three_cup_chicken': 'three-cup_chicken',
            'tube_shaped_migao': 'tube-shaped_migao',
        }
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        if self.is_test:
            # 測試集格式: [index, image_path]
            img_path = self.data.iloc[idx, 1]
            # 測試集的圖片直接在 test 目錄下
            img_filename = os.path.basename(img_path)
            img_name = os.path.join(self.img_dir, img_filename)
            # 測試集沒有標籤，返回 -1
            label = -1
        else:
            # 訓練集格式: [index, class_id, image_path]
            img_path = self.data.iloc[idx, 2]  # train/category/filename.jpg
            # 移除開頭的 "train/" 前綴，保留類別目錄結構
            if img_path.startswith('train/'):
                relative_path = img_path[6:]  # 移除 "train/" 前綴
            else:
                relative_path = img_path
            
            # 分離目錄名稱和檔案名稱
            path_parts = relative_path.split('/')
            if len(path_parts) >= 2:
                category_name = path_parts[0]
                filename = path_parts[1]
                
                # 檢查是否需要映射目錄名稱
                if category_name in self.dir_mapping:
                    category_name = self.dir_mapping[category_name]
                
                # 重新組合路徑
                relative_path = os.path.join(category_name, filename)
            
            # 正規化路徑分隔符號，確保在 Windows 下正確
            relative_path = relative_path.replace('/', os.sep).replace('\\', os.sep)
            # 組合完整路徑：base_dir + category + filename
            img_name = os.path.join(self.img_dir, relative_path)
            label = int(self.data.iloc[idx, 1])
            
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            print(f"找不到檔案: {img_name}")
            print(f"原始路徑: {img_path if not self.is_test else self.data.iloc[idx, 1]}")
            raise
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_dataloaders(train_csv, test_csv, train_img_dir, test_img_dir, batch_size=32, img_size=224, val_split=0.2, num_workers=0, pin_memory=False):
    """
    建立訓練、驗證和測試的 DataLoader
    
    重要：測試集不參與訓練過程！
    - 訓練集：用於模型學習
    - 驗證集：從訓練集分割出來，用於調參和早停
    - 測試集：完全獨立，僅用於最終模型評估
    
    參數：
    - num_workers：資料載入的工作執行緒數，建議 GPU 訓練時設為 4
    - pin_memory：是否使用固定記憶體，建議 GPU 訓練時設為 True
    """
    # 增強的訓練資料增強
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 驗證/測試資料不增強
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 建立兩個相同的資料集，一個用於訓練（有增強），一個用於驗證（無增強）
    train_dataset_aug = TaiwanFoodDataset(train_csv, train_img_dir, train_transform, is_test=False)
    train_dataset_no_aug = TaiwanFoodDataset(train_csv, train_img_dir, val_transform, is_test=False)
    
    # 分割訓練集為訓練/驗證集
    dataset_size = len(train_dataset_aug)
    train_size = int((1 - val_split) * dataset_size)
    val_size = dataset_size - train_size
    
    # 使用相同的隨機種子確保分割一致
    torch.manual_seed(42)
    train_indices, val_indices = torch.utils.data.random_split(range(dataset_size), [train_size, val_size])
    
    # 建立子集
    train_subset = torch.utils.data.Subset(train_dataset_aug, train_indices.indices)
    val_subset = torch.utils.data.Subset(train_dataset_no_aug, val_indices.indices)
    
    # 測試集（沒有標籤）
    test_dataset = TaiwanFoodDataset(test_csv, test_img_dir, val_transform, is_test=True)
    
    print(f"📥 數據加載器配置: num_workers={num_workers}, pin_memory={pin_memory}, batch_size={batch_size}")
    
    # 根據參數配置 DataLoader
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # 丟棄不完整批次，避免批次歸一化問題
    )
    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader
