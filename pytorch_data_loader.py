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
            # 組合完整路徑：base_dir + category + filename
            img_name = os.path.join(self.img_dir, relative_path.replace('/', os.sep))
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

def get_dataloaders(train_csv, test_csv, train_img_dir, test_img_dir, batch_size=32, img_size=224, val_split=0.2):
    """
    建立訓練、驗證和測試的 DataLoader
    
    重要：測試集不參與訓練過程！
    - 訓練集：用於模型學習
    - 驗證集：從訓練集分割出來，用於調參和早停
    - 測試集：完全獨立，僅用於最終模型評估
    """
    # 訓練資料增強
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(0.5),
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
    
    # Windows 環境下建議設 num_workers=0 避免多進程問題
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader
