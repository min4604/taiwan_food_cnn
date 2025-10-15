# 🚀 EfficientNet-B7 使用指南

## 📊 模型規格對比

| 項目 | EfficientNet-B3 | EfficientNet-B7 | 提升 |
|------|----------------|----------------|------|
| **參數量** | 12M | 66M | 5.5x |
| **預設輸入尺寸** | 300x300 | 600x600 | 4x |
| **顯存需求** | ~3GB | ~8-12GB | 3-4x |
| **建議批次大小** | 32-64 | 8-16 | -50% |
| **訓練時間** | 1x | 3-4x | 3-4x |
| **預期精度提升** | 基準 | +2-5% | +2-5% |

## ⚠️ 硬體需求

### 最低配置
- **GPU**: RTX 4060 8GB 或以上
- **系統記憶體**: 16GB+
- **批次大小**: 8
- **預期速度**: 慢但可行

### 推薦配置
- **GPU**: RTX 4070 12GB 或以上
- **系統記憶體**: 32GB
- **批次大小**: 16-32
- **預期速度**: 合理

### 理想配置
- **GPU**: RTX 4080/4090 16GB+
- **系統記憶體**: 32GB+
- **批次大小**: 32-64
- **預期速度**: 快速

## 🎯 使用場景分析

### ✅ 適合使用 B7 的場景：
- 追求最高精度的項目
- 有充足的計算資源和時間
- 資料集大於 10,000 張圖片
- 不急於快速完成訓練
- 商業級應用，精度要求高
- 有多卡訓練環境

### ⚠️ 建議使用 B3 的場景：
- 快速原型開發
- 資源有限（8GB 以下顯存）
- 資料集小於 5,000 張
- 需要快速得到結果
- 教學或學習目的
- 平衡精度與效率

## 🔧 B7 訓練優化設定

### 1. 記憶體優化策略

```python
# 1. 減小批次大小
batch_size = 8  # RTX 4060 8GB
batch_size = 16  # RTX 4070 12GB

# 2. 梯度累積（模擬大批次）
accumulate_grad_batches = 4  # 等效批次 = 8 * 4 = 32

# 3. 混合精度訓練
torch.cuda.amp.autocast()
scaler = torch.cuda.amp.GradScaler()

# 4. 清理記憶體
torch.cuda.empty_cache()
```

### 2. 學習率調整

```python
# B7 建議更小的學習率
learning_rate = 0.0001  # 比 B3 的 0.001 小

# 更保守的學習率調度
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
```

### 3. 資料增強策略

```python
# B7 可以處理更大解析度
img_size = 512  # 從預設的 224 提升

# 強化的資料增強
transforms.Compose([
    transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

## 📈 預期性能對比

### Taiwan Food 101 資料集

| 模型 | 資料集大小 | 預期準確率 | 訓練時間 (RTX 4060) |
|------|-----------|------------|-------------------|
| EfficientNet-B3 | 全集 | 85-88% | 3-4 小時 |
| EfficientNet-B7 | 全集 | 88-92% | 12-16 小時 |

### 不同資料集大小的效果

| 資料集大小 | B3 準確率 | B7 準確率 | B7 提升 |
|-----------|----------|----------|---------|
| 1,000 張 | 75% | 76% | +1% |
| 5,000 張 | 85% | 88% | +3% |
| 10,000+ 張 | 88% | 92% | +4% |

## ⚡ 訓練時間預估

### 不同硬體配置下的時間預估（50 epochs）

| GPU 型號 | 批次大小 | 每 epoch 時間 | 總訓練時間 |
|---------|---------|--------------|------------|
| RTX 4060 8GB | 8 | 20 分鐘 | ~17 小時 |
| RTX 4070 12GB | 16 | 12 分鐘 | ~10 小時 |
| RTX 4080 16GB | 32 | 8 分鐘 | ~7 小時 |

## 💡 實用技巧

### 1. 監控 GPU 使用情況
```bash
# 實時監控 GPU
nvidia-smi -l 1

# 查看記憶體使用
watch -n 1 nvidia-smi
```

### 2. 訓練中途檢查點
```python
# 更頻繁地保存模型
if epoch % 5 == 0:  # 每 5 個 epoch 保存一次
    torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pth')
```

### 3. 早停策略
```python
# 實施早停避免過擬合
patience = 10  # B7 可能需要更多耐心
best_val_acc = 0
no_improve_count = 0

if val_acc > best_val_acc:
    best_val_acc = val_acc
    no_improve_count = 0
else:
    no_improve_count += 1
    if no_improve_count >= patience:
        print("早停觸發")
        break
```

## 🚨 常見問題與解決方案

### Q1: CUDA out of memory 錯誤
**解決方案:**
```python
# 1. 減小批次大小
batch_size = 4  # 最小可用批次

# 2. 降低圖片解析度
img_size = 384  # 從 512 降到 384

# 3. 清理記憶體
torch.cuda.empty_cache()
```

### Q2: 訓練太慢
**解決方案:**
```python
# 1. 啟用混合精度
torch.backends.cudnn.benchmark = True

# 2. 增加 DataLoader workers
num_workers = 4

# 3. 使用 pin_memory
pin_memory = True
```

### Q3: 精度提升不明顯
**解決方案:**
```python
# 1. 增加訓練輪數
epochs = 100  # B7 可能需要更多訓練

# 2. 更細緻的學習率調度
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)

# 3. 更強的資料增強
# 使用 AutoAugment 或 RandAugment
```

## 📋 檢查清單

在開始 B7 訓練前，請確認：

- [ ] GPU 顯存 ≥ 8GB
- [ ] 系統記憶體 ≥ 16GB
- [ ] 磁碟空間足夠（模型檔案更大）
- [ ] 已設置批次大小 ≤ 16
- [ ] 已啟用混合精度（如果支援）
- [ ] 預期訓練時間充足
- [ ] 已設置模型檢查點保存

## 🎉 開始訓練

```bash
# 啟動 B7 訓練
python train_pytorch.py

# 選擇選項 3: EfficientNet-B7
# 確認使用 B7 模型
# 等待訓練完成...
```

**記住**: EfficientNet-B7 是精度與效率的權衡。雖然訓練時間較長，但在大型資料集上通常能獲得顯著更好的結果！🚀