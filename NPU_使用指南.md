# NPU 加速使用指南

## 🚀 什麼是 NPU 加速？

NPU (Neural Processing Unit) 是專門為 AI 運算設計的處理器，可以大幅提升深度學習模型的訓練和推理速度。

## 📋 支援的 NPU 類型

### 1. Intel NPU (DirectML)
- **支援型號**: Intel Arc GPU, Intel Xe GPU, Intel 內建顯示晶片
- **優點**: 廣泛相容性，適合大多數 Intel 處理器
- **安裝**: 執行 `install_npu_support.bat`

### 2. Apple Neural Engine (MPS)
- **支援型號**: Apple Silicon (M1/M2/M3 系列)
- **優點**: 原生支援，效能極佳
- **安裝**: PyTorch 1.12+ 自動支援

### 3. 華為昇騰 NPU
- **支援型號**: 華為昇騰系列 NPU
- **優點**: 專業 AI 晶片，性能強大
- **安裝**: 需要專用驅動和 torch_npu

## 🛠️ 安裝步驟

### 方法一：一鍵安裝 (推薦)
```bash
# 安裝 Intel NPU 支援
install_npu_support.bat
```

### 方法二：手動安裝
```bash
# 啟動虛擬環境
venvCNN\Scripts\activate

# 安裝 DirectML (Intel NPU)
pip install torch-directml

# 驗證安裝
python test_npu_performance.py
```

## 🧪 測試 NPU 功能

### 1. 性能測試
```bash
python test_npu_performance.py
```
這個腳本會：
- 檢測所有可用的加速設備
- 比較不同設備的運算性能
- 提供最佳設備建議

### 2. 訓練測試
```bash
python train_pytorch.py
```
訓練時會看到設備選擇選單：
```
🔧 選擇計算設備:
1. NVIDIA GPU (CUDA) - 推薦
2. Intel NPU (DirectML) - 高效能
3. CPU
```

## 📊 性能比較

| 設備類型 | 訓練速度 | 記憶體使用 | 功耗 | 適用場景 |
|---------|---------|-----------|------|---------|
| NVIDIA GPU | 🟢 很快 | 🟡 高 | 🔴 高 | 大型模型、研究 |
| Intel NPU | 🟢 快 | 🟢 低 | 🟢 低 | 日常訓練、推理 |
| Apple NPU | 🟢 很快 | 🟢 低 | 🟢 低 | MacBook 用戶 |
| CPU | 🔴 慢 | 🟢 中等 | 🟢 低 | 基礎測試 |

## 🎯 最佳實踐

### 1. 模型選擇建議
```python
# NPU 友好的模型 (推薦順序)
1. EfficientNet-B3    # 最佳平衡
2. ConvNeXt-Tiny      # 現代架構
3. RegNet-Y           # 高效設計
4. ResNet50           # 經典穩定
5. Vision Transformer # GPU/NPU 專用
```

### 2. 批次大小建議
- **NVIDIA GPU**: 64-128
- **Intel NPU**: 32-64
- **Apple NPU**: 32-64
- **CPU**: 8-16

### 3. 編譯優化
程式會自動啟用 `torch.compile()` 優化：
```
⚡ 啟用模型編譯優化...
✅ 模型編譯優化已啟用，性能將進一步提升
```

## 🔧 故障排除

### 問題 1：NPU 未檢測到
**解決方案**：
1. 確認硬體支援：
   ```bash
   # Windows
   dxdiag
   # 查看顯示卡型號
   
   # 檢查 DirectML 支援
   python -c "import torch_directml; print('NPU 可用')"
   ```

2. 重新安裝驅動：
   - Intel: 下載最新 Intel Graphics Driver
   - AMD: 下載最新 AMD Software

### 問題 2：訓練速度沒有提升
**檢查項目**：
1. 確認正確選擇了 NPU 設備
2. 檢查批次大小是否適當
3. 確認模型編譯優化已啟用

### 問題 3：記憶體不足
**解決方案**：
1. 降低批次大小：
   ```python
   batch_size = 16  # 從 64 降到 16
   ```

2. 使用較小的模型：
   ```python
   model = get_model('efficientnet_b0')  # 而非 b3
   ```

## 📈 性能優化技巧

### 1. 自動混合精度 (AMP)
```python
# 在訓練腳本中啟用 (自動啟用)
scaler = torch.cuda.amp.GradScaler()
```

### 2. 資料載入優化
```python
# 多程序資料載入
DataLoader(dataset, num_workers=4, pin_memory=True)
```

### 3. 模型檢查點
```python
# 自動保存最佳模型
torch.save(model.state_dict(), 'models/best_model.pth')
```

## 📞 支援

如果遇到問題：
1. 執行 `python test_npu_performance.py` 檢測設備
2. 檢查錯誤訊息中的具體問題
3. 確認硬體驅動是否為最新版本

## 🎉 開始使用

```bash
# 1. 安裝 NPU 支援
install_npu_support.bat

# 2. 測試性能
python test_npu_performance.py

# 3. 開始訓練
python train_pytorch.py
# 選擇 NPU 設備並享受加速訓練！
```

享受 NPU 加速帶來的高效訓練體驗！ 🚀