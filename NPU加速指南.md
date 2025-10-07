# NPU 加速完整指南 🚀

## 第一步：檢測您的硬體

### 檢查是否支援 NPU 加速

執行設備檢測腳本：
```bash
python test_device_support.py
```

### 常見支援的硬體
- **Intel Arc GPU** (A380, A750, A770 等)
- **Intel Xe 顯示晶片** (第11代 Tiger Lake 以後)
- **Intel 內建顯示晶片** (UHD Graphics 600 系列以後)
- **AMD Radeon GPU** (透過 DirectML)
- **Apple Silicon** (M1/M2/M3 - 自動支援 MPS)

## 第二步：安裝 NPU 支援

### 方法一：一鍵安裝（推薦）
```bash
install_npu.bat
```

### 方法二：手動安裝
```bash
# 啟動虛擬環境
venvCNN\Scripts\activate

# 安裝 DirectML
pip install torch-directml

# 驗證安裝
python test_device_support.py
```

## 第三步：開始 NPU 加速訓練

### 執行訓練
```bash
python train_pytorch.py
```

### 選擇設備時的建議
```
🔧 選擇計算設備:
1. NVIDIA GPU (CUDA) - 推薦        ← 如果有 NVIDIA GPU
2. Intel NPU (DirectML) - 高效能   ← 選擇這個啟用 NPU 🎯
3. CPU
```

### 模型架構建議（NPU 友好度排序）
```
🏗️ 選擇模型架構:
1. ResNet50 (基礎模型)              ← NPU 相容性最佳 ⭐⭐⭐
2. EfficientNet-B3 (推薦，效能佳)    ← NPU 效能很好 ⭐⭐⭐
3. ConvNeXt-Tiny (現代架構)         ← NPU 支援良好 ⭐⭐
4. RegNet-Y (高效網路)              ← NPU 支援一般 ⭐⭐
5. Vision Transformer (注意力機制)   ← 需要大記憶體 ⭐
```

## 性能調優

### NPU 最佳化設定

程式會自動進行以下優化：
- ✅ 批次大小自動調整 (NPU: 64, CPU: 16)
- ✅ 模型編譯優化 (`torch.compile()`)
- ✅ 記憶體使用優化
- ✅ 資料傳輸優化

### 手動調整批次大小

如果記憶體不足，修改 `train_pytorch.py` 中的批次大小：
```python
# 找到這行並修改
batch_size = 32  # 從 64 降到 32
```

## 故障排除

### 問題 1：找不到 NPU
```
❌ torch_directml 未安裝 (執行 install_npu.bat 安裝)
```
**解決方案**：執行 `install_npu.bat`

### 問題 2：NPU 基本測試失敗
```
警告: NPU 基本測試失敗 - ...
```
**解決方案**：
1. 更新顯示卡驅動程式
2. 確認硬體支援 DirectML
3. 重新啟動電腦

### 問題 3：訓練時記憶體不足
```
RuntimeError: Out of memory
```
**解決方案**：
1. 降低批次大小到 32 或 16
2. 選擇較小的模型（ResNet50）
3. 關閉其他占用記憶體的程式

### 問題 4：NPU 性能不如預期
**檢查項目**：
1. 確認選擇了正確的 NPU 設備
2. 檢查模型編譯優化是否啟用：
   ```
   ⚡ 啟用模型編譯優化...
   ✅ 模型編譯優化已啟用，性能將進一步提升
   ```
3. 確認使用了適合的模型架構

## 性能比較

### 實際測試結果（參考）

| 設備類型 | 訓練速度 | 每個 Epoch 時間 | 記憶體使用 |
|---------|---------|---------------|-----------|
| RTX 3060 | 🟢 很快 | ~3 分鐘 | 8GB |
| Intel Arc A750 | 🟢 快 | ~5 分鐘 | 4GB |
| Intel UHD 770 | 🟡 中等 | ~8 分鐘 | 2GB |
| CPU (Intel i7) | 🔴 慢 | ~25 分鐘 | 4GB |

### NPU 最佳使用場景
- ✅ **日常開發和實驗**：NPU 提供良好的性能/功耗比
- ✅ **筆記型電腦**：低功耗，不會讓風扇狂轉
- ✅ **小到中型模型**：ResNet50, EfficientNet 等
- ❌ **大型 Transformer 模型**：建議使用高階 GPU

## 進階設定

### 手動指定 NPU 設備
```python
# 在訓練腳本中手動指定
import torch_directml
device = torch_directml.device()
```

### 檢查 NPU 記憶體使用
```python
# 檢查 DirectML 記憶體
import torch_directml
print(f"NPU 記憶體使用: {torch_directml.memory_allocated()} bytes")
```

### 混合精度訓練（實驗性）
```python
# 在支援的 NPU 上啟用半精度
model = model.half()  # 使用 FP16
```

## 下一步

1. **測試不同模型架構**：比較 ResNet50 vs EfficientNet-B3 在您的 NPU 上的性能
2. **優化超參數**：調整學習率、批次大小等
3. **嘗試資料增強**：在 NPU 上測試不同的資料增強策略

---

🎉 **恭喜！** 您現在可以使用 NPU 加速來訓練台灣美食分類模型了！

如果遇到問題，請執行 `python test_device_support.py` 檢查設備狀態。