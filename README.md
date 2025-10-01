使用 python 3.10## GPU/NPU 支援

### ✅ AMD Ryzen AI 9HX NPU 已成功配置！

恭喜！您的 **AMD Ryzen AI 9 HX 370 w/ Radeon 890M** NPU 已完全就緒！

#### 🚀 現在可用的功能
- **AMD NPU 推理**: 使用 DirectML 執行提供者
- **ONNX Runtime 最佳化**: 專為 AMD NPU 調校
- **自動模型轉換**: PyTorch → ONNX → AMD NPU

## 🚀 快速開始

### 🎮 手動硬體選擇模式 (推薦)

#### 🔍 推理/評估 - 手動選擇硬體
```bash
# 使用手動硬體選擇工具進行推理
python manual_hardware_selection.py

# 或使用批次檔 (Windows)
.\run_manual_hardware_selection.bat
```

#### 🏋️ 訓練 - 手動選擇硬體  
```bash
# 使用手動硬體選擇工具進行訓練
python manual_training.py

# 或使用批次檔 (Windows)
.\run_manual_training.bat
```

### 🤖 自動模式

#### 💻 快速開始
```bash
# 使用 AMD NPU 進行模型評估 (自動選擇)
python evaluate_test_set.py

# 或使用優化啟動腳本
.\run_amd_npu_evaluation.bat

# 完整功能測試
python test_amd_npu_complete.py
```

#### 🎯 效能優勢
- **NPU 加速**: 專用神經處理單元，效能遠超 CPU
- **低功耗**: 相比 GPU 具有更好的能效比  
- **即時推理**: 適合實時台灣美食分類

## 🎯 硬體選擇功能

### 🎮 手動硬體選擇 (新功能!)
現在支援手動選擇推理和訓練硬體，讓您完全掌控運算資源：

#### 📋 可選擇的硬體：
- **🚀 AMD Ryzen AI NPU**: 最高效能神經處理器
- **✅ NVIDIA GPU (CUDA)**: 高效能圖形處理器  
- **🍎 Apple Silicon (MPS)**: Apple 晶片專用加速
- **💻 CPU**: 穩定可靠的處理器

#### 🔧 使用方式：
1. **手動模式**: 由您選擇要使用的硬體
2. **自動模式**: 系統自動選擇最佳硬體

### 檢查硬體支援
```bash
# 全面檢測系統中的計算裝置 (CPU/GPU/NPU)
python test_devices.py
```

### 檢查 CUDA/NPU 支援
```bash
# 檢查 PyTorch 是否識別到 GPU/NPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"CPU only\"}')"
```

### NPU 支援說明
本專案支援多種計算裝置：
- **NPU (神經處理單元)**: 最高效能，適合深度學習推理
- **GPU**: 高效能，適合訓練和推理  
- **CPU**: 穩定可靠，所有環境都支援

系統會自動檢測並選擇最佳的可用裝置。nv 建立環境方式
# py -3.10 -m venv venvCNN

開啟虛擬環境方式 CMD
# venvCNN\Scripts\activate.bat
照片爬蟲

# python bing_image_crawler.py --images-per-class 600 --start-id 51 --end-id 61                                               

開啟虛擬環境方式 powershell
#  .\venvCNN\Scripts\Activate.ps1

初次進入安裝插件
# pip install -r requirements.txt

## GPU 支援

### 检查 CUDA 支援
```bash
# 检查 PyTorch 是否識別到 GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else "CPU only"}')"
```

### 如果沒有 CUDA 支援，安裝 CUDA 版本的 PyTorch
```bash
# 直接安裝 CUDA 11.8 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或者安裝 CUDA 12.1 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 執行訓練

#### 🎮 手動硬體選擇模式
```bash
# 手動選擇訓練硬體
python manual_training.py
```

#### 🤖 自動模式
```bash
# 使用 PyTorch 訓練 (自動偵測 GPU，支援斷點續訓)
python train_pytorch.py
```

## 🎯 訓練流程說明

### 🔄 斷點續訓功能
執行訓練時，系統會自動檢測已保存的模型並讓您選擇：
- `0. 從頭開始訓練` - 建立全新模型
- `1. 繼續訓練: model_epoch10.pth` - 從已保存的模型繼續
- `2. 繼續訓練: model_epoch15.pth` - 選擇其他保存點

### 資料分離原則
- **訓練集** (`tw_food_101_train.csv`): 20,372 張圖片，用於模型學習
- **驗證集**: 從訓練集自動分割 20%，用於調參和早停 
- **測試集** (`tw_food_101_test_list.csv`): 5,093 張圖片，**完全不參與訓練**

### ⚠️ 重要提醒
測試集檔案 `tw_food_101_test_list.csv` 中的資料絕對不參加訓練過程！
這確保了模型評估的公正性和可靠性。

### 訓練後評估

#### 🎮 手動硬體選擇模式
```bash
# 手動選擇推理硬體進行測試集評估
python manual_hardware_selection.py
```

#### 🤖 自動模式
```bash
# 訓練完成後，在測試集上進行最終評估 (自動選擇硬體)
python evaluate_test_set.py
```