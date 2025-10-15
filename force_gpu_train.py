"""
嚴格強制使用 GPU 訓練的啟動腳本
此腳本會調用 train_pytorch.py 並強制設置 CUDA 環境
"""

import os
import torch
import sys
import subprocess

print("=" * 60)
print("🚀 強制 GPU 訓練啟動器")
print("=" * 60)

# 1. 檢查 CUDA 可用性
print("\n1. 檢查 CUDA 是否可用...")
if not torch.cuda.is_available():
    print("❌ CUDA 不可用，請先安裝 GPU 驅動和 CUDA 支持的 PyTorch")
    print("   執行 fix_pytorch_gpu.bat 以安裝正確版本")
    sys.exit(1)
    
print("✅ CUDA 可用")

# 2. 啟用 cuDNN
print("\n2. 強制啟用 cuDNN...")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
print("✅ cuDNN 已啟用")

# 3. 設置 CUDA 環境變數
print("\n3. 設置 CUDA 環境變數...")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第一個 GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # 提高記憶體分配效率
print("✅ 環境變數已設置")

# 4. 測試 GPU 功能
print("\n4. 測試 GPU 基本功能...")
try:
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    z = torch.matmul(x, y)
    end.record()
    torch.cuda.synchronize()
    
    time_ms = start.elapsed_time(end)
    print(f"✅ GPU 矩陣乘法完成: {time_ms:.2f} ms")
    
    # 清理記憶體
    del x, y, z
    torch.cuda.empty_cache()
except Exception as e:
    print(f"❌ GPU 功能測試失敗: {e}")
    print("請檢查 CUDA 驅動和 PyTorch 安裝")
    sys.exit(1)

# 5. 啟動訓練腳本
print("\n5. 啟動訓練腳本...")
print("=" * 60)
print("訓練開始，請確認以下信息:")
print("- 設備是否顯示為 'cuda'")
print("- 是否有 GPU 記憶體使用量報告")
print("- 每個 epoch 的速度是否明顯快於 CPU")
print("=" * 60)
print()

try:
    # 使用 subprocess 啟動訓練腳本，保持輸出實時顯示
    subprocess.run([sys.executable, "train_pytorch.py"], check=True)
except Exception as e:
    print(f"❌ 訓練腳本執行失敗: {e}")
    sys.exit(1)