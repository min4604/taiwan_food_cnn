#!/usr/bin/env python3
"""
GPU 測試腳本 - 檢查 GPU 是否可用
"""

import torch
import sys

def test_gpu():
    """測試 GPU 可用性"""
    print("=" * 60)
    print("🔍 GPU 檢測測試")
    print("=" * 60)
    
    # 基本資訊
    print(f"🐍 Python 版本: {sys.version}")
    print(f"🔥 PyTorch 版本: {torch.__version__}")
    
    # CUDA 檢測
    print(f"\n💻 CUDA 資訊:")
    print(f"   CUDA 可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA 版本: {torch.version.cuda}")
        print(f"   cuDNN 版本: {torch.backends.cudnn.version()}")
        print(f"   GPU 數量: {torch.cuda.device_count()}")
        
        # 列出所有 GPU
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            print(f"   GPU {i}: {props.name}")
            print(f"           記憶體: {memory_gb:.1f} GB")
            print(f"           運算能力: {props.major}.{props.minor}")
        
        # 測試基本 GPU 運算
        print(f"\n🧪 GPU 運算測試:")
        try:
            device = torch.device('cuda')
            x = torch.randn(1000, 1000).to(device)
            y = torch.randn(1000, 1000).to(device)
            z = torch.mm(x, y)
            print(f"   ✅ 基本矩陣運算: 成功")
            print(f"   📊 結果形狀: {z.shape}")
            print(f"   💾 GPU 記憶體使用: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        except Exception as e:
            print(f"   ❌ GPU 運算失敗: {e}")
    else:
        print(f"   ❌ GPU 不可用")
        print(f"\n💡 可能的解決方案:")
        print(f"   1. 確認您有 NVIDIA GPU")
        print(f"   2. 安裝 NVIDIA 驅動程式")
        print(f"   3. 安裝 CUDA Toolkit")
        print(f"   4. 重新安裝 PyTorch (CUDA 版本)")
        print(f"      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    # 其他加速選項
    print(f"\n🚀 其他加速選項:")
    
    # DirectML (Windows)
    try:
        import torch_directml
        if torch_directml.is_available():
            print(f"   ✅ DirectML (Windows NPU/GPU): 可用")
            print(f"      設備: {torch_directml.device()}")
        else:
            print(f"   ❌ DirectML: 不可用")
    except ImportError:
        print(f"   ➡️  DirectML: 未安裝")
        print(f"      安裝: pip install torch-directml")
    
    # MPS (macOS)
    if hasattr(torch.backends, 'mps'):
        if torch.backends.mps.is_available():
            print(f"   ✅ Apple MPS: 可用")
        else:
            print(f"   ❌ Apple MPS: 不可用")
    
    print("=" * 60)

if __name__ == "__main__":
    test_gpu()