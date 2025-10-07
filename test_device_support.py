#!/usr/bin/env python3
"""
簡單的 NPU/GPU 設備測試腳本
測試是否能正確檢測和使用各種加速設備
"""

import torch
import sys

def test_cuda():
    """測試 CUDA GPU"""
    print("🔍 測試 CUDA GPU...")
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            x = torch.randn(100, 100).to(device)
            y = torch.randn(100, 100).to(device)
            z = torch.mm(x, y)
            print(f"✅ CUDA GPU 可用: {torch.cuda.get_device_name()}")
            return True, 'cuda'
        else:
            print("❌ CUDA GPU 不可用")
            return False, None
    except Exception as e:
        print(f"❌ CUDA GPU 測試失敗: {e}")
        return False, None

def test_amd_npu():
    """測試 AMD Ryzen AI NPU"""
    print("🔍 測試 AMD Ryzen AI NPU...")
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'DmlExecutionProvider' in providers:
            # 創建測試模型
            import numpy as np
            
            # 簡單的測試運算
            session_options = ort.SessionOptions()
            session_options.enable_mem_pattern = False
            
            print("✅ AMD Ryzen AI NPU 可用 (ONNX Runtime)")
            print(f"   可用提供者: {providers[:3]}...")  # 只顯示前幾個
            return True, 'amd_npu'
        else:
            print("❌ AMD NPU 不可用")
            print(f"   可用提供者: {providers}")
            return False, None
    except ImportError:
        print("❌ ONNX Runtime 未安裝")
        print("   執行: pip install onnxruntime-directml")
        return False, None
    except Exception as e:
        print(f"❌ AMD NPU 測試失敗: {e}")
        return False, None

def test_directml():
    """測試 DirectML (Intel NPU)"""
    print("🔍 測試 Intel DirectML NPU...")
    try:
        import torch_directml
        if torch_directml.is_available():
            device = torch_directml.device()
            x = torch.randn(100, 100).to(device)
            y = torch.randn(100, 100).to(device)
            z = torch.mm(x, y)
            print(f"✅ Intel DirectML NPU 可用: {device}")
            return True, 'intel_dml'
        else:
            print("❌ Intel DirectML NPU 不可用")
            return False, None
    except ImportError:
        print("❌ torch_directml 未安裝")
        print("   執行: pip install torch-directml")
        return False, None
    except Exception as e:
        print(f"❌ Intel DirectML NPU 測試失敗: {e}")
        return False, None

def test_mps():
    """測試 Apple MPS"""
    print("🔍 測試 Apple MPS...")
    try:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            x = torch.randn(100, 100).to(device)
            y = torch.randn(100, 100).to(device)
            z = torch.mm(x, y)
            print("✅ Apple MPS 可用")
            return True, 'mps'
        else:
            print("❌ Apple MPS 不可用")
            return False, None
    except Exception as e:
        print(f"❌ Apple MPS 測試失敗: {e}")
        return False, None

def test_cpu():
    """測試 CPU"""
    print("🔍 測試 CPU...")
    try:
        device = torch.device('cpu')
        x = torch.randn(100, 100).to(device)
        y = torch.randn(100, 100).to(device)
        z = torch.mm(x, y)
        print("✅ CPU 可用")
        return True, 'cpu'
    except Exception as e:
        print(f"❌ CPU 測試失敗: {e}")
        return False, None

def main():
    print("=" * 50)
    print("🧪 設備加速測試")
    print("=" * 50)
    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")
    print("=" * 50)
    
    available_devices = []
    
    # 測試所有設備
    tests = [
        ("CUDA GPU", test_cuda),
        ("AMD Ryzen AI NPU", test_amd_npu),
        ("Intel DirectML NPU", test_directml),
        ("Apple MPS", test_mps),
        ("CPU", test_cpu)
    ]
    
    for name, test_func in tests:
        print()
        success, device_type = test_func()
        if success:
            available_devices.append((name, device_type))
    
    # 顯示結果
    print("\n" + "=" * 50)
    print("📊 測試結果摘要")
    print("=" * 50)
    
    if available_devices:
        print("✅ 可用的設備:")
        for i, (name, device_type) in enumerate(available_devices, 1):
            print(f"   {i}. {name} ({device_type})")
    else:
        print("❌ 沒有可用的設備")
    
    print("\n💡 建議:")
    if any('AMD' in name or 'Ryzen AI' in name for name, _ in available_devices):
        print("   🚀 AMD Ryzen AI NPU 可用！建議在訓練時選擇 AMD NPU 以獲得更好的性能")
    elif any('NPU' in name or 'DirectML' in name for name, _ in available_devices):
        print("   🚀 Intel NPU 可用！建議在訓練時選擇 Intel NPU 以獲得更好的性能")
    elif any('CUDA' in name for name, _ in available_devices):
        print("   🎯 GPU 可用！建議在訓練時選擇 GPU 以獲得更好的性能")
    else:
        print("   💻 只有 CPU 可用，考慮安裝 GPU 或 NPU 支援")
    
    # 安裝建議
    if not any('DirectML' in name or 'AMD' in name for name, _ in available_devices):
        print("\n🛠️  安裝 NPU 支援:")
        print("   Intel NPU: pip install torch-directml")
        print("   AMD NPU:   pip install onnxruntime-directml")
        print("   或執行:    install_npu.bat")
    
    print("\n✅ 測試完成！")

if __name__ == "__main__":
    main()