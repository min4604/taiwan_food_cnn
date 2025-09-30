#!/usr/bin/env python3
"""
NPU/GPU/CPU 裝置檢測測試腳本
"""

import torch
import sys

def comprehensive_device_detection():
    """
    全面檢測系統中可用的計算裝置
    """
    print("🔍 系統計算裝置檢測報告")
    print("=" * 60)
    
    # PyTorch 版本
    print(f"📦 PyTorch 版本: {torch.__version__}")
    
    # CPU 資訊
    print("💻 CPU:")
    print(f"   可用: ✅")
    print(f"   執行緒數: {torch.get_num_threads()}")
    
    # CUDA GPU 檢測
    print("\n🎮 CUDA GPU:")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"   可用: ✅ ({gpu_count} 個裝置)")
        print(f"   CUDA 版本: {torch.version.cuda}")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"   GPU {i}: {props.name} ({memory_gb:.1f} GB)")
    else:
        print("   可用: ❌")
    
    # NPU 檢測 (多種方式)
    print("\n🚀 NPU (神經處理單元):")
    npu_found = False
    
    # AMD Ryzen AI NPU 檢測
    print("   AMD Ryzen AI NPU:")
    try:
        # 檢測 DirectML
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'DmlExecutionProvider' in providers:
            print("   ✅ ONNX Runtime DirectML 可用")
            npu_found = True
            
            # 嘗試檢測具體硬體
            import platform
            import subprocess
            if platform.system() == 'Windows':
                try:
                    result = subprocess.run(['wmic', 'cpu', 'get', 'name'], 
                                          capture_output=True, text=True, timeout=5)
                    if 'AMD Ryzen' in result.stdout and 'AI' in result.stdout:
                        print("   🎯 檢測到 AMD Ryzen AI 處理器")
                        npu_found = True
                    else:
                        print("   ⚠️  未檢測到 AMD Ryzen AI 處理器")
                except:
                    print("   ⚠️  無法檢測處理器型號")
        else:
            print("   ❌ DirectML 不可用")
            print("   💡 請執行 install_amd_npu.bat 安裝支援")
    except ImportError:
        print("   ❌ ONNX Runtime 未安裝")
        print("   💡 請執行: pip install onnxruntime-directml")
    except Exception as e:
        print(f"   ❌ 檢測失敗: {e}")
    
    # torch-directml 檢測
    try:
        import torch_directml
        if torch_directml.is_available():
            print("   ✅ torch-directml 可用")
            npu_found = True
    except ImportError:
        print("   ℹ️  torch-directml 未安裝 (可選)")
    except Exception as e:
        print(f"   ⚠️  torch-directml 檢測失敗: {e}")
    
    # 方式 1: torch.npu (華為等)
    try:
        if hasattr(torch, 'npu'):
            if torch.npu.is_available():
                npu_count = torch.npu.device_count()
                print(f"   可用 (torch.npu): ✅ ({npu_count} 個裝置)")
                for i in range(npu_count):
                    try:
                        name = torch.npu.get_device_name(i)
                        print(f"   NPU {i}: {name}")
                    except:
                        print(f"   NPU {i}: 未知型號")
                npu_found = True
            else:
                print("   torch.npu 存在但不可用")
        else:
            print("   torch.npu: 不存在")
    except Exception as e:
        print(f"   torch.npu 檢測錯誤: {e}")
    
    # 方式 2: torch.backends.npu
    try:
        if hasattr(torch.backends, 'npu'):
            if torch.backends.npu.is_available():
                print("   可用 (backends.npu): ✅")
                npu_found = True
            else:
                print("   backends.npu 存在但不可用")
        else:
            print("   torch.backends.npu: 不存在")
    except Exception as e:
        print(f"   backends.npu 檢測錯誤: {e}")
    
    # 方式 3: MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps'):
        print("\n🍎 MPS (Apple Silicon):")
        try:
            if torch.backends.mps.is_available():
                print("   可用: ✅")
                npu_found = True
            else:
                print("   可用: ❌")
        except:
            print("   檢測失敗")
    
    if not npu_found:
        print("   整體 NPU 支援: ❌")
        print("   💡 可能需要:")
        print("      - 安裝 NPU 專用 PyTorch 版本")
        print("      - 安裝 NPU 驅動程式")
        print("      - 確認硬體支援")
    
    # 建議的使用順序
    print("\n🎯 建議使用順序:")
    devices = []
    if npu_found:
        devices.append("🥇 NPU (最高效能)")
    if torch.cuda.is_available():
        devices.append("🥈 GPU (高效能)")
    devices.append("🥉 CPU (穩定)")
    
    for i, device in enumerate(devices, 1):
        print(f"   {i}. {device}")
    
    print("=" * 60)

def test_device_creation():
    """
    測試不同裝置的 tensor 建立
    """
    print("\n🧪 裝置功能測試")
    print("-" * 40)
    
    # CPU 測試
    try:
        cpu_tensor = torch.randn(3, 3, device='cpu')
        print("✅ CPU tensor 建立成功")
    except Exception as e:
        print(f"❌ CPU tensor 建立失敗: {e}")
    
    # GPU 測試
    if torch.cuda.is_available():
        try:
            gpu_tensor = torch.randn(3, 3, device='cuda:0')
            print("✅ GPU tensor 建立成功")
        except Exception as e:
            print(f"❌ GPU tensor 建立失敗: {e}")
    
    # NPU 測試
    npu_devices = []
    if hasattr(torch, 'npu') and torch.npu.is_available():
        npu_devices.append('npu:0')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        npu_devices.append('mps')
    
    for device in npu_devices:
        try:
            npu_tensor = torch.randn(3, 3, device=device)
            print(f"✅ {device.upper()} tensor 建立成功")
        except Exception as e:
            print(f"❌ {device.upper()} tensor 建立失敗: {e}")

if __name__ == '__main__':
    comprehensive_device_detection()
    test_device_creation()
    
    print("\n💡 如果需要 NPU 支援，請確認:")
    print("   1. 硬體是否支援 NPU")
    print("   2. 是否安裝了正確的 NPU 驅動")
    print("   3. 是否使用了支援 NPU 的 PyTorch 版本")
    print("   4. 環境變數是否正確設定")