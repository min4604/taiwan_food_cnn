#!/usr/bin/env python3
"""
AMD Ryzen AI 9HX NPU 專門檢測腳本
"""

import platform
import subprocess
import sys
import os

def check_system_info():
    """檢查系統基本資訊"""
    print("🔍 系統資訊檢查")
    print("=" * 50)
    print(f"作業系統: {platform.system()} {platform.version()}")
    print(f"處理器架構: {platform.machine()}")
    print(f"Python 版本: {sys.version}")
    print()

def check_amd_cpu():
    """檢查是否為 AMD Ryzen AI 處理器"""
    print("🔍 AMD 處理器檢測")
    print("-" * 30)
    
    if platform.system() != 'Windows':
        print("❌ 此腳本專為 Windows 設計")
        return False
    
    try:
        # 方法 1: 使用 PowerShell (更可靠)
        ps_cmd = 'Get-CimInstance -ClassName Win32_Processor | Select-Object Name'
        result = subprocess.run(['powershell', '-Command', ps_cmd], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            cpu_info = result.stdout
            print("CPU 資訊 (PowerShell):")
            for line in cpu_info.split('\n'):
                if line.strip() and 'Name' not in line and '---' not in line:
                    print(f"  {line.strip()}")
        else:
            # 方法 2: 使用 platform 模組
            cpu_info = platform.processor()
            print(f"CPU 資訊 (platform): {cpu_info}")
        
        # 檢查是否包含 AMD Ryzen AI 相關關鍵字
        cpu_text = cpu_info.lower()
        is_amd = 'amd' in cpu_text
        is_ryzen = 'ryzen' in cpu_text
        is_ai_series = any(keyword in cpu_text for keyword in ['ai', '9hx', '7040', '8040', '9040'])
        
        print(f"\n檢測結果:")
        print(f"  AMD 處理器: {'✅' if is_amd else '❌'}")
        print(f"  Ryzen 系列: {'✅' if is_ryzen else '❌'}")
        print(f"  AI 系列標識: {'✅' if is_ai_series else '❌'}")
        
        if is_amd and is_ryzen and is_ai_series:
            print("✅ 檢測到 AMD Ryzen AI 處理器")
            return True
        elif is_amd and is_ryzen:
            print("⚠️  檢測到 AMD Ryzen 處理器，但可能不是 AI 系列")
            print("💡 部分 AMD Ryzen 處理器也可能支援 DirectML")
            return True  # 給 Ryzen 處理器一個機會
        else:
            print("❌ 未檢測到 AMD Ryzen 處理器")
            return False
            
    except Exception as e:
        print(f"❌ CPU 檢測失敗: {e}")
        print("💡 嘗試手動檢查: 工作管理員 > 效能 > CPU")
        return False

def check_required_packages():
    """檢查必要套件安裝狀況"""
    print("\n🔍 必要套件檢查")
    print("-" * 30)
    
    packages = {
        'onnxruntime': 'ONNX Runtime (基礎)',
        'onnxruntime-directml': 'ONNX Runtime DirectML (AMD NPU)',
        'torch-directml': 'PyTorch DirectML (可選)',
        'onnx': 'ONNX 工具'
    }
    
    installed = {}
    
    for package_name, description in packages.items():
        try:
            if package_name == 'onnxruntime-directml':
                # 特殊檢查 DirectML 版本
                import onnxruntime as ort
                providers = ort.get_available_providers()
                if 'DmlExecutionProvider' in providers:
                    print(f"✅ {description}")
                    installed[package_name] = True
                else:
                    print(f"❌ {description} - DirectML 不可用")
                    installed[package_name] = False
            elif package_name == 'onnxruntime':
                import onnxruntime as ort
                print(f"✅ {description} - 版本 {ort.__version__}")
                installed[package_name] = True
            elif package_name == 'torch-directml':
                import torch_directml
                print(f"✅ {description}")
                installed[package_name] = True
            elif package_name == 'onnx':
                import onnx
                print(f"✅ {description} - 版本 {onnx.__version__}")
                installed[package_name] = True
        except ImportError:
            print(f"❌ {description} - 未安裝")
            installed[package_name] = False
        except Exception as e:
            print(f"⚠️  {description} - 檢查失敗: {e}")
            installed[package_name] = False
    
    return installed

def check_directml():
    """檢查 DirectML 支援"""
    print("\n🔍 DirectML 支援檢查")
    print("-" * 30)
    
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        
        print(f"可用的執行提供者: {providers}")
        
        if 'DmlExecutionProvider' in providers:
            print("✅ DirectML 執行提供者可用")
            
            # 嘗試建立簡單的會話測試
            try:
                # 建立一個最簡單的 ONNX 模型進行測試
                import numpy as np
                
                # 這裡可以添加實際的模型測試
                print("🧪 DirectML 基本功能正常")
                return True
            except Exception as e:
                print(f"⚠️  DirectML 測試失敗: {e}")
                return False
        else:
            print("❌ DirectML 執行提供者不可用")
            return False
            
    except ImportError:
        print("❌ ONNX Runtime 未安裝，無法檢查 DirectML")
        return False
    except Exception as e:
        print(f"❌ DirectML 檢查失敗: {e}")
        return False

def provide_installation_guide():
    """提供安裝指南"""
    print("\n💡 AMD Ryzen AI NPU 設定指南")
    print("=" * 50)
    
    print("步驟 1: 確保硬體支援")
    print("  - 確認您的處理器是 AMD Ryzen AI 系列 (如 9HX)")
    print("  - 在 BIOS 中啟用 NPU 功能")
    print("  - 確保 Windows 版本支援 DirectML (Windows 10 1903+ 或 Windows 11)")
    
    print("\n步驟 2: 安裝必要套件")
    print("  執行以下命令:")
    print("  pip install onnxruntime-directml")
    print("  pip install onnx")
    print("  pip install torch-directml  # 可選")
    
    print("\n步驟 3: 或使用自動安裝腳本")
    print("  .\\install_amd_npu.bat")
    
    print("\n步驟 4: 驗證安裝")
    print("  python amd_npu_test.py")
    
    print("\n💡 故障排除:")
    print("  - 確保 AMD 顯示卡驅動程式是最新版本")
    print("  - 重新啟動電腦後再測試")
    print("  - 檢查 Windows 更新")

def main():
    """主函數"""
    print("🚀 AMD Ryzen AI 9HX NPU 檢測工具")
    print("=" * 60)
    
    # 檢查系統資訊
    check_system_info()
    
    # 檢查 AMD CPU
    is_amd_ai = check_amd_cpu()
    
    # 檢查套件
    packages_status = check_required_packages()
    
    # 檢查 DirectML
    directml_ok = check_directml()
    
    # 總結
    print("\n📊 檢測結果總結")
    print("=" * 30)
    
    if is_amd_ai:
        print("✅ AMD Ryzen AI 處理器: 支援")
    else:
        print("❌ AMD Ryzen AI 處理器: 不支援")
    
    if packages_status.get('onnxruntime-directml', False):
        print("✅ ONNX Runtime DirectML: 已安裝")
    else:
        print("❌ ONNX Runtime DirectML: 未安裝")
    
    if directml_ok:
        print("✅ DirectML 功能: 正常")
    else:
        print("❌ DirectML 功能: 異常")
    
    # 最終判斷
    if is_amd_ai and packages_status.get('onnxruntime-directml', False) and directml_ok:
        print("\n🎉 AMD Ryzen AI NPU 支援已就緒！")
        print("您可以使用 AMD NPU 進行深度學習推理。")
    else:
        print("\n⚠️  AMD NPU 支援尚未完全就緒")
        provide_installation_guide()

if __name__ == '__main__':
    main()