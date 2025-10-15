"""
GPU 使用檢查工具
在訓練開始後運行此腳本，可以實時監控 GPU 使用情況
"""

import torch
import time
import os

def check_gpu_status():
    """檢查當前 GPU 狀態"""
    print("=" * 70)
    print("🔍 GPU 使用狀態檢查")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用 - 無法使用 GPU")
        print("💡 請確認:")
        print("   1. 電腦有 NVIDIA GPU")
        print("   2. 已安裝 NVIDIA 驅動程式")
        print("   3. 已安裝 CUDA 版本的 PyTorch")
        return False
    
    print(f"✅ CUDA 版本: {torch.version.cuda}")
    print(f"✅ PyTorch 版本: {torch.__version__}")
    print(f"✅ cuDNN 版本: {torch.backends.cudnn.version()}")
    print()
    
    # 檢查每個 GPU
    device_count = torch.cuda.device_count()
    print(f"🎯 檢測到 {device_count} 個 GPU:")
    print("-" * 70)
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        memory_total = props.total_memory / 1024**3
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
        memory_free = memory_total - memory_reserved
        
        print(f"\n📱 GPU {i}: {props.name}")
        print(f"   總記憶體: {memory_total:.2f} GB")
        print(f"   已分配: {memory_allocated:.2f} GB ({memory_allocated/memory_total*100:.1f}%)")
        print(f"   已保留: {memory_reserved:.2f} GB ({memory_reserved/memory_total*100:.1f}%)")
        print(f"   可用: {memory_free:.2f} GB ({memory_free/memory_total*100:.1f}%)")
        
        # 檢查是否有張量在 GPU 上
        if memory_allocated > 0:
            print(f"   ✅ GPU 正在被使用 (已分配 {memory_allocated:.2f} GB)")
        else:
            print(f"   ⚠️  GPU 未被使用 (無分配記憶體)")
    
    print("\n" + "=" * 70)
    return True

def test_gpu_computation():
    """測試 GPU 計算功能"""
    if not torch.cuda.is_available():
        print("❌ 無法進行 GPU 測試")
        return
    
    print("\n🧪 GPU 計算測試")
    print("=" * 70)
    
    device = torch.device('cuda')
    
    # 測試 1: 簡單的張量運算
    print("\n測試 1: 基本張量運算")
    try:
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        
        start_time = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()  # 等待 GPU 完成
        gpu_time = time.time() - start_time
        
        print(f"   ✅ GPU 矩陣乘法完成")
        print(f"   ⏱️  耗時: {gpu_time*1000:.2f} ms")
        print(f"   📊 結果形狀: {z.shape}")
        print(f"   🎯 結果設備: {z.device}")
        
    except Exception as e:
        print(f"   ❌ GPU 計算失敗: {e}")
        return
    
    # 測試 2: CNN 模型推理
    print("\n測試 2: CNN 模型推理")
    try:
        import torch.nn as nn
        
        # 創建簡單的 CNN
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10)
        ).to(device)
        
        # 創建測試輸入
        test_input = torch.randn(32, 3, 224, 224).to(device)
        
        start_time = time.time()
        with torch.no_grad():
            output = model(test_input)
        torch.cuda.synchronize()
        inference_time = time.time() - start_time
        
        print(f"   ✅ CNN 推理完成")
        print(f"   ⏱️  耗時: {inference_time*1000:.2f} ms")
        print(f"   📊 輸入形狀: {test_input.shape}")
        print(f"   📊 輸出形狀: {output.shape}")
        print(f"   🎯 模型設備: {next(model.parameters()).device}")
        
    except Exception as e:
        print(f"   ❌ CNN 推理失敗: {e}")
    
    print("\n" + "=" * 70)

def monitor_gpu_realtime(duration=10, interval=1):
    """實時監控 GPU 使用情況"""
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用，無法監控")
        return
    
    print(f"\n📊 實時 GPU 監控 (持續 {duration} 秒)")
    print("=" * 70)
    print(f"{'時間':<10} {'GPU':<5} {'記憶體使用':<15} {'使用率':<10} {'溫度':<10}")
    print("-" * 70)
    
    try:
        for i in range(duration):
            for gpu_id in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                utilization = memory_allocated / memory_total * 100
                
                print(f"{i+1:>3}/{duration:<5} GPU{gpu_id:<2} "
                      f"{memory_allocated:>5.2f}/{memory_total:<5.2f} GB "
                      f"{utilization:>6.1f}%", end="")
                
                if memory_allocated > 0.1:
                    print(" ✅ 使用中")
                else:
                    print(" ⚠️  閒置")
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  監控已停止")
    
    print("=" * 70)

def verify_training_gpu_usage():
    """驗證訓練腳本是否正確設置 GPU"""
    print("\n🔍 訓練腳本 GPU 配置檢查")
    print("=" * 70)
    
    # 檢查 train_pytorch.py
    train_file = 'train_pytorch.py'
    if not os.path.exists(train_file):
        print(f"❌ 找不到 {train_file}")
        return
    
    with open(train_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = {
        'device = torch.device': '✅ 設備選擇代碼存在',
        '.to(device)': '✅ 模型移至設備代碼存在',
        'images.to(device)': '✅ 數據移至設備代碼存在',
        'labels.to(device)': '✅ 標籤移至設備代碼存在',
        'torch.cuda.is_available()': '✅ GPU 檢測代碼存在',
    }
    
    print(f"\n檢查 {train_file}:")
    for check, message in checks.items():
        if check in content:
            print(f"   {message}")
        else:
            print(f"   ❌ 缺少: {check}")
    
    print("\n" + "=" * 70)

def main():
    print("\n🚀 GPU 使用狀態完整檢查\n")
    
    # 1. 檢查 GPU 狀態
    gpu_available = check_gpu_status()
    
    if not gpu_available:
        print("\n❌ GPU 不可用，無法繼續測試")
        return
    
    # 2. 驗證訓練腳本配置
    verify_training_gpu_usage()
    
    # 3. 測試 GPU 計算
    test_gpu_computation()
    
    # 4. 詢問是否進行實時監控
    print("\n💡 建議:")
    print("   1. 如果訓練尚未開始，請先運行: python train_pytorch.py")
    print("   2. 訓練開始後，重新運行此腳本查看 GPU 使用情況")
    print("   3. 可以使用 'nvidia-smi' 命令查看詳細 GPU 狀態")
    
    try:
        choice = input("\n是否進行 10 秒實時 GPU 監控? (y/n) [n]: ").strip().lower()
        if choice == 'y':
            monitor_gpu_realtime(duration=10, interval=1)
    except KeyboardInterrupt:
        print("\n\n👋 檢查完成")

if __name__ == '__main__':
    main()
