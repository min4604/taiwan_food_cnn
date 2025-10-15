"""
快速驗證 PyTorch GPU 環境是否正確配置
"""

import torch
import torch.nn as nn
import time
import os

def test_gpu():
    print("=" * 60)
    print("🔍 PyTorch GPU 環境測試")
    print("=" * 60)
    
    # 檢查 CUDA 是否可用
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        print("💡 建議安裝 GPU 版本的 PyTorch:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False
        
    # 顯示 CUDA 資訊
    print(f"✅ PyTorch 版本: {torch.__version__}")
    print(f"✅ CUDA 版本: {torch.version.cuda}")
    print(f"✅ cuDNN 版本: {torch.backends.cudnn.version()}")
    
    # 顯示 GPU 資訊
    device_count = torch.cuda.device_count()
    print(f"✅ 找到 {device_count} 個 GPU")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {props.name}")
        print(f"      記憶體: {props.total_memory / 1024**3:.1f} GB")
        print(f"      計算能力: {props.major}.{props.minor}")
    
    # 進行一次簡單的 GPU 計算測試
    print("\n📊 進行 GPU 計算測試...")
    device = torch.device('cuda')
    
    # 1. 測試基本操作
    try:
        # 建立測試張量
        x = torch.rand(1000, 1000, device=device)
        y = torch.rand(1000, 1000, device=device)
        
        # 確保 GPU 運算同步
        torch.cuda.synchronize()
        start_time = time.time()
        
        # 矩陣乘法測試
        z = torch.matmul(x, y)
        
        # 確保運算完成
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"✅ GPU 矩陣乘法測試通過，耗時: {gpu_time*1000:.2f} ms")
        print(f"   結果張量位於: {z.device}")
        
        # 比較 CPU 的速度
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        start_time = time.time()
        z_cpu = torch.matmul(x_cpu, y_cpu)
        cpu_time = time.time() - start_time
        
        print(f"✅ CPU 矩陣乘法測試，耗時: {cpu_time*1000:.2f} ms")
        print(f"✅ GPU 加速比: {cpu_time/gpu_time:.1f}x")
        
        if cpu_time / gpu_time < 1:
            print("⚠️  警告: GPU 運算速度慢於 CPU，可能有問題！")
        
    except Exception as e:
        print(f"❌ GPU 矩陣運算測試失敗: {e}")
        return False
    
    # 2. 測試 CNN 模型
    print("\n📊 進行 CNN 模型測試...")
    try:
        # 建立一個簡單的 CNN 模型
        class SimpleCNN(nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
                self.relu = nn.ReLU()
                self.maxpool = nn.MaxPool2d(kernel_size=2)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
                self.fc = nn.Linear(32 * 56 * 56, 10)
            
            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.maxpool(x)
                x = self.relu(self.conv2(x))
                x = self.maxpool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        # 將模型移到 GPU
        model = SimpleCNN().to(device)
        
        # 準備輸入資料
        batch_size = 16
        input_tensor = torch.rand(batch_size, 3, 224, 224, device=device)
        
        # 測試前向傳播
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            output = model(input_tensor)
        
        torch.cuda.synchronize()
        forward_time = time.time() - start_time
        
        print(f"✅ CNN 前向傳播測試通過，耗時: {forward_time*1000:.2f} ms")
        print(f"   輸入形狀: {input_tensor.shape}")
        print(f"   輸出形狀: {output.shape}")
        print(f"   模型位於: {next(model.parameters()).device}")
        
    except Exception as e:
        print(f"❌ CNN 模型測試失敗: {e}")
        return False
    
    # 3. 測試訓練迴圈
    print("\n📊 進行訓練迴圈測試...")
    try:
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 模擬訓練
        model.train()
        torch.cuda.synchronize()
        start_time = time.time()
        
        # 前向傳播
        output = model(input_tensor)
        
        # 計算損失
        target = torch.randint(0, 10, (batch_size,), device=device)
        loss = criterion(output, target)
        
        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize()
        train_time = time.time() - start_time
        
        print(f"✅ 訓練迴圈測試通過，耗時: {train_time*1000:.2f} ms")
        print(f"   損失值: {loss.item():.4f}")
        print(f"   損失位於: {loss.device}")
        
    except Exception as e:
        print(f"❌ 訓練迴圈測試失敗: {e}")
        return False
    
    # 4. GPU 記憶體測試
    print("\n📊 檢查 GPU 記憶體使用...")
    try:
        memory_allocated = torch.cuda.memory_allocated() / (1024**2)
        memory_reserved = torch.cuda.memory_reserved() / (1024**2)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        
        print(f"✅ GPU 記憶體已分配: {memory_allocated:.1f} MB")
        print(f"✅ GPU 記憶體已保留: {memory_reserved:.1f} MB")
        print(f"✅ GPU 總記憶體: {total_memory:.1f} MB")
        print(f"✅ 使用率: {memory_allocated/total_memory*100:.1f}%")
        
        # 釋放記憶體
        del model, input_tensor, output, x, y, z
        torch.cuda.empty_cache()
        
        memory_after = torch.cuda.memory_allocated() / (1024**2)
        print(f"✅ 釋放後記憶體使用: {memory_after:.1f} MB")
        
    except Exception as e:
        print(f"❌ 記憶體測試失敗: {e}")
    
    print("\n" + "=" * 60)
    print("✅ GPU 測試全部通過！")
    print("=" * 60)
    return True

def fix_gpu_issues():
    """嘗試修復常見的 GPU 問題"""
    print("\n🔧 嘗試修復 GPU 問題...")
    
    # 檢查是否安裝了 GPU 版本的 PyTorch
    if not torch.cuda.is_available():
        print("\n⚠️ 您可能未安裝 GPU 版本的 PyTorch")
        print("\n執行以下命令安裝 GPU 版本：")
        print("pip uninstall torch torchvision torchaudio")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return
    
    # 確認 CUDA_VISIBLE_DEVICES 環境變量
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
    print(f"CUDA_VISIBLE_DEVICES = {cuda_visible}")
    if cuda_visible == '':
        print("⚠️ CUDA_VISIBLE_DEVICES 為空，設置為 0")
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # 嘗試設置 cudnn benchmark
    torch.backends.cudnn.benchmark = True
    print("✅ 已啟用 cuDNN benchmark 模式")
    
    # 調整記憶體分配
    torch.cuda.empty_cache()
    print("✅ 已清理 CUDA 緩存")
    
    print("\n💡 建議：")
    print("1. 確認您的 PyTorch 版本支援您的 CUDA 版本")
    print("2. 更新 GPU 驅動程式到最新版本")
    print("3. 重啟電腦後再試")
    print("4. 使用較小批次大小 (batch_size)")
    print("5. 檢查 nvidia-smi 指令是否正常運行")
    print("\n運行以下指令檢查 GPU 使用情況：")
    print("nvidia-smi")

if __name__ == "__main__":
    test_result = test_gpu()
    
    if not test_result:
        fix_gpu_issues()
    else:
        print("\n🎉 您的環境已經可以使用 GPU 訓練！")
        print("💡 您可以執行：python train_pytorch.py")
        print("💡 記得監控 GPU 使用情況：nvidia-smi -l 1")