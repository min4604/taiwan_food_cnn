"""
PyTorch GPU 驅動問題診斷工具
此腳本檢測 GPU 驅動、兼容性和性能問題，特別是 CUDA 與 PyTorch 的配合問題
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def print_section(title):
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)

def check_cuda_availability():
    print_header("📋 基本 CUDA 信息")
    
    # 檢查 CUDA 是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 是否可用: {'✅' if cuda_available else '❌'} {cuda_available}")
    
    if not cuda_available:
        print("\n❌ 無法使用 CUDA！")
        print("可能的原因:")
        print("1. NVIDIA 驅動未安裝或過期")
        print("2. CUDA 工具包未安裝")
        print("3. PyTorch 安裝的是 CPU 版本")
        print("\n解決方案:")
        print("1. 更新 NVIDIA 驅動到最新版本")
        print("2. 重新安裝帶 CUDA 支持的 PyTorch 版本:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False

    # CUDA 信息
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"cuDNN 版本: {torch.backends.cudnn.version() if hasattr(torch.backends.cudnn, 'version') else '未知'}")
    
    # GPU 信息
    device_count = torch.cuda.device_count()
    print(f"可用 GPU 數量: {device_count}")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        name = torch.cuda.get_device_name(i)
        mem_gb = props.total_memory / 1024**3
        print(f"\nGPU {i}: {name}")
        print(f"  總記憶體: {mem_gb:.1f} GB")
        print(f"  CUDA 計算能力: {props.major}.{props.minor}")
        print(f"  多處理器數量: {props.multi_processor_count}")
    
    return True

def verify_cuda_operations():
    print_header("🧪 CUDA 操作測試")
    
    try:
        # 創建測試張量
        print("創建 GPU 張量...", end="")
        x = torch.randn(100, 100, device="cuda")
        print(" ✅")
        
        # 測試基本運算
        print("測試 GPU 運算...", end="")
        y = x + 2
        z = y * y
        print(f" ✅ (結果在 {z.device})")
        
        # 測試 GPU 和 CPU 張量互轉
        print("測試 GPU<->CPU 傳輸...", end="")
        z_cpu = z.cpu()
        z_gpu = z_cpu.cuda()
        print(f" ✅ (結果在 {z_gpu.device})")
        
        return True
        
    except Exception as e:
        print(f"\n❌ CUDA 操作測試失敗: {e}")
        print("\n可能的原因:")
        print("1. CUDA 驅動與 PyTorch 版本不兼容")
        print("2. GPU 記憶體不足")
        print("3. GPU 硬體問題")
        return False

def measure_performance_comparison():
    print_header("⚡ GPU vs CPU 性能對比")
    
    sizes = [(1000, 1000), (2000, 2000), (4000, 4000)]
    
    for size in sizes:
        m, n = size
        print(f"\n矩陣大小: {m}x{n}")
        
        # CPU 測試
        try:
            a_cpu = torch.randn(m, n)
            b_cpu = torch.randn(m, n)
            
            # 預熱
            _ = torch.matmul(a_cpu, b_cpu)
            
            # 計時
            start = time.time()
            c_cpu = torch.matmul(a_cpu, b_cpu)
            cpu_time = time.time() - start
            print(f"CPU 時間: {cpu_time*1000:.2f} ms")
        except Exception as e:
            print(f"CPU 測試失敗: {e}")
            continue
        
        # GPU 測試
        try:
            a_gpu = torch.randn(m, n, device="cuda")
            b_gpu = torch.randn(m, n, device="cuda")
            
            # 預熱
            _ = torch.matmul(a_gpu, b_gpu)
            torch.cuda.synchronize()
            
            # 計時
            start = time.time()
            c_gpu = torch.matmul(a_gpu, b_gpu)
            torch.cuda.synchronize()  # 等待 GPU 完成
            gpu_time = time.time() - start
            print(f"GPU 時間: {gpu_time*1000:.2f} ms")
            
            # 計算加速比
            speedup = cpu_time / gpu_time
            print(f"加速比: {speedup:.2f}x")
            
            if speedup < 1:
                print("⚠️  警告: GPU 比 CPU 慢！可能存在驅動問題")
                if m < 2000:
                    print("     (小矩陣測試對 GPU 不利，稍後將嘗試更大矩陣)")
                else:
                    print("     即使對大矩陣測試，GPU 仍較慢，這表明存在嚴重問題")
            elif speedup < 5 and m >= 2000:
                print("⚠️  警告: GPU 加速比較低，可能未充分利用")
            else:
                print("✅ GPU 加速正常")
                
        except Exception as e:
            print(f"GPU 測試失敗: {e}")
    
def test_cnn_training():
    print_header("🔄 CNN 訓練測試")
    
    # 定義簡單 CNN
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.fc = nn.Linear(32 * 56 * 56, 10)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    batch_size = 8
    
    try:
        # 創建模型
        print("創建 CNN 模型...")
        model_cpu = SimpleCNN()
        model_gpu = SimpleCNN().cuda()
        
        # 創建優化器
        optimizer_cpu = optim.Adam(model_cpu.parameters(), lr=0.001)
        optimizer_gpu = optim.Adam(model_gpu.parameters(), lr=0.001)
        
        # 創建損失函數
        criterion = nn.CrossEntropyLoss()
        criterion_gpu = nn.CrossEntropyLoss().cuda()
        
        # 創建假數據
        print("創建測試數據...")
        inputs_cpu = torch.randn(batch_size, 3, 224, 224)
        targets_cpu = torch.randint(0, 10, (batch_size,))
        inputs_gpu = inputs_cpu.cuda()
        targets_gpu = targets_cpu.cuda()
        
        # CPU 訓練步驟
        print("\n測試 CPU 訓練...")
        model_cpu.train()
        start = time.time()
        
        optimizer_cpu.zero_grad()
        outputs_cpu = model_cpu(inputs_cpu)
        loss_cpu = criterion(outputs_cpu, targets_cpu)
        loss_cpu.backward()
        optimizer_cpu.step()
        
        cpu_time = time.time() - start
        print(f"CPU 訓練步驟: {cpu_time*1000:.2f} ms")
        print(f"Loss: {loss_cpu.item():.4f}")
        
        # GPU 訓練步驟
        print("\n測試 GPU 訓練...")
        model_gpu.train()
        torch.cuda.synchronize()
        start = time.time()
        
        optimizer_gpu.zero_grad()
        outputs_gpu = model_gpu(inputs_gpu)
        loss_gpu = criterion_gpu(outputs_gpu, targets_gpu)
        loss_gpu.backward()
        optimizer_gpu.step()
        
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        print(f"GPU 訓練步驟: {gpu_time*1000:.2f} ms")
        print(f"Loss: {loss_gpu.item():.4f}")
        
        # 比較
        speedup = cpu_time / gpu_time
        print(f"\nGPU 加速比: {speedup:.2f}x")
        
        if speedup < 1:
            print("❌ GPU 訓練比 CPU 慢！這是嚴重問題")
            print("   可能的原因:")
            print("   1. GPU 驅動問題")
            print("   2. PyTorch 和 CUDA 版本不匹配")
            print("   3. GPU 功率限制或過熱")
        elif speedup < 3:
            print("⚠️  GPU 加速比較低，可能未充分利用")
        else:
            print("✅ GPU 訓練正常加速")
        
        return speedup > 1
        
    except Exception as e:
        print(f"\n❌ CNN 訓練測試失敗: {e}")
        return False

def check_cudnn():
    print_header("⚙️ cuDNN 配置檢查")
    
    try:
        print(f"cuDNN 可用: {'✅' if torch.backends.cudnn.is_available() else '❌'}")
        print(f"cuDNN 已啟用: {'✅' if torch.backends.cudnn.enabled else '❌'}")
        print(f"cuDNN benchmark 模式: {'✅' if torch.backends.cudnn.benchmark else '❌'}")
        print(f"cuDNN 確定性模式: {'✅' if torch.backends.cudnn.deterministic else '❌'}")
        
        if not torch.backends.cudnn.is_available():
            print("\n❌ cuDNN 不可用！這會嚴重影響卷積操作性能")
            print("可能的原因:")
            print("1. cuDNN 未安裝")
            print("2. cuDNN 與 CUDA 版本不匹配")
            print("\n解決方案:")
            print("重新安裝帶 cuDNN 支持的 PyTorch:")
            print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            return False
            
        if not torch.backends.cudnn.enabled:
            print("\n⚠️  cuDNN 未啟用！將顯著降低性能")
            print("解決方案:")
            print("在代碼中添加: torch.backends.cudnn.enabled = True")
            
        if not torch.backends.cudnn.benchmark:
            print("\n⚠️  cuDNN benchmark 模式未開啟")
            print("對於固定尺寸輸入的訓練，建議開啟 benchmark 模式來提高性能")
            print("解決方案:")
            print("在代碼中添加: torch.backends.cudnn.benchmark = True")
            
        return torch.backends.cudnn.is_available() and torch.backends.cudnn.enabled
    
    except Exception as e:
        print(f"\n❌ cuDNN 檢查失敗: {e}")
        return False

def check_memory_issues():
    print_header("💾 GPU 記憶體測試")
    
    try:
        # 獲取 GPU 總記憶體
        total_memory = torch.cuda.get_device_properties(0).total_memory
        total_memory_gb = total_memory / 1024**3
        print(f"GPU 總記憶體: {total_memory_gb:.2f} GB")
        
        # 當前記憶體使用
        memory_allocated = torch.cuda.memory_allocated(0)
        memory_reserved = torch.cuda.memory_reserved(0)
        print(f"已分配記憶體: {memory_allocated/1024**2:.2f} MB")
        print(f"已保留記憶體: {memory_reserved/1024**2:.2f} MB")
        
        # 嘗試分配並釋放記憶體
        print("\n測試記憶體分配...")
        try:
            # 嘗試分配 1GB 記憶體
            test_size = min(1 * 1024**3, int(total_memory * 0.7))  # 分配 1GB 或 70% 記憶體（取小者）
            test_size_mb = test_size / 1024**2
            print(f"嘗試分配 {test_size_mb:.2f} MB 記憶體...", end="")
            
            x = torch.empty(test_size // 4, device='cuda')  # 4 bytes per float
            print(" ✅")
            
            del x
            torch.cuda.empty_cache()
            print("記憶體釋放成功 ✅")
            
            # 再次檢查記憶體
            memory_allocated_after = torch.cuda.memory_allocated(0)
            print(f"釋放後已分配記憶體: {memory_allocated_after/1024**2:.2f} MB")
        
        except Exception as e:
            print(f"\n❌ 記憶體分配失敗: {e}")
            print("可能的原因:")
            print("1. GPU 記憶體不足")
            print("2. 其他程序佔用了 GPU 記憶體")
            print("3. CUDA 驅動問題")
            return False
            
        return True
        
    except Exception as e:
        print(f"\n❌ GPU 記憶體測試失敗: {e}")
        return False

def check_train_pytorch_file():
    print_header("📄 train_pytorch.py 檢查")
    
    file_path = "train_pytorch.py"
    if not os.path.exists(file_path):
        print(f"❌ 找不到 {file_path}")
        return False
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # 檢查關鍵 GPU 代碼
        checks = {
            "torch.cuda.is_available()": "CUDA 可用性檢查",
            "device = torch.device": "設備選擇代碼",
            ".to(device)": "將模型/數據移至設備",
            "torch.backends.cudnn": "cuDNN 配置"
        }
        
        print("檢查關鍵 GPU 代碼:")
        for code, desc in checks.items():
            if code in content:
                print(f"✅ {desc} 存在")
            else:
                print(f"❌ {desc} 缺失")
        
        # 檢查潛在問題
        issues = {
            ".cuda()": "直接使用 .cuda() 而非 .to(device)",
            "device='cuda'": "直接硬編碼 device='cuda'",
            "torch.device('cpu')": "可能強制使用 CPU"
        }
        
        found_issues = False
        print("\n檢查潛在問題:")
        for issue, desc in issues.items():
            if issue in content:
                print(f"⚠️  {desc}")
                found_issues = True
                
        if not found_issues:
            print("✅ 未發現明顯問題")
            
        return True
        
    except Exception as e:
        print(f"❌ 檢查失敗: {e}")
        return False

def fix_train_pytorch():
    print_header("🔧 修復建議")
    
    print("基於測試結果，這裡是修復 GPU 訓練的建議：\n")
    
    print("1. 在 train_pytorch.py 開頭添加 GPU 強制檢測：")
    print("""
    # 強制啟用 CUDA 和 cuDNN
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    """)
    
    print("\n2. 在 main() 函數中使用明確的 GPU 設置：")
    print("""
    # 直接指定 GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 測試 GPU 功能
    if device.type == 'cuda':
        test_tensor = torch.randn(100, 100).to(device)
        result = test_tensor.sum().item()
        print(f"✅ GPU 測試成功: {test_tensor.device}")
    """)
    
    print("\n3. 在數據加載時添加 pin_memory 和 num_workers：")
    print("""
    # 數據加載器
    pin_memory = True if device.type == 'cuda' else False
    num_workers = 4 if device.type == 'cuda' else 0
    
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, 
        pin_memory=pin_memory, num_workers=num_workers
    )
    """)
    
    print("\n4. 確保正確將模型移至 GPU：")
    print("""
    # 模型初始化
    model = get_model(model_architecture, num_classes=num_classes)
    model = model.to(device)  # 先確保移至正確設備
    
    # 打印確認信息
    print(f"模型位於: {next(model.parameters()).device}")
    """)
    
    print("\n5. 在訓練迴圈前添加 GPU 記憶體使用監控：")
    print("""
    # 顯示 GPU 使用情況
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"GPU 記憶體: 已分配 {allocated:.1f} MB, 已保留 {reserved:.1f} MB")
    """)

def check_extensions_loaded():
    print_section("🔍 檢查 PyTorch CUDA 擴展")
    
    try:
        print("檢查 CUDA 算子是否能正確加載...")
        
        # 嘗試使用一些會觸發 CUDA 擴展的操作
        if torch.cuda.is_available():
            # 檢查 cuDNN 卷積
            x = torch.randn(2, 3, 10, 10).cuda()
            conv = nn.Conv2d(3, 5, 3).cuda()
            y = conv(x)
            
            # 檢查 CUDA RNN
            rnn = nn.GRU(10, 20, batch_first=True).cuda()
            h = torch.randn(1, 2, 10).cuda()
            y, _ = rnn(h)
            
            print("✅ CUDA 擴展測試通過")
            return True
            
    except Exception as e:
        print(f"❌ CUDA 擴展測試失敗: {e}")
        print("可能的原因:")
        print("1. CUDA 工具鏈安裝錯誤")
        print("2. PyTorch 編譯問題")
        return False

def prepare_batch_file():
    print_header("🛠️ 創建修復腳本")
    
    try:
        script_content = """@echo off
echo ====================================
echo PyTorch GPU 修復工具
echo ====================================
echo.

echo 步驟 1: 檢查 NVIDIA 驅動...
nvidia-smi
if %ERRORLEVEL% NEQ 0 (
    echo ❌ NVIDIA 驅動未安裝或有問題
    echo 請先安裝或更新 NVIDIA 驅動
    pause
    exit /b
)

echo.
echo 步驟 2: 卸載現有 PyTorch...
pip uninstall -y torch torchvision torchaudio
echo.

echo 步驟 3: 安裝 CUDA 版本的 PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo.

echo 步驟 4: 驗證安裝...
python -c "import torch; print('CUDA 可用:', torch.cuda.is_available()); print('PyTorch 版本:', torch.__version__); print('CUDA 版本:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
echo.

echo 步驟 5: 清理 PyTorch 快取...
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else print('CUDA 不可用')"
echo.

echo 修復完成！請嘗試重新運行 train_pytorch.py
pause
"""
        
        with open("fix_pytorch_gpu.bat", "w") as f:
            f.write(script_content)
        
        print("✅ 修復批處理檔案已創建: fix_pytorch_gpu.bat")
        print("   執行此檔案將卸載並重新安裝 PyTorch (CUDA 版本)")
    
    except Exception as e:
        print(f"❌ 創建修復腳本失敗: {e}")

def main():
    print_header("🚀 PyTorch GPU 診斷工具")
    print("此工具將全面檢測 PyTorch GPU 配置問題\n")
    
    import time
    from datetime import datetime
    
    print(f"執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")
    
    # 檢查階段
    cuda_ok = check_cuda_availability()
    
    if not cuda_ok:
        print("\n❌ CUDA 不可用，無法進行後續測試")
        prepare_batch_file()
        return

    ops_ok = verify_cuda_operations()
    
    if not ops_ok:
        print("\n❌ 基本 CUDA 操作失敗，無法進行後續測試")
        prepare_batch_file()
        return
    
    cudnn_ok = check_cudnn()
    measure_performance_comparison()
    memory_ok = check_memory_issues()
    training_ok = test_cnn_training()
    extensions_ok = check_extensions_loaded()
    
    # 檢查 train_pytorch.py
    file_ok = check_train_pytorch_file()
    
    # 總結
    print_header("📊 診斷總結")
    
    print(f"CUDA 基礎檢查: {'✅' if cuda_ok else '❌'}")
    print(f"CUDA 操作測試: {'✅' if ops_ok else '❌'}")
    print(f"cuDNN 配置檢查: {'✅' if cudnn_ok else '❌'}")
    print(f"GPU 記憶體測試: {'✅' if memory_ok else '❌'}")
    print(f"CNN 訓練測試: {'✅' if training_ok else '❌'}")
    print(f"CUDA 擴展測試: {'✅' if extensions_ok else '❌'}")
    
    if file_ok:
        print(f"train_pytorch.py 檢查: ✅")
    elif file_ok is False:
        print(f"train_pytorch.py 檢查: ⚠️  有問題")
    else:
        print(f"train_pytorch.py 檢查: ❓ 未檢查")
    
    # 建議修復方案
    if not all([cuda_ok, ops_ok, cudnn_ok, memory_ok, training_ok, extensions_ok]):
        fix_train_pytorch()
        prepare_batch_file()
        
        print("\n⚠️  診斷發現問題，請查看上述報告和修復建議")
    else:
        print("\n✅ 所有測試通過！")
        print("如果訓練仍然不使用 GPU，請查看以下建議:")
        fix_train_pytorch()
    
    # 創建修復批處理檔案
    prepare_batch_file()

if __name__ == "__main__":
    main()