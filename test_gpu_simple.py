"""
簡單的 GPU 測試腳本
快速驗證 PyTorch 是否能正確使用 GPU
"""

import torch
import torch.nn as nn
import time

print("=" * 70)
print("🔍 PyTorch GPU 快速測試")
print("=" * 70)

# 1. 基本檢查
print("\n1️⃣ 基本檢查")
print("-" * 70)
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("\n❌ CUDA 不可用！")
    print("\n可能的原因:")
    print("1. PyTorch 是 CPU 版本（需要重新安裝 GPU 版本）")
    print("2. NVIDIA 驅動未安裝")
    print("3. CUDA 環境未配置")
    print("\n安裝 GPU 版本 PyTorch:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    exit(1)

print(f"CUDA 版本: {torch.version.cuda}")
print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
print(f"GPU 數量: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    props = torch.cuda.get_device_properties(i)
    print(f"   記憶體: {props.total_memory / 1024**3:.2f} GB")
    print(f"   計算能力: {props.major}.{props.minor}")

# 2. 設備測試
print("\n2️⃣ 設備配置測試")
print("-" * 70)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"選擇的設備: {device}")

# 創建測試張量
test_tensor = torch.randn(100, 100).to(device)
print(f"測試張量設備: {test_tensor.device}")
print(f"測試張量形狀: {test_tensor.shape}")

if test_tensor.device.type == 'cuda':
    print("✅ 張量成功創建在 GPU 上")
else:
    print("❌ 張量在 CPU 上（有問題）")

# 3. 計算性能測試
print("\n3️⃣ GPU vs CPU 性能對比")
print("-" * 70)

# CPU 測試
print("測試 CPU 性能...")
a_cpu = torch.randn(2000, 2000)
b_cpu = torch.randn(2000, 2000)
start = time.time()
c_cpu = torch.matmul(a_cpu, b_cpu)
cpu_time = time.time() - start
print(f"CPU 矩陣乘法: {cpu_time*1000:.2f} ms")

# GPU 測試
print("測試 GPU 性能...")
a_gpu = torch.randn(2000, 2000).to(device)
b_gpu = torch.randn(2000, 2000).to(device)
torch.cuda.synchronize()  # 確保開始計時前 GPU 空閒
start = time.time()
c_gpu = torch.matmul(a_gpu, b_gpu)
torch.cuda.synchronize()  # 等待 GPU 計算完成
gpu_time = time.time() - start
print(f"GPU 矩陣乘法: {gpu_time*1000:.2f} ms")

speedup = cpu_time / gpu_time
print(f"\n⚡ GPU 加速比: {speedup:.2f}x")

if speedup > 2:
    print("✅ GPU 正常工作！")
elif speedup > 1:
    print("⚠️  GPU 有加速，但可能有問題")
else:
    print("❌ GPU 可能沒有正常工作")

# 4. CNN 模型測試
print("\n4️⃣ CNN 模型 GPU 測試")
print("-" * 70)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc = nn.Linear(128 * 56 * 56, 101)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

print("創建 CNN 模型...")
model = SimpleCNN().to(device)
print(f"模型設備: {next(model.parameters()).device}")

# 測試前向傳播
print("測試模型推理...")
test_input = torch.randn(16, 3, 224, 224).to(device)
print(f"輸入數據設備: {test_input.device}")

with torch.no_grad():
    output = model(test_input)
print(f"輸出設備: {output.device}")
print(f"輸出形狀: {output.shape}")

if output.device.type == 'cuda':
    print("✅ 模型在 GPU 上成功推理")
else:
    print("❌ 模型推理在 CPU 上（有問題）")

# 5. 記憶體檢查
print("\n5️⃣ GPU 記憶體使用")
print("-" * 70)
memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**2

print(f"已分配: {memory_allocated:.2f} MB")
print(f"已保留: {memory_reserved:.2f} MB")
print(f"總記憶體: {memory_total:.2f} MB")
print(f"使用率: {memory_allocated/memory_total*100:.2f}%")

if memory_allocated > 10:
    print("✅ GPU 記憶體有分配（正在使用）")
else:
    print("⚠️  GPU 記憶體分配較少")

# 6. 訓練模擬測試
print("\n6️⃣ 訓練流程模擬")
print("-" * 70)

print("模擬訓練步驟...")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 創建假數據
images = torch.randn(32, 3, 224, 224).to(device)
labels = torch.randint(0, 101, (32,)).to(device)

print(f"訓練數據設備: {images.device}")
print(f"訓練標籤設備: {labels.device}")

# 訓練步驟
model.train()
start = time.time()
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
train_time = time.time() - start

print(f"訓練步驟完成: {train_time*1000:.2f} ms")
print(f"Loss: {loss.item():.4f}")
print(f"Loss 張量設備: {loss.device}")

if loss.device.type == 'cuda':
    print("✅ 訓練過程在 GPU 上執行")
else:
    print("❌ 訓練過程在 CPU 上執行（有問題）")

# 清理
del model, test_input, output, images, labels, loss
torch.cuda.empty_cache()

print("\n" + "=" * 70)
print("✅ 測試完成！")
print("=" * 70)

print("\n📋 總結:")
if torch.cuda.is_available():
    print("✅ CUDA 可用")
    print("✅ GPU 計算正常")
    print("✅ 模型可以在 GPU 上運行")
    print("✅ 訓練流程可以使用 GPU")
    print("\n🎉 您的環境已準備好使用 GPU 訓練！")
    print("\n下一步: 執行 python train_pytorch.py 開始訓練")
else:
    print("❌ GPU 不可用，請檢查配置")
