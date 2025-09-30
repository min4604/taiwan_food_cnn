import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from pytorch_model import TaiwanFoodResNet50
from pytorch_data_loader import get_dataloaders
from tqdm import tqdm
import os

def check_gpu():
    """詳細檢查 GPU 可用性"""
    print("=" * 60)
    print("🔍 GPU 環境檢測")
    print("=" * 60)
    
    gpu_available = torch.cuda.is_available()
    
    if gpu_available:
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        
        print(f"✅ CUDA 可用: {torch.version.cuda}")
        print(f"🎯 GPU 數量: {device_count}")
        print(f"🚀 當前 GPU: {device_name}")
        
        # 顯示 GPU 記憶體資訊
        if device_count > 0:
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                memory = props.total_memory / 1024**3  # 轉換為 GB
                print(f"   GPU {i}: {props.name} ({memory:.1f} GB)")
                
        print(f"🔧 PyTorch 版本: {torch.__version__}")
    else:
        print("❌ CUDA 不可用")
        print("💡 建議:")
        print("   1. 確認您的電腦有 NVIDIA GPU")
        print("   2. 安裝 NVIDIA 驅動程式")
        print("   3. 執行 install_pytorch_gpu.bat 安裝 CUDA 版本的 PyTorch")
        print(f"🔧 當前 PyTorch 版本: {torch.__version__}")
    
    print("=" * 60)
    return gpu_available

def choose_model_to_continue():
    """選擇要繼續訓練的模型"""
    print("\n" + "=" * 60)
    print("🎯 選擇訓練模式")
    print("=" * 60)
    
    # 檢查是否有已保存的模型
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print("📁 建立 models 目錄")
        return None, 0
    
    # 獲取所有 .pth 模型檔案
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    
    if not model_files:
        print("📋 沒有找到已保存的模型，將從頭開始訓練")
        return None, 0
    
    # 顯示選項
    print("請選擇訓練模式:")
    print("0. 🆕 從頭開始訓練 (新模型)")
    
    # 顯示可用的模型檔案
    for i, model_file in enumerate(model_files, 1):
        # 嘗試從檔名中提取 epoch 資訊
        if 'epoch' in model_file:
            print(f"{i}. 🔄 繼續訓練: {model_file}")
        else:
            print(f"{i}. 🔄 繼續訓練: {model_file}")
    
    print("-" * 60)
    
    # 獲取用戶選擇
    while True:
        try:
            choice = input(f"請輸入選擇 (0-{len(model_files)}): ").strip()
            choice = int(choice)
            
            if choice == 0:
                print("✅ 選擇從頭開始訓練")
                return None, 0
            elif 1 <= choice <= len(model_files):
                selected_model = model_files[choice - 1]
                print(f"✅ 選擇繼續訓練: {selected_model}")
                
                # 嘗試從檔名中提取 epoch 數
                start_epoch = 0
                try:
                    if 'epoch' in selected_model:
                        # 假設檔名格式為 taiwan_food_resnet50_epoch10.pth
                        epoch_part = selected_model.split('epoch')[1].split('.')[0]
                        start_epoch = int(epoch_part)
                        print(f"📊 檢測到從第 {start_epoch} 個 epoch 繼續訓練")
                    else:
                        print("⚠️  無法從檔名檢測 epoch，將從 epoch 0 開始計數")
                except:
                    print("⚠️  無法解析 epoch 資訊，將從 epoch 0 開始計數")
                
                return os.path.join(models_dir, selected_model), start_epoch
            else:
                print(f"❌ 無效選擇，請輸入 0-{len(model_files)} 之間的數字")
        except ValueError:
            print("❌ 請輸入有效的數字")
        except KeyboardInterrupt:
            print("\n🚫 取消選擇，退出程式")
            exit(0)

def main():
    # 資料路徑
    train_csv = 'archive/tw_food_101/tw_food_101_train.csv'
    # 注意：測試集不參與訓練，僅用於最終評估
    test_csv = 'archive/tw_food_101/tw_food_101_test_list.csv'  
    train_img_dir = 'archive/tw_food_101/train'
    test_img_dir = 'archive/tw_food_101/test'
    num_classes = 101
    total_epochs = 50
    lr = 1e-3
    img_size = 224

    gpu = check_gpu()
    device = torch.device('cuda' if gpu else 'cpu')
    
    # 根據 GPU/CPU 調整批次大小
    if gpu:
        batch_size = 64  # GPU 可以處理更大的批次
        print(f"🚀 使用 GPU 訓練，批次大小: {batch_size}")
    else:
        batch_size = 16  # CPU 使用較小批次避免記憶體不足
        print(f"💻 使用 CPU 訓練，批次大小: {batch_size}")
        
    print()

    # 選擇要繼續訓練的模型
    model_path, start_epoch = choose_model_to_continue()
    remaining_epochs = total_epochs - start_epoch

    # DataLoader - 只使用訓練集進行訓練和驗證
    # 測試集不參與任何訓練過程
    train_loader, val_loader, _ = get_dataloaders(
        train_csv, test_csv, train_img_dir, test_img_dir, batch_size, img_size
    )
    
    print("📊 資料集資訊:")
    print(f"   訓練集大小: {len(train_loader.dataset)} (用於訓練)")
    print(f"   驗證集大小: {len(val_loader.dataset)} (從訓練集分割，用於驗證)")
    print(f"   ⚠️  測試集: 不參與訓練過程，僅供最終評估使用")
    print()

    # Model
    model = TaiwanFoodResNet50(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 載入已存在的模型（如果選擇了）
    if model_path:
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✅ 成功載入模型: {model_path}")
            print(f"🔄 從 epoch {start_epoch + 1} 開始繼續訓練")
        except Exception as e:
            print(f"❌ 載入模型失敗: {e}")
            print("🆕 將從頭開始訓練")
            start_epoch = 0
            remaining_epochs = total_epochs
    else:
        print("🆕 從頭開始訓練新模型")
        
    print(f"📈 訓練計畫: 從 epoch {start_epoch + 1} 到 epoch {total_epochs} (共 {remaining_epochs} 個 epochs)")
    print("=" * 60)

    # Training loop
    for epoch in range(start_epoch, total_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", ncols=100)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'acc': f'{correct/total:.4f}'
            })
        print(f"訓練 Loss: {running_loss/total:.4f} | 訓練 Acc: {correct/total:.4f}")

        # 驗證
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="驗證中", ncols=80, leave=False)
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)
                val_pbar.set_postfix({
                    'val_loss': f'{val_loss/val_total:.4f}',
                    'val_acc': f'{val_correct/val_total:.4f}'
                })
        
        print(f"驗證 Loss: {val_loss/val_total:.4f} | 驗證 Acc: {val_correct/val_total:.4f}")
        print("-" * 60)

        # 可加模型保存
        os.makedirs('models', exist_ok=True)
        model_filename = f'taiwan_food_resnet50_epoch{epoch+1}.pth'
        torch.save(model.state_dict(), f'models/{model_filename}')
        print(f"💾 模型已保存: models/{model_filename}")
        print()

if __name__ == '__main__':
    main()
